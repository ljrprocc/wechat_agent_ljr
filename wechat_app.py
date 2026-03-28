from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from wechatpy import create_reply as create_official_reply
from wechatpy import parse_message as parse_official_message
from wechatpy.crypto import WeChatCrypto as OfficialCrypto
from wechatpy.exceptions import InvalidAppIdException, InvalidSignatureException
from wechatpy.utils import check_signature
from wechatpy.work import create_reply as create_work_reply
from wechatpy.work import parse_message as parse_work_message
from wechatpy.work.crypto import WeChatCrypto as WorkCrypto

from wechat_agent import build_agent_from_env


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Qwen WeChat Agent")
agent = build_agent_from_env()


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing environment variable: {name}")
    return value


def _official_crypto() -> OfficialCrypto | None:
    token = _get_required_env("WECHAT_OFFICIAL_TOKEN")
    app_id = os.getenv("WECHAT_OFFICIAL_APP_ID", "").strip()
    aes_key = os.getenv("WECHAT_OFFICIAL_AES_KEY", "").strip()
    if not app_id or not aes_key:
        return None
    return OfficialCrypto(token, aes_key, app_id)


def _work_crypto() -> WorkCrypto:
    return WorkCrypto(
        _get_required_env("WECOM_TOKEN"),
        _get_required_env("WECOM_AES_KEY"),
        _get_required_env("WECOM_CORP_ID"),
    )


def _build_reply_text(platform: str, msg) -> str:
    msg_type = getattr(msg, "type", "")
    if msg_type == "text":
        session_id = f"{platform}:{msg.source}"
        return agent.reply(session_id, msg.content)

    event = getattr(msg, "event", "")
    if event in {"subscribe", "enter_agent"}:
        return "本地 Qwen Agent 已连接，直接发送文本即可开始对话。"

    return "当前最小版本只支持文本消息。"


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "agent": agent.status(),
            "official_configured": bool(os.getenv("WECHAT_OFFICIAL_TOKEN")),
            "work_configured": all(
                [
                    os.getenv("WECOM_TOKEN"),
                    os.getenv("WECOM_AES_KEY"),
                    os.getenv("WECOM_CORP_ID"),
                ]
            ),
        }
    )


@app.get("/wechat/official/callback")
def verify_official_callback(
    signature: str = "",
    timestamp: str = "",
    nonce: str = "",
    echostr: str = "",
    msg_signature: str = "",
) -> PlainTextResponse:
    token = _get_required_env("WECHAT_OFFICIAL_TOKEN")
    crypto = _official_crypto()

    try:
        if crypto:
            echo_text = crypto.check_signature(msg_signature, timestamp, nonce, echostr)
            return PlainTextResponse(echo_text)

        check_signature(token, signature, timestamp, nonce)
        return PlainTextResponse(echostr)
    except InvalidSignatureException as exc:
        raise HTTPException(status_code=403, detail="Invalid official signature") from exc


@app.post("/wechat/official/callback")
async def receive_official_callback(request: Request) -> PlainTextResponse:
    token = _get_required_env("WECHAT_OFFICIAL_TOKEN")
    crypto = _official_crypto()
    raw_body = (await request.body()).decode("utf-8")

    timestamp = request.query_params.get("timestamp", "")
    nonce = request.query_params.get("nonce", "")

    try:
        if crypto:
            msg_signature = request.query_params.get("msg_signature", "")
            raw_body = crypto.decrypt_message(raw_body, msg_signature, timestamp, nonce)
        else:
            signature = request.query_params.get("signature", "")
            if signature:
                check_signature(token, signature, timestamp, nonce)

        msg = parse_official_message(raw_body)
        reply_text = _build_reply_text("official", msg)
        reply_xml = create_official_reply(reply_text, msg).render()

        if crypto:
            encrypted_xml = crypto.encrypt_message(
                reply_xml,
                nonce or "nonce",
                timestamp or str(int(time.time())),
            )
            return PlainTextResponse(encrypted_xml, media_type="application/xml")

        return PlainTextResponse(reply_xml, media_type="application/xml")
    except (InvalidAppIdException, InvalidSignatureException) as exc:
        raise HTTPException(status_code=403, detail="Official callback validation failed") from exc
    except Exception as exc:  # pragma: no cover - keep callback failure visible in logs
        logger.exception("Official callback error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/wechat/work/callback")
def verify_work_callback(
    msg_signature: str = "",
    timestamp: str = "",
    nonce: str = "",
    echostr: str = "",
) -> PlainTextResponse:
    crypto = _work_crypto()

    try:
        echo_text = crypto.check_signature(msg_signature, timestamp, nonce, echostr)
        return PlainTextResponse(echo_text)
    except InvalidSignatureException as exc:
        raise HTTPException(status_code=403, detail="Invalid WeCom signature") from exc


@app.post("/wechat/work/callback")
async def receive_work_callback(request: Request) -> PlainTextResponse:
    crypto = _work_crypto()
    raw_body = (await request.body()).decode("utf-8")

    msg_signature = request.query_params.get("msg_signature", "")
    timestamp = request.query_params.get("timestamp", "")
    nonce = request.query_params.get("nonce", "")

    try:
        decrypted_xml = crypto.decrypt_message(raw_body, msg_signature, timestamp, nonce)
        msg = parse_work_message(decrypted_xml)
        reply_text = _build_reply_text("work", msg)
        reply_xml = create_work_reply(reply_text, msg).render()
        encrypted_xml = crypto.encrypt_message(
            reply_xml,
            nonce or "nonce",
            timestamp or str(int(time.time())),
        )
        return PlainTextResponse(encrypted_xml, media_type="application/xml")
    except InvalidSignatureException as exc:
        raise HTTPException(status_code=403, detail="WeCom callback validation failed") from exc
    except Exception as exc:  # pragma: no cover - keep callback failure visible in logs
        logger.exception("WeCom callback error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
