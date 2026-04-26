from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agent import build_agent_from_env


app = FastAPI(title="Local Qwen API")
agent = build_agent_from_env()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    model_id: str = "qwen2.5-0.5b"
    stream: bool = False
    debug: bool = False


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model_id: str
    stream: bool
    debug: dict[str, Any] | None = None


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "agent": agent.status(),
    }


@app.post("/api/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = agent.reply_with_debug(
            session_id=req.session_id,
            user_text=req.message,
            model_id=req.model_id,
            stream=req.stream,
            debug=req.debug,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        reply=result.reply,
        session_id=req.session_id,
        model_id=req.model_id,
        stream=False,
        debug=result.debug,
    )
