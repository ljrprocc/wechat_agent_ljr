from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL_PATHS = {
    "qwen2.5-0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}
DEFAULT_LOCAL_MODEL_PATHS = {
    "qwen2.5-0.5b": "./mlx/Qwen2.5-0.5B-Instruct-4bit",
    "qwen2.5-3b": "./mlx/Qwen2.5-3B-Instruct-4bit",
}

DEFAULT_SYSTEM_PROMPT = "你是一个本地部署的 Qwen 助手，请优先用中文回答并保持简洁。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standalone MLX smoke test for Qwen2.5 on Apple Silicon."
    )
    parser.add_argument(
        "--model-id",
        choices=sorted(DEFAULT_MODEL_PATHS.keys()),
        default=os.getenv("MLX_QWEN_MODEL_ID", "qwen2.5-0.5b"),
        help="Logical model id used to pick a default MLX model path.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MLX_QWEN_MODEL_PATH", ""),
        help="Override the MLX model repo id or local converted model path.",
    )
    parser.add_argument(
        "--prompt",
        default="请用三句话介绍你自己。",
        help="Prompt sent to the model.",
    )
    parser.add_argument(
        "--system",
        default=os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="System prompt used when the tokenizer supports chat templates.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MLX_QWEN_MAX_TOKENS", "128")),
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("MLX_QWEN_TEMPERATURE", "0.7")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("MLX_QWEN_TOP_P", "0.9")),
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--eos-token",
        default=os.getenv("MLX_QWEN_EOS_TOKEN", ""),
        help="Optional eos_token forwarded to tokenizer_config, useful for some Qwen MLX checkpoints.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the tokenizer config when required.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Let mlx-lm print token generation progress.",
    )
    return parser


def resolve_model_path(model_id: str, override: str) -> str:
    if override.strip():
        return override.strip()
    local_path = DEFAULT_LOCAL_MODEL_PATHS[model_id]
    if Path(local_path).exists():
        return local_path
    return DEFAULT_MODEL_PATHS[model_id]


def build_prompt(tokenizer: object, system_prompt: str, user_prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)

    if callable(apply_chat_template) and chat_template is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return user_prompt


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        print(
            "mlx-lm is not installed. Run `pip install -r requirements-mlx.txt` first.",
            file=sys.stderr,
        )
        return 1

    model_path = resolve_model_path(args.model_id, args.model)
    tokenizer_config: dict[str, object] = {}
    if args.eos_token.strip():
        tokenizer_config["eos_token"] = args.eos_token.strip()
    if args.trust_remote_code:
        tokenizer_config["trust_remote_code"] = True

    print(f"[mlx] loading model: {model_path}", file=sys.stderr)
    model, tokenizer = load(model_path, tokenizer_config=tokenizer_config or None)
    prompt = build_prompt(tokenizer, args.system, args.prompt)
    sampler = None
    if args.temperature > 0 or args.top_p > 0:
        sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    print(f"[mlx] prompt ready, generating with max_tokens={args.max_tokens}", file=sys.stderr)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=args.verbose,
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
