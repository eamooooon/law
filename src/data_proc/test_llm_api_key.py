#!/usr/bin/env python3
"""
Quick API key/model availability test for chat-completions compatible endpoints.

Usage:
1) Default (reuse values from build_reasoning.py):
   python src/data_proc/test_llm_api_key.py
2) Override by CLI:
   python src/data_proc/test_llm_api_key.py --api-url http://host:port --api-key xxx --model yyy
"""

import argparse
import json
import os
import sys
from urllib import error, request
from urllib.parse import urlparse


try:
    import build_reasoning as cfg
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] failed to import build_reasoning.py: {exc}", file=sys.stderr)
    sys.exit(2)


DEFAULT_API_URL = getattr(cfg, "DEFAULT_API_URL", "http://101.33.45.201:8317")
DEFAULT_API_KEY = getattr(cfg, "DEFAULT_API_KEY", "sk-6Lw43uf1fvahDCTLpg0lVOQaLsVPrs3rN3XDtlCzdQFTr")
DEFAULT_MODEL = getattr(cfg, "DEFAULT_MODEL", "gpt-5.4")


def normalize_chat_url(api_url: str) -> str:
    """Allow users to provide base URL or full chat-completions URL."""
    raw = (api_url or "").strip()
    if not raw:
        raise ValueError("api_url is empty")

    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw

    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")

    if path.endswith("/v1/chat/completions"):
        return raw

    if path == "":
        return raw.rstrip("/") + "/v1/chat/completions"

    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether LLM API key/model are usable.")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Base URL or chat-completions URL.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--timeout", type=int, default=45, help="Request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max output tokens for the test call.")
    return parser.parse_args()


def run_probe(api_url: str, api_key: str, model: str, timeout: int, temperature: float, max_tokens: int) -> None:
    chat_url = normalize_chat_url(api_url)

    test_question = "请用2-3句话解释《欧洲人权公约》第6条的核心内容，并给一个简短案例场景。"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个专业、简洁的法律助手。"},
            {"role": "user", "content": test_question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        print("[FAIL] HTTP error")
        print(f"status: {exc.code}")
        print(f"reason: {exc.reason}")
        if err_body:
            print(f"body: {err_body[:1200]}")
        sys.exit(1)
    except error.URLError as exc:
        print("[FAIL] URL/network error")
        print(f"reason: {exc.reason}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] unexpected error")
        print(f"reason: {exc}")
        sys.exit(1)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        print("[FAIL] response is not JSON")
        print(f"http_status: {status}")
        print(f"body: {body[:1200]}")
        sys.exit(1)

    choices = parsed.get("choices", [])
    content = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        content = str(message.get("content", "")).strip()

    # Basic sanity check: key/model is considered usable only if we get a non-trivial answer.
    if not content or content.lower() == "ok" or len(content) < 12:
        print("[FAIL] request succeeded but reply looks abnormal")
        print(f"http_status: {status}")
        print(f"chat_url: {chat_url}")
        print(f"model: {model}")
        print(f"reply_preview: {content[:200]}")
        sys.exit(1)

    print("[PASS] endpoint reachable and request succeeded")
    print(f"http_status: {status}")
    print(f"chat_url: {chat_url}")
    print(f"model: {model}")
    print(f"question: {test_question}")
    print(f"reply_preview: {content[:200]}")


if __name__ == "__main__":
    args = parse_args()

    if not args.api_url:
        print("[FAIL] api_url is empty. Set DEFAULT_API_URL in build_reasoning.py or pass --api-url")
        sys.exit(2)
    if not args.model:
        print("[FAIL] model is empty. Set DEFAULT_MODEL in build_reasoning.py or pass --model")
        sys.exit(2)

    run_probe(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
