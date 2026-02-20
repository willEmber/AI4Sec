from __future__ import annotations

import json
import os
import sys
import time
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_dotenv(dotenv_path: Path) -> None:
    """Load KEY=VALUE pairs into environment (minimal .env parser).

    - Does NOT override existing environment variables.
    - Ignores blank lines and comments starting with '#'.
    - Supports optional single/double quotes around values.
    """

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (len(value) >= 2) and (
            (value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _join_base_url(base_url: str, path: str) -> str:
    base_url = base_url.strip()
    if not base_url:
        raise ValueError("LLM_BASEURL is empty")

    # Normalize: allow user to pass https://.../v1 or https://.../v1/
    if not base_url.endswith("/"):
        base_url += "/"

    return urllib.parse.urljoin(base_url, path.lstrip("/"))


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **headers,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            body_bytes = resp.read()
            body_text = body_bytes.decode("utf-8", errors="replace")
            return status, json.loads(body_text) if body_text else {}
    except (TimeoutError, socket.timeout):
        return 0, {"error": {"type": "timeout", "message": f"Read timed out after {timeout_s}s"}}
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return 0, {"error": {"type": "timeout", "message": f"Connection/read timed out after {timeout_s}s"}}
        return 0, {"error": {"type": "url_error", "message": str(e)}}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body) if body else {"error": {"message": "Empty error body"}}
        except json.JSONDecodeError:
            return e.code, {"error": {"message": body}}


def _get_json(url: str, headers: Dict[str, str], timeout_s: int) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "Accept": "application/json",
            **headers,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            body_bytes = resp.read()
            body_text = body_bytes.decode("utf-8", errors="replace")
            return status, json.loads(body_text) if body_text else {}
    except (TimeoutError, socket.timeout):
        return 0, {"error": {"type": "timeout", "message": f"Read timed out after {timeout_s}s"}}
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return 0, {"error": {"type": "timeout", "message": f"Connection/read timed out after {timeout_s}s"}}
        return 0, {"error": {"type": "url_error", "message": str(e)}}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body) if body else {"error": {"message": "Empty error body"}}
        except json.JSONDecodeError:
            return e.code, {"error": {"message": body}}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenAI-compatible API connectivity")
    parser.add_argument(
        "--mode",
        choices=["chat", "models"],
        default="chat",
        help="Request type to send: chat (POST /chat/completions) or models (GET /models)",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句话回复：连接测试成功",
        help="User prompt used for chat mode",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("LLM_TIMEOUT_S", "30")),
        help="Timeout seconds for the HTTP request",
    )
    args = parser.parse_args()

    # Load .env from repo root by default
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")

    base_url = os.getenv("LLM_BASEURL", "").strip()
    api_key = os.getenv("LLM_APIKEY", "").strip()
    model = (
        os.getenv("THINKING_MODELNAME")
        or os.getenv("LLM_MODEL")
        or os.getenv("MODEL")
        or "gpt-4o-mini"
    ).strip()

    if not base_url:
        print("Missing env var: LLM_BASEURL", file=sys.stderr)
        return 2
    if not api_key:
        print("Missing env var: LLM_APIKEY", file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {api_key}"}

    timeout_s = int(args.timeout)

    if args.mode == "models":
        url = _join_base_url(base_url, "/models")
        start = time.perf_counter()
        status, resp_json = _get_json(url, headers=headers, timeout_s=timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"GET {url}")
        print(f"HTTP: {status}  Time: {elapsed_ms:.0f}ms")
        if status >= 400 or status == 0:
            print("\nError response:")
            print(json.dumps(resp_json, ensure_ascii=False, indent=2))
            if status == 0:
                _print_timeout_hints(base_url, model)
            return 1
        # Print just a small preview
        data = resp_json.get("data")
        if isinstance(data, list) and data:
            ids = [d.get("id") for d in data[:10] if isinstance(d, dict)]
            print("\nModels (first 10):")
            for mid in ids:
                if mid:
                    print(mid)
        else:
            print("\nRaw JSON:")
            print(json.dumps(resp_json, ensure_ascii=False, indent=2))
        return 0

    # Standard OpenAI Chat Completions endpoint + payload
    url = _join_base_url(base_url, "/chat/completions")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }

    start = time.perf_counter()
    status, resp_json = _post_json(url, headers=headers, payload=payload, timeout_s=timeout_s)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"POST {url}")
    print(f"Model: {model}")
    print(f"HTTP: {status}  Time: {elapsed_ms:.0f}ms")

    if status >= 400 or status == 0:
        # Avoid printing secrets; show structured error if present
        print("\nError response:")
        print(json.dumps(resp_json, ensure_ascii=False, indent=2))
        if status == 0:
            _print_timeout_hints(base_url, model)
        return 1

    # Try to extract content in OpenAI-compatible shape
    content: Optional[str] = None
    try:
        choices = resp_json.get("choices") or []
        if choices and isinstance(choices, list):
            message = choices[0].get("message") or {}
            content = message.get("content")
    except Exception:
        content = None

    print("\nAssistant reply:")
    if content is not None:
        print(content)
    else:
        print("(Could not parse choices[0].message.content; raw JSON below)")
        print(json.dumps(resp_json, ensure_ascii=False, indent=2))

    return 0


def _print_timeout_hints(base_url: str, model: str) -> None:
    print("\n可能原因/排查建议：")
    print(f"- 先试轻量探活：python test_ai_connection.py --mode models --timeout 60")
    print("- 适当加大超时：设置环境变量 LLM_TIMEOUT_S=120 或使用 --timeout 120")
    print("- 确认 baseurl 是否为 OpenAI 兼容入口（通常以 /v1 结尾）：")
    print(f"  当前 LLM_BASEURL={base_url!r}")
    print("- 如果 /models 很快返回但 /chat/completions 超时，可能是模型推理慢或模型名不被支持：")
    print(f"  当前 model={model!r}（可尝试换成该服务商文档里的模型 id）")


if __name__ == "__main__":
    raise SystemExit(main())
