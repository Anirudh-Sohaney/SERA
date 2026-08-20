"""OpenAI client that reuses the local Codex CLI credentials.

Connects to the same OpenAI account as the Codex CLI (ChatGPT auth mode) by
reading ``~/.codex/auth.json`` and speaking the Codex Responses WebSocket
protocol (``wss://chatgpt.com/backend-api/codex/responses``).

Model: gpt-5.6-luna, reasoning effort: medium (matches ~/.codex/config.toml).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import websockets

log = logging.getLogger("openai_client")

CODEX_AUTH_PATH = os.path.expanduser("~/.codex/auth.json")
WS_URL = "wss://chatgpt.com/backend-api/codex/responses"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # from the id_token `aud` claim
REDIRECT_URI = "com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback"

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING = "medium"


class AuthError(RuntimeError):
    """Raised when Codex credentials are missing or cannot be refreshed."""


class APIError(RuntimeError):
    """Raised when the model call fails after retries."""


class RateLimitError(APIError):
    """Raised when the server rejects calls with HTTP 403 (account rate
    limit). Callers should back off and retry later instead of hammering."""


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def _load_auth() -> dict[str, Any]:
    if not os.path.exists(CODEX_AUTH_PATH):
        raise AuthError(
            f"Codex auth file not found at {CODEX_AUTH_PATH}. "
            "Run `codex login` first."
        )
    with open(CODEX_AUTH_PATH) as f:
        return json.load(f)


def _token_expiry(token: str) -> int:
    try:
        return int(_decode_jwt_payload(token).get("exp", 0))
    except Exception:
        return 0


def refresh_access_token() -> str:
    """Refresh the ChatGPT access token using the stored refresh token."""
    auth = _load_auth()
    tokens = auth.get("tokens") or {}
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise AuthError("No refresh_token in codex auth.json; cannot refresh.")
    body = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
        }
    ).encode()
    req = urllib.request.Request(
        OAUTH_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise AuthError(f"Token refresh failed: HTTP {e.code} {e.read()[:300]!r}") from e
    new_access = data.get("access_token")
    if not new_access:
        raise AuthError(f"Token refresh returned no access_token: {data}")
    # Persist the refreshed tokens back into codex auth.json so codex stays in sync.
    tokens["access_token"] = new_access
    if data.get("refresh_token"):
        tokens["refresh_token"] = data["refresh_token"]
    auth["tokens"] = tokens
    auth["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    tmp = CODEX_AUTH_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(auth, f, indent=2)
    os.replace(tmp, CODEX_AUTH_PATH)
    log.info("Refreshed ChatGPT access token (expires in ~%dh)",
             (_token_expiry(new_access) - int(time.time())) // 3600)
    return new_access


def get_access_token() -> str:
    """Return a valid access token, refreshing it if near expiry."""
    auth = _load_auth()
    tokens = auth.get("tokens") or {}
    access = tokens.get("access_token")
    if not access:
        raise AuthError("No access_token in codex auth.json.")
    exp = _token_expiry(access)
    # Refresh when under 30 minutes remain.
    if exp and exp - int(time.time()) < 1800:
        log.info("Access token near expiry; refreshing.")
        return refresh_access_token()
    return access


@dataclass
class CompletionResult:
    text: str
    response_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    model: Optional[str] = None


async def _ws_complete(
    messages: list[dict[str, str]],
    model: str,
    reasoning: str,
    token: str,
    timeout: float = 300.0,
) -> CompletionResult:
    """Run one Responses-API turn over the Codex WebSocket."""
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "codex_cli_rs/0.147.0",
        "Origin": "https://chatgpt.com",
        "OpenAI-Beta": "responses_websockets=2026-02-06",
        "originator": "codex_cli_rs",
        "version": "0.147.0",
    }
    frame = {
        "type": "response.create",
        "model": model,
        "input": messages,
        "reasoning": {"effort": reasoning},
        "stream": True,
    }
    text_parts: list[str] = []
    response_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    completed = False

    async with websockets.connect(
        WS_URL, additional_headers=headers, max_size=None, open_timeout=60
    ) as ws:
        await ws.send(json.dumps(frame))
        async for raw in ws:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "response.output_text.delta":
                text_parts.append(ev.get("delta", ""))
            elif etype == "response.completed":
                resp = ev.get("response", {})
                response_id = resp.get("id")
                usage = resp.get("usage")
                completed = True
                break
            elif etype == "response.failed":
                err = ev.get("response", {}).get("error")
                raise APIError(f"Model call failed: {err}")
            elif etype == "error":
                err = ev.get("error", {})
                raise APIError(
                    f"API error {err.get('status', '')}: {err.get('message', raw)}"
                )
    if not completed:
        raise APIError("WebSocket closed before response.completed")
    return CompletionResult(
        text="".join(text_parts), response_id=response_id, usage=usage, model=model
    )


def complete(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    reasoning: str = DEFAULT_REASONING,
    max_retries: int = 3,
    timeout: float = 300.0,
) -> CompletionResult:
    """Synchronous completion helper with retry and token refresh."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            token = get_access_token()
            return asyncio.run(
                _ws_complete(messages, model, reasoning, token, timeout=timeout)
            )
        except (AuthError, APIError) as e:
            last_err = e
            if isinstance(e, AuthError):
                raise
            log.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)
        except Exception as e:  # websocket/network errors: reconnect and retry
            last_err = e
            # HTTP 403 on the WebSocket connection = account rate limit.
            # Signal it explicitly so callers can back off meaningfully.
            if "403" in str(e):
                raise RateLimitError(
                    f"server rejected WebSocket connection with HTTP 403 "
                    f"(account rate limit): {e}"
                ) from e
            log.warning(
                "Attempt %d/%d transport error (%s: %s); retrying",
                attempt + 1, max_retries, type(e).__name__, e,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)
    raise APIError(f"All {max_retries} attempts failed: {last_err}")


def complete_text(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    reasoning: str = DEFAULT_REASONING,
    max_retries: int = 3,
) -> str:
    """Convenience: single-turn text completion."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return complete(messages, model=model, reasoning=reasoning,
                    max_retries=max_retries).text


def check_rate_limit() -> dict[str, Any]:
    """Query the account's codex usage/rate-limit status.

    Returns {"allowed": bool, "used_percent": int|None, "limit_reached": bool}.
    """
    try:
        token = get_access_token()
        req = urllib.request.Request(
            "https://chatgpt.com/backend-api/codex/usage",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "codex_cli_rs/0.147.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        rl = data.get("rate_limit", {})
        pw = rl.get("primary_window") or {}
        return {
            "allowed": bool(rl.get("allowed", True)),
            "limit_reached": bool(rl.get("limit_reached", False)),
            "used_percent": pw.get("used_percent"),
            "reset_after_seconds": pw.get("reset_after_seconds"),
            "plan_type": data.get("plan_type"),
        }
    except Exception as e:  # never fail the generator because of a probe
        log.warning("rate-limit probe failed: %s", e)
        return {"allowed": True, "limit_reached": False, "used_percent": None,
                "reset_after_seconds": None, "plan_type": None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(complete_text("Reply with exactly: PONG"))