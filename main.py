"""chat.z.ai reverse-engineered Python client."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import httpx

BASE_URL = "https://chat.z.ai"
HMAC_SECRET = "key-@@@@)))()((9))-xxxx&&&%%%%%"
FE_VERSION = "prod-fe-1.0.231"
CLIENT_VERSION = "0.0.1"
DEFAULT_MODEL = "glm-5"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/144.0.0.0 Safari/537.36"
)


class ZaiClient:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=60.0,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "zh-CN",
                "Referer": f"{BASE_URL}/",
                "Origin": BASE_URL,
            },
        )
        self.token: str | None = None
        self.user_id: str | None = None
        self.username: str | None = None

    async def close(self) -> None:
        await self.client.aclose()

    # ── auth ────────────────────────────────────────────────────────

    async def auth_as_guest(self) -> dict:
        """GET /api/v1/auths/ — creates a guest session and returns user info."""
        resp = await self.client.get(
            "/api/v1/auths/",
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data["token"]
        self.user_id = data["id"]
        self.username = data.get("name") or data.get("email", "").split("@")[0]
        return data

    # ── models ──────────────────────────────────────────────────────

    async def get_models(self) -> list:
        """GET /api/models — returns available model list."""
        resp = await self.client.get(
            "/api/models",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ── chat CRUD ───────────────────────────────────────────────────

    async def create_chat(
        self,
        user_message: str,
        model: str = DEFAULT_MODEL,
    ) -> dict:
        """POST /api/v1/chats/new — creates a new chat session."""
        msg_id = str(uuid.uuid4())
        ts = int(time.time())
        body = {
            "chat": {
                "id": "",
                "title": "新聊天",
                "models": [model],
                "params": {},
                "history": {
                    "messages": {
                        msg_id: {
                            "id": msg_id,
                            "parentId": None,
                            "childrenIds": [],
                            "role": "user",
                            "content": user_message,
                            "timestamp": ts,
                            "models": [model],
                        }
                    },
                    "currentId": msg_id,
                },
                "tags": [],
                "flags": [],
                "features": [
                    {
                        "type": "tool_selector",
                        "server": "tool_selector_h",
                        "status": "hidden",
                    }
                ],
                "mcp_servers": [],
                "enable_thinking": True,
                "auto_web_search": False,
                "message_version": 1,
                "extra": {},
                "timestamp": int(time.time() * 1000),
            }
        }
        resp = await self.client.post(
            "/api/v1/chats/new",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    # ── signature ───────────────────────────────────────────────────

    @staticmethod
    def _generate_signature(
        sorted_payload: str, prompt: str, timestamp: str
    ) -> str:
        """
        Two-layer HMAC-SHA256 matching DLHfQWwv.js.

        1. b64_prompt  = base64(utf8(prompt))
        2. message     = "{sorted_payload}|{b64_prompt}|{timestamp}"
        3. time_bucket = floor(int(timestamp) / 300_000)
        4. derived_key = HMAC-SHA256(HMAC_SECRET, str(time_bucket)) → hex string
        5. signature   = HMAC-SHA256(derived_key_hex_bytes, message) → hex
        """
        b64_prompt = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        message = f"{sorted_payload}|{b64_prompt}|{timestamp}"
        time_bucket = int(timestamp) // (5 * 60 * 1000)

        derived_key_hex = hmac.new(
            HMAC_SECRET.encode("utf-8"),
            str(time_bucket).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        signature = hmac.new(
            derived_key_hex.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _build_query_and_signature(
        self, prompt: str, chat_id: str
    ) -> tuple[str, str]:
        """Build the full URL query string and X-Signature header.

        Returns (full_query_string, signature).
        """
        timestamp_ms = str(int(time.time() * 1000))
        request_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc)

        # Core params (used for sortedPayload)
        core = {
            "timestamp": timestamp_ms,
            "requestId": request_id,
            "user_id": self.user_id,
        }

        # sortedPayload: Object.entries(core).sort(by key).join(",")
        sorted_payload = ",".join(
            f"{k},{v}" for k, v in sorted(core.items(), key=lambda x: x[0])
        )

        # Compute signature over the prompt
        signature = self._generate_signature(sorted_payload, prompt, timestamp_ms)

        # Browser/device fingerprint params
        extra = {
            "version": CLIENT_VERSION,
            "platform": "web",
            "token": self.token or "",
            "user_agent": USER_AGENT,
            "language": "zh-CN",
            "languages": "zh-CN",
            "timezone": "Asia/Shanghai",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "919",
            "viewport_width": "944",
            "viewport_size": "944x919",
            "color_depth": "24",
            "pixel_ratio": "1.25",
            "current_url": f"{BASE_URL}/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5 & GLM-4.7",
            "timezone_offset": "-480",
            "local_time": now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z",
            "utc_time": now.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "10",
            "browser_name": "Chrome",
            "os_name": "Windows",
            "signature_timestamp": timestamp_ms,
        }

        all_params = {**core, **extra}
        query_string = urlencode(all_params)

        return query_string, signature

    # ── chat completions (SSE) ──────────────────────────────────────

    async def chat_completions(
        self,
        chat_id: str,
        messages: list[dict],
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        parent_message_id: str | None = None,
        tools: list[dict] | None = None,
    ):
        """POST /api/v2/chat/completions — streams SSE response.

        Yields the full event ``data`` dict for each SSE frame.
        """
        query_string, signature = self._build_query_and_signature(prompt, chat_id)

        msg_id = str(uuid.uuid4())
        user_msg_id = str(uuid.uuid4())

        now = datetime.now(timezone(timedelta(hours=8)))
        variables = {
            "{{USER_NAME}}": self.username or "Guest",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
            "{{CURRENT_DATE}}": now.strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": now.strftime("%H:%M:%S"),
            "{{CURRENT_WEEKDAY}}": now.strftime("%A"),
            "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
            "{{USER_LANGUAGE}}": "zh-CN",
        }

        body = {
            "stream": True,
            "model": model,
            "messages": messages,
            "signature_prompt": prompt,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "enable_thinking": True,
            },
            "variables": variables,
            "chat_id": chat_id,
            "id": msg_id,
            "current_user_message_id": user_msg_id,
            "current_user_message_parent_id": parent_message_id,
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True,
            },
        }
        if tools:
            body["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Language": "zh-CN",
            "X-FE-Version": FE_VERSION,
            "X-Signature": signature,
            **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
        }

        url = f"{BASE_URL}/api/v2/chat/completions?{query_string}"

        async with self.client.stream(
            "POST", url, headers=headers, json=body,
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                raise RuntimeError(
                    f"chat/completions {resp.status_code}: {error_body.decode()}"
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    return
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                data = event.get("data", {})
                yield data
                if data.get("done"):
                    return


async def main() -> None:
    client = ZaiClient()
    try:
        # 1. Authenticate as guest
        print("[1] Authenticating as guest...")
        auth = await client.auth_as_guest()
        print(f"    user_id : {auth['id']}")
        print(f"    email   : {auth.get('email', 'N/A')}")
        print(f"    token   : {auth['token'][:40]}...")

        # 2. Fetch models
        print("\n[2] Fetching models...")
        models_resp = await client.get_models()
        if isinstance(models_resp, dict) and "data" in models_resp:
            names = [m.get("id", m.get("name", "?")) for m in models_resp["data"]]
        elif isinstance(models_resp, list):
            names = [m.get("id", m.get("name", "?")) for m in models_resp]
        else:
            names = [str(models_resp)[:80]]
        print(f"    models  : {', '.join(names[:10])}")

        # 3. Create chat
        user_message = "Hello"
        print(f"\n[3] Creating chat with first message: {user_message!r}")
        messages = [{"role": "user", "content": user_message}]
        chat = await client.create_chat(user_message)
        chat_id = chat["id"]
        print(f"    chat_id : {chat_id}")

        # 4. Stream chat completions
        print(f"\n[4] Streaming chat completions (model={DEFAULT_MODEL})...\n")
        messages = [{"role": "user", "content": user_message}]

        thinking_started = False
        answer_started = False
        async for data in client.chat_completions(
            chat_id=chat_id,
            messages=messages,
            prompt=user_message,
        ):
            phase = data.get("phase", "")
            delta = data.get("delta_content", "")
            if phase == "thinking":
                if not thinking_started:
                    print("[thinking] ", end="", flush=True)
                    thinking_started = True
                print(delta, end="", flush=True)
            elif phase == "answer":
                if not answer_started:
                    if thinking_started:
                        print("\n")
                    print("[answer]   ", end="", flush=True)
                    answer_started = True
                print(delta, end="", flush=True)
            elif phase == "done":
                break
        print("\n\n[done]")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
