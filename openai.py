"""OpenAI-compatible proxy server for chat.z.ai."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from main import ZaiClient

# ── Session Pool ─────────────────────────────────────────────────────


class SessionPool:
    """Manages a single ZaiClient instance with automatic auth refresh."""

    def __init__(self) -> None:
        self._client = ZaiClient()
        self._lock = asyncio.Lock()
        self._authed = False

    async def close(self) -> None:
        await self._client.close()

    async def ensure_auth(self) -> None:
        """Authenticate if not already done."""
        if not self._authed:
            await self._client.auth_as_guest()
            self._authed = True

    async def refresh_auth(self) -> None:
        """Force-refresh the guest token (locked to avoid concurrent rebuilds)."""
        async with self._lock:
            await self._client.auth_as_guest()
            self._authed = True

    async def get_models(self) -> list | dict:
        await self.ensure_auth()
        return await self._client.get_models()

    async def create_chat(self, user_message: str, model: str) -> dict:
        return await self._client.create_chat(user_message, model)

    def chat_completions(
        self,
        chat_id: str,
        messages: list[dict],
        prompt: str,
        *,
        model: str,
        tools: list[dict] | None = None,
    ):
        return self._client.chat_completions(
            chat_id=chat_id,
            messages=messages,
            prompt=prompt,
            model=model,
            tools=tools,
        )


pool = SessionPool()

# ── FastAPI app ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await pool.ensure_auth()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)

# ── Helpers ──────────────────────────────────────────────────────────


class Toolify:
    """ToolCall 适配层：统一工具定义、消息扁平化与上游 tool_call 解析。"""

    @staticmethod
    def normalize_tools(tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None

        normalized: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue
            fn = tool.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not name:
                continue

            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )

        return normalized or None

    @staticmethod
    def extract_prompt(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
        return ""

    @staticmethod
    def flatten_messages(messages: list[dict]) -> list[dict]:
        """把多轮/工具消息转换为上游可消费的单 user message。"""
        parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant" and msg.get("tool_calls"):
                parts.append(f"<ASSISTANT>{content or ''}</ASSISTANT>")
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "")
                    arguments = fn.get("arguments", "")
                    call_id = tc.get("id", "") if isinstance(tc, dict) else ""
                    parts.append(
                        f"<TOOL_CALL id='{call_id}' name='{name}'>{arguments}</TOOL_CALL>"
                    )
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                parts.append(f"<TOOL id='{tool_call_id}'>{content}</TOOL>")
                continue

            parts.append(f"<{role.upper()}>{content}</{role.upper()}>")

        return [{"role": "user", "content": "\n".join(parts)}]

    @staticmethod
    def parse_tool_calls(data: dict) -> list[dict]:
        """兼容多种上游 tool call 字段格式。"""
        if not isinstance(data, dict):
            return []

        raw_calls = data.get("tool_calls") or data.get("toolCalls") or []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]

        parsed: list[dict] = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue

            fn = tc.get("function", {}) if isinstance(tc.get("function", {}), dict) else {}
            name = (
                fn.get("name")
                or tc.get("name")
                or tc.get("tool_name")
                or ""
            )
            arguments = fn.get("arguments")
            if arguments is None:
                arguments = tc.get("arguments", "")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)

            parsed.append(
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else "",
                    },
                }
            )

        # 兜底兼容旧式 function_call
        if not parsed and isinstance(data.get("function_call"), dict):
            fc = data["function_call"]
            arguments = fc.get("arguments", "")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            parsed.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": arguments if isinstance(arguments, str) else "",
                    },
                }
            )

        return parsed


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def _openai_chunk(
    completion_id: str,
    model: str,
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _openai_completion(
    completion_id: str,
    model: str,
    content: str,
    reasoning_content: str,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ── /v1/models ───────────────────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    models_resp = await pool.get_models()
    # Normalize to list
    if isinstance(models_resp, dict) and "data" in models_resp:
        models_list = models_resp["data"]
    elif isinstance(models_resp, list):
        models_list = models_resp
    else:
        models_list = []

    data = []
    for m in models_list:
        mid = m.get("id") or m.get("name", "unknown")
        data.append(
            {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "z.ai",
            }
        )
    return {"object": "list", "data": data}


# ── /v1/chat/completions ────────────────────────────────────────────


async def _do_request(
    messages: list[dict],
    model: str,
    prompt: str,
    tools: list[dict] | None = None,
):
    """Create a new chat and return (chat_id, async generator).

    Raises on Zai errors so the caller can retry.
    """
    chat = await pool.create_chat(prompt, model)
    chat_id = chat["id"]
    gen = pool.chat_completions(
        chat_id=chat_id,
        messages=messages,
        prompt=prompt,
        model=model,
        tools=tools,
    )
    return chat_id, gen


async def _stream_response(
    messages: list[dict],
    model: str,
    prompt: str,
    tools: list[dict] | None = None,
):
    """SSE generator with one retry on error."""
    completion_id = _make_id()
    retried = False

    while True:
        try:
            _chat_id, gen = await _do_request(messages, model, prompt, tools)

            # Send initial role chunk
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

            tool_call_idx = 0
            async for data in gen:
                phase = data.get("phase", "")
                delta = data.get("delta_content", "")

                # Tool call events from Zai / Toolify
                tool_calls = Toolify.parse_tool_calls(data)
                if tool_calls:
                    for tc in tool_calls:
                        tc_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": tool_call_idx,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": tc["function"],
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"
                        tool_call_idx += 1
                elif phase == "thinking" and delta:
                    chunk = _openai_chunk(
                        completion_id, model, reasoning_content=delta
                    )
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                elif phase == "answer" and delta:
                    chunk = _openai_chunk(completion_id, model, content=delta)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                elif phase == "done":
                    break

            # Send finish chunk
            finish_reason = "tool_calls" if tool_call_idx > 0 else "stop"
            finish_chunk = _openai_chunk(
                completion_id, model, finish_reason=finish_reason
            )
            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        except Exception:
            if retried:
                # Already retried once — yield error and stop
                error = {
                    "error": {
                        "message": "Upstream Zai error after retry",
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(error)}\n\n"
                yield "data: [DONE]\n\n"
                return
            retried = True
            await pool.refresh_auth()
            # Loop back and retry


async def _sync_response(
    messages: list[dict],
    model: str,
    prompt: str,
    tools: list[dict] | None = None,
) -> dict:
    """Non-streaming response with one retry on error."""
    completion_id = _make_id()

    for attempt in range(2):
        try:
            _chat_id, gen = await _do_request(messages, model, prompt, tools)

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict] = []

            async for data in gen:
                phase = data.get("phase", "")
                delta = data.get("delta_content", "")

                parsed_tool_calls = Toolify.parse_tool_calls(data)
                if parsed_tool_calls:
                    tool_calls.extend(parsed_tool_calls)
                elif phase == "thinking" and delta:
                    reasoning_parts.append(delta)
                elif phase == "answer" and delta:
                    content_parts.append(delta)
                elif phase == "done":
                    break

            if tool_calls:
                message: dict = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                if reasoning_parts:
                    message["reasoning_content"] = "".join(reasoning_parts)
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }

            return _openai_completion(
                completion_id,
                model,
                "".join(content_parts),
                "".join(reasoning_parts),
            )

        except Exception:
            if attempt == 0:
                await pool.refresh_auth()
                continue
            return {
                "error": {
                    "message": "Upstream Zai error after retry",
                    "type": "server_error",
                }
            }

    # Unreachable, but satisfy type checker
    return {"error": {"message": "Unexpected error", "type": "server_error"}}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model: str = body.get("model", "glm-5")
    messages: list[dict] = body.get("messages", [])
    stream: bool = body.get("stream", False)
    tools: list[dict] | None = Toolify.normalize_tools(body.get("tools"))

    # 通过 Toolify 统一抽取 prompt 与上下文扁平化，支持 tool / tool_calls 消息
    prompt = Toolify.extract_prompt(messages)
    messages = Toolify.flatten_messages(messages)

    if not prompt:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No user message found in messages",
                    "type": "invalid_request_error",
                }
            },
        )

    if stream:
        return StreamingResponse(
            _stream_response(messages, model, prompt, tools),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await _sync_response(messages, model, prompt, tools)
        if "error" in result:
            return JSONResponse(status_code=502, content=result)
        return result


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
