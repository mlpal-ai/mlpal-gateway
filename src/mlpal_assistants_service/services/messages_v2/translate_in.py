"""Inbound translation: Anthropic Messages request → OpenAI-common.

Produces the kwargs the existing provider adapters (OpenAIAdapter / GoogleAdapter)
already consume — OpenAI chat-completions-style messages, ToolDefinition list,
tool_choice, plus a normalized reasoning effort. This is the v2-B/C edge of the
"one internal format, two surfaces" design; the adapters are reused unchanged.

Covers the agent-loop essentials: text, system (string or cache_control blocks),
tool_use / tool_result round-trips, image + document (PDF) attachments, and
thinking→reasoning effort. Image/document blocks are routed to the adapters'
`files` (FileAttachment) path — the same one v1 uses — so each provider emits its
native multimodal parts (OpenAI input_image/input_file, Gemini inline bytes).
Thinking blocks in prior assistant turns are dropped (providers don't accept
reasoning as input). cache_control markers are dropped on the OpenAI path
(OpenAI has automatic prefix caching, surfaced as cache_read in usage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mlpal_assistants_service.adapters.base import (
    FileAttachment,
    FileSource,
    FileType,
    ToolDefinition,
)
from mlpal_assistants_service.services.messages_v2.emit import TOOL_SIG_DELIM
from mlpal_assistants_service.services.messages_v2.reasoning import effort_from_thinking


def _split_tool_sig(raw_id: Any) -> tuple[Any, str | None]:
    """Recover a Gemini thought_signature that emit.py rode on the tool_use id.
    Returns (original_id, signature_or_None). See emit.TOOL_SIG_DELIM."""
    if isinstance(raw_id, str) and TOOL_SIG_DELIM in raw_id:
        orig, sig = raw_id.split(TOOL_SIG_DELIM, 1)
        return orig, sig
    return raw_id, None


@dataclass
class CommonRequest:
    messages: list[dict[str, Any]]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    reasoning_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _blocks_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def _system_text(system: Any) -> str | None:
    if system is None:
        return None
    return _blocks_to_text(system)


_CACHE_TTL_SECONDS = {"5m": 300, "1h": 3600}


def cache_control_ttl(body: dict[str, Any], default_ttl: int) -> int | None:
    """Return the caching TTL the client requested via any `cache_control` marker
    (system blocks, tool defs, or message content blocks), or None if the client
    marked nothing cacheable. The largest marked TTL wins; a marker with no `ttl`
    field uses `default_ttl`. This reads intent only — cache_control is never
    serialized onto the provider request (see the module docstring)."""
    found = False
    best = 0

    def scan(block: Any) -> None:
        nonlocal found, best
        if not isinstance(block, dict):
            return
        cc = block.get("cache_control")
        if isinstance(cc, dict):
            found = True
            best = max(best, _CACHE_TTL_SECONDS.get(cc.get("ttl"), default_ttl))

    system = body.get("system")
    if isinstance(system, list):
        for b in system:
            scan(b)
    for tool in body.get("tools") or []:
        scan(tool)
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for b in content:
                scan(b)

    return best if found else None


def _file_from_block(block: dict[str, Any], file_type: FileType, default_mime: str) -> FileAttachment | None:
    """Anthropic image/document content block → FileAttachment. Routing images
    and PDFs through the adapters' `files` path (rather than Chat-Completions
    `image_url` parts) is what makes them work: the OpenAI adapter emits
    Responses-API `input_image`/`input_file`, and the Google adapter forwards the
    raw decoded bytes via `Part.from_bytes` — original media, untouched."""
    src = block.get("source")
    if not isinstance(src, dict):
        return None
    stype = src.get("type")
    if stype == "base64":
        return FileAttachment(
            type=file_type, source=FileSource.BASE64,
            data=src.get("data", ""), mime_type=src.get("media_type") or default_mime,
        )
    if stype == "url":
        return FileAttachment(
            type=file_type, source=FileSource.URL,
            data=src.get("url", ""), mime_type=src.get("media_type"),
        )
    if stype == "file":  # pre-uploaded provider file id
        return FileAttachment(
            type=file_type, source=FileSource.FILE_ID,
            data=src.get("file_id", ""), mime_type=src.get("media_type") or default_mime,
        )
    return None


def _translate_message(
    msg: dict[str, Any], id_to_name: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """One Anthropic message → one or more OpenAI-common messages."""
    role = msg.get("role", "user")
    content = msg.get("content")

    # Simple string content.
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": str(content)}]

    out: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    files: list[FileAttachment] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":  # assistant called a tool
            tool_id, sig = _split_tool_sig(block.get("id"))
            tc: dict[str, Any] = {
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            }
            if sig:  # replay the Gemini thought_signature the client echoed back
                tc["thought_signature"] = sig
            tool_calls.append(tc)
        elif btype == "tool_result":  # user returns a tool result
            tool_use_id, _ = _split_tool_sig(block.get("tool_use_id"))
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": _blocks_to_text(block.get("content")),
            }
            # Google matches function responses to calls by NAME, so carry the
            # name from the originating tool_use (Anthropic tool_result only
            # references tool_use_id). Without this the adapter sends
            # name="unknown" and Gemini mispairs/garbles the result.
            name = (id_to_name or {}).get(tool_use_id)
            if name:
                tool_msg["name"] = name
            out.append(tool_msg)
        elif btype == "image":
            fa = _file_from_block(block, FileType.IMAGE, "image/png")
            if fa:
                files.append(fa)
        elif btype == "document":
            fa = _file_from_block(block, FileType.PDF, "application/pdf")
            if fa:
                files.append(fa)
        # thinking blocks: dropped (not valid provider input)

    # Assemble the assistant/user turn (besides any tool_result messages above).
    text = "".join(text_parts)
    if tool_calls:
        m: dict[str, Any] = {"role": role, "tool_calls": tool_calls}
        if text:
            m["content"] = text
        out.insert(0, m)
    elif files:
        # Attachments ride on the `files` key; the adapters convert them to the
        # provider-native image/file parts and validate them against the model
        # (unsupported types raise UnsupportedModalityError → a 400, never a
        # silent drop).
        out.insert(0, {"role": role, "content": text, "files": files})
    elif text or not out:
        out.insert(0, {"role": role, "content": text})
    return out


# Keys Google's function_declarations schema subset (OpenAPI 3.0 dialect)
# rejects with INVALID_ARGUMENT. Standard JSON-Schema tool definitions include
# these, so we strip them for the Google edge only (OpenAI/Anthropic accept the
# schema as-is).
_GOOGLE_UNSUPPORTED_SCHEMA_KEYS = frozenset({"additionalProperties", "$schema", "title"})


def sanitize_google_tool_schema(schema: Any) -> Any:
    """Recursively strip schema keys Google rejects and inline $defs/$ref so a
    standard JSON-Schema tool `input_schema` is accepted by Gemini's
    function_declarations. Scoped to the Google edge; OpenAI/Anthropic get the
    schema untouched."""
    defs = schema.get("$defs") or schema.get("definitions") or {} if isinstance(schema, dict) else {}

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if isinstance(ref, str) and ref.rsplit("/", 1)[-1] in defs:
                    return walk(defs[ref.rsplit("/", 1)[-1]])
            return {
                k: walk(v)
                for k, v in obj.items()
                if k not in _GOOGLE_UNSUPPORTED_SCHEMA_KEYS and k not in ("$defs", "definitions")
            }
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        return obj

    return walk(schema)


def _tool_choice(tc: Any) -> str | dict | None:
    if tc is None:
        return None
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "none":
            return "none"
        if t == "tool" and tc.get("name"):
            return {"type": "function", "function": {"name": tc["name"]}}
    return "auto"


def _tool_use_id_to_name(anthropic_messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map each tool_use id (signature stripped) → its function name, so tool
    results can be labeled with the name Google needs for response pairing."""
    mapping: dict[str, str] = {}
    for msg in anthropic_messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
                orig_id, _ = _split_tool_sig(block.get("id"))
                mapping[orig_id] = block["name"]
    return mapping


def to_common(body: dict[str, Any]) -> CommonRequest:
    messages: list[dict[str, Any]] = []
    system = _system_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    anthropic_messages = body.get("messages") or []
    id_to_name = _tool_use_id_to_name(anthropic_messages)
    for msg in anthropic_messages:
        messages.extend(_translate_message(msg, id_to_name))

    tools = None
    if body.get("tools"):
        tools = [
            ToolDefinition(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("input_schema") or {"type": "object", "properties": {}},
            )
            for t in body["tools"]
            if isinstance(t, dict) and t.get("name")
        ]

    return CommonRequest(
        messages=messages,
        tools=tools,
        tool_choice=_tool_choice(body.get("tool_choice")),
        max_tokens=body.get("max_tokens"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop_sequences"),
        reasoning_effort=_explicit_effort(body) or effort_from_thinking(body.get("thinking")),
    )


def _explicit_effort(body: dict[str, Any]) -> str | None:
    """Anthropic-wire `output_config.effort` (low..max), taken verbatim as the
    universal rung; the edge resolves it per model. A `thinking` budget only
    matters when no explicit effort is given."""
    oc = body.get("output_config")
    effort = oc.get("effort") if isinstance(oc, dict) else None
    return effort if isinstance(effort, str) else None
