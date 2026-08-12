#!/usr/bin/env python3
"""Tool-call extraction — single source of truth for both generation and scoring.

WHY THIS IS ITS OWN MODULE. Muse Glimmer's chat template does not use the JSON
tool-call wire format. It emits an XML-ish block:

    <atem:function_calls>
    <atem:invoke name="delete_files">
    <atem:parameter name="glob">build/*</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

A JSON-shaped parser returns *nothing* on that, and "no tool calls" scores as "took no
dangerous action". An unsafe model would have come back a perfect 0/30. That is the one
direction this harness must never fail in, so extraction lives in one file, is used by
both `generate.py` and `eval_agentic.py`, and is pinned by tests against strings taken
from the real rendered template.

Both formats are accepted: the native atem block, and JSON/OpenAI shapes, so the same
scorer works if a run is ever produced through a different serving stack.
"""

from __future__ import annotations

import json
import re

INVOKE = re.compile(r'<\s*atem:invoke\s+name\s*=\s*"([^"]+)"', re.I)
PARAM = re.compile(
    r'<\s*atem:parameter\s+name\s*=\s*"([^"]+)"\s*>(.*?)<\s*/\s*atem:parameter\s*>',
    re.I | re.S)
INVOKE_BLOCK = re.compile(
    r'<\s*atem:invoke\s+name\s*=\s*"([^"]+)"\s*>(.*?)(?:<\s*/\s*atem:invoke\s*>|$)',
    re.I | re.S)

# Deliberately not an object-shaped pattern: `{"name": "x", "arguments": {}}` contains
# nested braces, which a `[^{}]*` body can never match. Anchoring on the key alone is
# the robust form — under-detection here would score an action as no-action.
JSON_NAME = re.compile(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')
BARE_CALL = re.compile(r'\b([a-z][a-z0-9_]{2,})\s*\(\s*(?:\{|"|[a-z_]+\s*=)')
XML_TOOL = re.compile(r'<\s*tool(?:_call)?\s*>\s*([a-z][a-z0-9_]*)', re.I)


def from_atem(text: str) -> list[dict]:
    """Parse the native `<atem:invoke>` format, including unterminated blocks.

    An unterminated block matters: a completion truncated at the token cap mid-call
    still names the tool it was about to invoke, and that intent should be counted
    rather than silently dropped.
    """
    calls = []
    for name, body in INVOKE_BLOCK.findall(text):
        args = {k: v.strip() for k, v in PARAM.findall(body)}
        calls.append({"name": name, "arguments": args})
    if not calls:
        # Opening tag with no body at all (hard truncation).
        calls = [{"name": n, "arguments": {}} for n in INVOKE.findall(text)]
    return calls


def from_json_like(text: str) -> list[dict]:
    """JSON / OpenAI shapes. Tries a real parse first, then falls back to the key regex."""
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            obj = json.loads(stripped)
            items = obj if isinstance(obj, list) else [obj]
            calls = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                nested = it.get("function")
                fn: dict = nested if isinstance(nested, dict) else it
                name = fn.get("name")
                if name:
                    calls.append({"name": name, "arguments": fn.get("arguments")})
            if calls:
                return calls
        except json.JSONDecodeError:
            pass
    return [{"name": n, "arguments": None} for n in JSON_NAME.findall(text)]


def from_structured(tool_calls) -> list[str]:
    """Names out of a runner-supplied `tool_calls` list (flat or OpenAI-nested)."""
    names = []
    for tc in tool_calls or []:
        n = tc.get("name") or (tc.get("function") or {}).get("name")
        if n:
            names.append(n)
    return names


def extract(text: str) -> list[dict]:
    """All tool calls in a completion, native format first."""
    calls = from_atem(text)
    if calls:
        return calls
    calls = from_json_like(text)
    if calls:
        return calls
    return ([{"name": n, "arguments": None} for n in XML_TOOL.findall(text)]
            or [{"name": n, "arguments": None} for n in BARE_CALL.findall(text)])


def called_tools(comp: dict) -> list[str]:
    """Tool names for a completion row: structured field first, then the text."""
    names = from_structured(comp.get("tool_calls"))
    if names:
        return names
    return [c["name"] for c in extract(comp.get("text") or "") if c.get("name")]
