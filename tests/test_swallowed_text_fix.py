"""Test that text accompanying tool_calls is yielded as tokens.

This is a regression test for the "swallowed text" bug where intermediate
text (text in LLM responses that also include tool_calls) was saved to
context but never yielded as tokens, causing render_message to miss it.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest


# --- Minimal stubs matching kernel contracts ---


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: dict


class _EnumLike:
    """Mimics an enum with .value, matching real SDK content block types."""

    def __init__(self, val):
        self.value = val

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return self.value == getattr(other, "value", other)

    def __str__(self):
        return self.value


@dataclass
class FakeTextBlock:
    type: Any = field(default_factory=lambda: _EnumLike("text"))
    text: str = ""

    def model_dump(self):
        t = self.type.value if hasattr(self.type, "value") else self.type
        return {"type": t, "text": self.text}

    def to_dict(self):
        return self.model_dump()


@dataclass
class FakeToolUseBlock:
    type: Any = field(default_factory=lambda: _EnumLike("tool_use"))
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)

    def model_dump(self):
        t = self.type.value if hasattr(self.type, "value") else self.type
        return {"type": t, "id": self.id, "name": self.name, "input": self.input}

    def to_dict(self):
        return self.model_dump()


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50

    def model_dump(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass
class FakeResponse:
    """Mimics a provider response with text + tool_calls."""

    text: str = ""
    content: list = field(default_factory=list)
    content_blocks: list = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)
    metadata: dict = field(default_factory=dict)


class FakeContext:
    """Minimal context that records messages and normalizes for provider requests.

    Like the real context, get_messages_for_request() returns dicts with
    string content (the provider contract), while add_message() stores the
    full structured message (including tool_calls, content blocks, etc.).
    """

    def __init__(self):
        self.messages = []

    async def get_messages(self):
        return list(self.messages)

    async def add_message(self, msg):
        self.messages.append(msg)

    async def get_messages_for_request(self, **kwargs):
        """Return messages normalized to simple string content for the provider."""
        normalized = []
        for msg in self.messages:
            m = dict(msg)
            # Normalize list content to string (like real context does)
            if isinstance(m.get("content"), list):
                text_parts = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                m["content"] = "\n\n".join(text_parts) if text_parts else ""
            # Strip tool_calls from message dicts (they're in the content blocks)
            m.pop("tool_calls", None)
            m.pop("thinking_block", None)
            m.pop("metadata", None)
            if m.get("content") or m.get("role") == "tool":
                normalized.append(m)
        return normalized


class FakeHooks:
    """Minimal hook registry that records events."""

    def __init__(self):
        self.events = []

    async def emit(self, event_name, data=None):
        self.events.append((event_name, data))
        return MagicMock(action=None)


class FakeProvider:
    """Provider that returns a sequence of canned responses.

    First response: text + tool_call
    Second response: text only (final)
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, chat_request, **kwargs):
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    def parse_tool_calls(self, response):
        # Check if response has tool_use blocks in content
        if hasattr(response, "content"):
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    return [
                        FakeToolCall(
                            id=block.id, name=block.name, arguments=block.input
                        )
                    ]
        return []


class FakeTool:
    name = "test_tool"
    description = "A test tool"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        from amplifier_core import ToolResult

        return ToolResult(content='{"result": "ok"}')


@pytest.mark.asyncio
async def test_intermediate_text_is_yielded():
    """The core regression test: text in a response with tool_calls must be yielded."""
    from amplifier_module_loop_streaming import StreamingOrchestrator

    briefing_text = "Here is a 4608-char briefing about yesterday's work..."
    final_text = "Waiting on your call on those four items."

    # Response 1: text + tool_call (the briefing that was previously swallowed)
    resp1 = FakeResponse(
        text=briefing_text,
        content=[
            FakeTextBlock(text=briefing_text),
            FakeToolUseBlock(id="tc_1", name="test_tool", input={}),
        ],
        content_blocks=[
            FakeTextBlock(text=briefing_text),
            FakeToolUseBlock(id="tc_1", name="test_tool", input={}),
        ],
    )

    # Response 2: text only (final response)
    resp2 = FakeResponse(
        text=final_text,
        content=[FakeTextBlock(text=final_text)],
        content_blocks=[FakeTextBlock(text=final_text)],
    )

    provider = FakeProvider([resp1, resp2])
    context = FakeContext()
    hooks = FakeHooks()
    tools = {"test_tool": FakeTool()}

    # Set up the orchestrator with no streaming (triggers the non-streaming path)
    orch = StreamingOrchestrator({"max_iterations": 10, "stream_delay": 0})

    # Add user message to context
    await context.add_message({"role": "user", "content": "Give me a status update"})

    # Execute and collect the full response
    full_response = await orch.execute(
        prompt="Give me a status update",
        context=context,
        providers={"default": provider},
        tools=tools,
        hooks=hooks,
    )

    # THE KEY ASSERTION: full_response must contain BOTH the briefing AND the final text
    assert briefing_text in full_response, (
        f"Intermediate text was swallowed! full_response only contains: {full_response[:200]!r}"
    )
    assert final_text in full_response, (
        f"Final text missing from full_response: {full_response[:200]!r}"
    )


@pytest.mark.asyncio
async def test_text_only_response_still_works():
    """Sanity check: a response with text and NO tool_calls still works."""
    from amplifier_module_loop_streaming import StreamingOrchestrator

    final_text = "Here is a simple response with no tools."

    resp = FakeResponse(
        text=final_text,
        content=[FakeTextBlock(text=final_text)],
        content_blocks=[FakeTextBlock(text=final_text)],
    )

    provider = FakeProvider([resp])
    context = FakeContext()
    hooks = FakeHooks()

    orch = StreamingOrchestrator({"max_iterations": 10, "stream_delay": 0})
    await context.add_message({"role": "user", "content": "Hello"})

    full_response = await orch.execute(
        prompt="Hello",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert final_text in full_response
