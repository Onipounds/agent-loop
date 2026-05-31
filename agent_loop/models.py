"""Model interface: a real Anthropic-backed model and a deterministic offline fake.

The whole point of this module is that the agent runtime depends only on the
small ``Model`` protocol below. In tests and CI we use ``FakeModel`` (scripted,
deterministic, no network), and in real use we use ``AnthropicModel``. Nothing
else in the codebase imports the Anthropic SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# --- The data the runtime passes around -------------------------------------

@dataclass
class ToolCall:
    """A request from the model to run one tool."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelResponse:
    """What a model returns for one step of the loop.

    Either it asks for one or more tool calls, or it returns final ``text``.
    """
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class Model(Protocol):
    """The only contract the runtime depends on."""

    def respond(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> ModelResponse:
        ...


# --- Offline fake model (deterministic, used by tests/CI) -------------------

@dataclass
class Scripted:
    """One scripted step for the fake model.

    ``tool_calls`` is a list of (name, input) tuples to request, OR set
    ``text`` to produce a final answer. A callable ``when`` can gate the step
    on the current messages, but by default steps are returned in order.
    """
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    text: str | None = None


class FakeModel:
    """Returns a fixed sequence of responses. Fully deterministic.

    Example:
        FakeModel([
            Scripted(tool_calls=[("calculate", {"expression": "mean(1,2,3)"})]),
            Scripted(text="The average is 2."),
        ])
    """

    def __init__(self, script: list[Scripted]):
        self._script = list(script)
        self._step = 0
        self.calls_seen: list[dict[str, Any]] = []  # for assertions in tests

    def respond(self, messages, tool_schemas) -> ModelResponse:
        self.calls_seen.append({"messages": list(messages), "tools": list(tool_schemas)})
        if self._step >= len(self._script):
            # Ran out of script: behave like a model that gives up cleanly.
            return ModelResponse(text="(no further response)")
        step = self._script[self._step]
        self._step += 1
        if step.tool_calls:
            calls = [
                ToolCall(id=f"call_{self._step}_{i}", name=name, input=inp)
                for i, (name, inp) in enumerate(step.tool_calls)
            ]
            return ModelResponse(tool_calls=calls)
        return ModelResponse(text=step.text)


# --- Real Anthropic model (lazy import; not needed for tests) ---------------

class AnthropicModel:
    """Wraps the Anthropic Messages API tool-use flow.

    The SDK is imported lazily so the package (and its tests) work with no
    SDK installed and no API key set. Configure via the ANTHROPIC_API_KEY env
    var; the model name defaults to a current Claude model and is overridable
    with AGENT_LOOP_MODEL.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import os
        self.model = model or os.environ.get("AGENT_LOOP_MODEL", "claude-sonnet-4-6")
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed for real calls
            self._client = anthropic.Anthropic()
        return self._client

    def respond(self, messages, tool_schemas) -> ModelResponse:
        client = self._get_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tool_schemas,
        )
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
            elif btype == "text":
                text_parts.append(block.text)
        if tool_calls:
            return ModelResponse(tool_calls=tool_calls)
        return ModelResponse(text="".join(text_parts))
