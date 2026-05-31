"""Working memory: the running message transcript plus a record of tool results.

Messages are kept in the shape the Anthropic Messages API expects, so the same
transcript drives both the real and fake models. ``tool_results_seen`` is a flat
log the verifier uses to check the agent's answer is grounded in real outputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .models import ToolCall


@dataclass
class Memory:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results_seen: list[Any] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_tool_calls(self, calls: list[ToolCall]) -> None:
        content = [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
            for c in calls
        ]
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[tuple[ToolCall, Any]]) -> None:
        content = []
        for call, result in results:
            self.tool_results_seen.append(result)
            content.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result),
            })
        self.messages.append({"role": "user", "content": content})

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def numbers_seen(self) -> list[float]:
        """Every numeric value that appeared in a tool result this run.

        Used by the grounding verifier to confirm the answer's numbers are real.
        """
        nums: list[float] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, bool):
                return
            if isinstance(obj, (int, float)):
                nums.append(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)

        walk(self.tool_results_seen)
        return nums
