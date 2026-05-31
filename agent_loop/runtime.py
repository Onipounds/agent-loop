"""The agent runtime: the core loop.

A sentence in, a verified answer out. Each step:
  1. ask the model what to do given the transcript and the available tools
  2. if it requests tools, run them (errors are fed back, never crash the loop)
  3. if it gives a final answer, verify it; on failure, feed back and retry
  4. stop at a final verified answer or when max_steps is hit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .memory import Memory
from .models import Model, ToolCall
from .tools import ToolRegistry
from .verify import Check, verify as default_verify


@dataclass
class Step:
    kind: str            # "tool_calls" | "answer" | "rejected"
    detail: Any = None


@dataclass
class Result:
    answer: str
    verified: bool
    steps: list[Step] = field(default_factory=list)
    n_model_calls: int = 0


def run(
    task: str,
    model: Model,
    tools: ToolRegistry,
    *,
    max_steps: int = 8,
    verifier: Callable[[str, str, Memory], Check] = default_verify,
    memory: Memory | None = None,
) -> Result:
    mem = memory or Memory()
    mem.add_user(task)
    result = Result(answer="", verified=False)

    for _ in range(max_steps):
        response = model.respond(mem.messages, tools.schemas())
        result.n_model_calls += 1

        if response.has_tool_calls():
            mem.add_assistant_tool_calls(response.tool_calls)
            executed: list[tuple[ToolCall, Any]] = []
            for call in response.tool_calls:
                out = tools.execute(call.name, call.input)
                executed.append((call, out))
            mem.add_tool_results(executed)
            result.steps.append(Step("tool_calls", [(c.name, c.input, o) for c, o in executed]))
            continue

        # Final answer candidate -> verify (close the loop)
        answer = response.text or ""
        mem.add_assistant_text(answer)
        check = verifier(task, answer, mem)
        if check.ok:
            result.steps.append(Step("answer", answer))
            result.answer = answer
            result.verified = True
            return result

        # Rejected: tell the agent why and let it try again
        result.steps.append(Step("rejected", check.feedback))
        mem.add_user(check.feedback)

    # Out of steps: return best effort, unverified
    result.answer = result.answer or "(no verified answer within step budget)"
    return result
