"""Eval harness: run the agent over a task set and score it.

A task is a prompt plus a checker. The checker decides pass/fail from the
Result (e.g. does the answer contain the expected substring, was it verified).
The runner reports pass rate and average steps -- the numbers you quote when
someone asks "how do you know your agent works?".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .runtime import Result, run
from .models import Model
from .tools import ToolRegistry

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


@dataclass
class Task:
    name: str
    prompt: str
    expect_substring: str | None = None
    expect_verified: bool = True


@dataclass
class TaskOutcome:
    name: str
    passed: bool
    reason: str
    n_steps: int
    answer: str


def check_task(task: Task, result: Result) -> tuple[bool, str]:
    if task.expect_verified and not result.verified:
        return False, "answer was not verified"
    if task.expect_substring and task.expect_substring.lower() not in result.answer.lower():
        return False, f"expected substring {task.expect_substring!r} not found"
    return True, "ok"


def load_tasks(path: Path | None = None) -> list[Task]:
    path = path or (TASKS_DIR / "tasks.json")
    with open(path) as f:
        raw = json.load(f)
    return [Task(**t) for t in raw]


@dataclass
class EvalReport:
    outcomes: list[TaskOutcome]

    @property
    def passed(self) -> int:
        return sum(o.passed for o in self.outcomes)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def avg_steps(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.n_steps for o in self.outcomes) / len(self.outcomes)

    def summary(self) -> str:
        lines = [f"{o.name:<28} {'PASS' if o.passed else 'FAIL':<5} "
                 f"({o.n_steps} steps) {'' if o.passed else '- ' + o.reason}"
                 for o in self.outcomes]
        lines.append("-" * 50)
        lines.append(f"{self.passed}/{self.total} tasks passed, "
                     f"avg {self.avg_steps:.1f} model calls")
        return "\n".join(lines)


def run_evals(
    tasks: list[Task],
    make_model: Callable[[Task], Model],
    tools: ToolRegistry,
    *,
    max_steps: int = 8,
) -> EvalReport:
    """Run each task. ``make_model`` returns a model for a given task so that
    the offline fake can be scripted per-task; for the real model it just
    ignores the task argument."""
    outcomes = []
    for task in tasks:
        result = run(task.prompt, make_model(task), tools, max_steps=max_steps)
        passed, reason = check_task(task, result)
        outcomes.append(TaskOutcome(task.name, passed, reason,
                                    result.n_model_calls, result.answer))
    return EvalReport(outcomes)
