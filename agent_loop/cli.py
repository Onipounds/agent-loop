"""Command-line interface for agent-loop.

  agent-loop run "your task"        # real model (needs ANTHROPIC_API_KEY)
  agent-loop run --offline "task"   # deterministic offline demo, no key
  agent-loop eval [--offline]       # run the eval suite

The offline mode uses a tiny scripted planner (``offline.py``) so the whole
agent demonstrably works end-to-end with no network. The real mode uses the
Anthropic tool-use flow via AnthropicModel.
"""
from __future__ import annotations

import argparse
import sys

from .runtime import run
from .tools import build_default_registry
from .models import AnthropicModel
from .offline import make_offline_model
from .evals import load_tasks, run_evals


def _print_result(result) -> None:
    for i, step in enumerate(result.steps, 1):
        if step.kind == "tool_calls":
            for name, inp, out in step.detail:
                print(f"  step {i}: tool {name}({inp}) -> {out}")
        elif step.kind == "rejected":
            print(f"  step {i}: answer rejected -> {step.detail}")
        elif step.kind == "answer":
            print(f"  step {i}: final answer accepted")
    flag = "verified" if result.verified else "UNVERIFIED"
    print(f"\nAnswer [{flag}]: {result.answer}")


def cmd_run(args) -> int:
    tools = build_default_registry()
    if args.offline:
        model = make_offline_model(args.task, tools)
    else:
        model = AnthropicModel()
    result = run(args.task, model, tools, max_steps=args.max_steps)
    _print_result(result)
    return 0 if result.verified else 1


def cmd_eval(args) -> int:
    tools = build_default_registry()
    tasks = load_tasks()
    if args.offline:
        def make_model(task):
            return make_offline_model(task.prompt, tools)
    else:
        def make_model(task):
            return AnthropicModel()
    report = run_evals(tasks, make_model, tools, max_steps=args.max_steps)
    print(report.summary())
    return 0 if report.passed == report.total else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent-loop")
    parser.add_argument("--max-steps", type=int, default=8)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the agent on a task")
    p_run.add_argument("task")
    p_run.add_argument("--offline", action="store_true", help="use the offline scripted model")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="run the eval suite")
    p_eval.add_argument("--offline", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
