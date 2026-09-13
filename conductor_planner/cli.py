"""Command-line entry points."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run(argv: list[str] | None = None) -> int:
    """Run a task on whatever robot the config names."""
    ap = argparse.ArgumentParser(prog="conductor-run")
    ap.add_argument("config", help="path to a YAML config, e.g. configs/stretch_glass_task.yaml")
    ap.add_argument("--task", default=None, help="override the task string")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--bootstrap", action="store_true",
                    help="exploratory mode: generous budget, write what worked to memory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the assembled system prompt and retrieved trace, "
                         "then exit. Needs no robot and no API key")
    a = ap.parse_args(argv)

    from .config import build_from_config, load_config

    cfg = load_config(a.config)
    task = a.task or cfg["task"]
    h = build_from_config(a.config, task=task,
                          connect_robot=not a.dry_run,
                          connect_models=not a.dry_run)
    if a.bootstrap:
        h.mode = "bootstrap"
        h.max_steps = a.max_steps or 60
    elif a.max_steps:
        h.max_steps = a.max_steps

    if a.dry_run:
        print(h.planner.system)
        print("\n--- retrieved trace ---\n")
        print(h.tm.render(task))
        return 0

    def show(call, res):
        mark = {"success": "ok  ", "failure": "FAIL", "rejected": "REJ "}.get(
            res.outcome.value, "?   ")
        print(f"  {mark} {call.action.value:<20} {res.message[:100]}", flush=True)
        if res.monitor_verdict:
            print(f"       monitor: {res.monitor_verdict[:160]}", flush=True)
    h.on_step = show

    print(f"TASK: {task}\n")
    ep = h.run(task)
    print(f"\nsucceeded={ep.succeeded}  steps={len(ep.calls)}\n{ep.termination_reason}")
    for role, backend in (("planner", h.planner.backend),
                          ("grounder", h.grounder.backend),
                          ("monitor", h.monitor.backend)):
        report = getattr(backend, "cost_report", None)
        if callable(report):
            print(f"  {role}: {report()}")
    return 0 if ep.succeeded else 1


def mock(argv: list[str] | None = None) -> int:
    """Shortcut for the offline oracle run."""
    here = Path(__file__).resolve().parent.parent
    sys.argv = [sys.argv[0], *(argv or sys.argv[1:])]
    exec((here / "scripts" / "run_mock.py").read_text(), {"__name__": "__main__"})
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
