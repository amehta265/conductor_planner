#!/usr/bin/env python3
"""Run the glass task end to end in the mock world, with no models.

    python scripts/run_mock.py --fail-picks 1 -v

First thing to run after cloning, and after any change to the conductor.

Watch the pick_up lines: the injected empty grasp should be "stopped by monitor"
part-way through, not diagnosed once the skill finished. The vlm column should
stay at 0 throughout -- supervision is free by default.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conductor_planner import GlobalMemory, Grounder, Conductor, Monitor, Planner, TaskMemory
from conductor_planner.robot.mock import MockRobot
from conductor_planner.testing import OracleGrounder, OraclePlanner

TASK = "Go to the living room, pick up a glass, and place it on the kitchen table"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-picks", type=int, default=1,
                    help="inject N empty grasps; the supervisor should stop each "
                         "one mid-lift rather than diagnosing it afterwards")
    ap.add_argument("--memory", default="memory_store/mock")
    ap.add_argument("--fresh", action="store_true", help="wipe memory first")
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    mem = Path(a.memory)
    if a.fresh and mem.exists():
        shutil.rmtree(mem)

    robot = MockRobot(fail_picks=a.fail_picks)
    tm = TaskMemory(mem / "task_traces.jsonl")
    gm = GlobalMemory(mem / "global")

    planner = Planner(OraclePlanner(robot), TASK, tm, gm)
    grounder = Grounder(OracleGrounder(robot))
    monitor = Monitor(OracleGrounder(robot), gm)

    def show(call, res):
        mark = {"success": "ok  ", "failure": "FAIL", "rejected": "REJ ",
                "timeout": "TIME"}.get(res.outcome.value, "?   ")
        sup = (f" [{res.supervisor_ticks} ticks, {res.vlm_checks} vlm]"
               if res.supervisor_ticks else "")
        print(f"  {mark} {call.action.value:<12}{sup:<22} {res.message[:88]}")
        if res.monitor_verdict and a.verbose:
            print(f"       monitor: {res.monitor_verdict[:150]}")

    h = Conductor(robot, planner, grounder, monitor, tm, gm,
                max_steps=a.max_steps, log_dir="logs/mock", on_step=show,
                supervision={"tick_hz": 50.0})   # mock skills run compressed

    print(f"TASK: {TASK}\n")
    ep = h.run(TASK)
    print(f"\nsucceeded={ep.succeeded}  steps={len(ep.calls)}")
    print(f"reason: {ep.termination_reason}")
    print(f"traces in memory: {len(tm.traces)}")
    if tm.traces:
        print("\nabstracted trace now in Task-Specific Memory:")
        for i, s in enumerate(tm.traces[-1].steps, 1):
            print(f"  {i:>2}. {s.action.value:<20} {s.args}")
    return 0 if ep.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
