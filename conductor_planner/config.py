"""Config loading and object graph construction.

One YAML file describes a deployment: which models fill the planner, grounder
and monitor roles, which robot backend, where memory lives, how often memory is
re-scoped, and the semantic map. Everything that differs between the mock and
the real Stretch lives here rather than in code.

API keys never appear in a config. They are resolved by `backends.keys` from
`keys/<provider>.key` or the environment -- see `keys/README.md`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .backends import build_backend
from .executor import Conductor
from .grounding import Grounder
from .memory import GlobalMemory, TaskMemory
from .monitor import Monitor
from .planner import Planner
from .robot.base import RobotInterface


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    base = Path(path).parent
    inc = cfg.pop("include_map", None)
    if inc:
        cfg["semantic_map"] = yaml.safe_load((base / inc).read_text())
    return cfg


def build_robot(cfg: dict[str, Any]) -> RobotInterface:
    kind = cfg.get("kind", "mock")
    params = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "mock":
        from .robot.mock import MockRobot
        return MockRobot(**params)
    if kind == "stretch_ros2":
        from .robot.stretch_ros2 import StretchRobot
        return StretchRobot(**params)
    raise ValueError(f"unknown robot kind '{kind}'")


def build_from_config(path: str | Path, robot: RobotInterface | None = None,
                      task: str | None = None, connect_robot: bool = True,
                      connect_models: bool = True) -> Conductor:
    """Build the whole object graph from a config file.

    `connect_robot=False` substitutes the mock robot and `connect_models=False`
    substitutes stub backends, so you can inspect the assembled prompt and the
    retrieved trace with no ROS, no robot and no API key present. That combination is what `--dry-run` uses, and it is the cheapest
    way to check a prompt change.
    """
    cfg = load_config(path)
    task = task or cfg["task"]
    mem_dir = Path(cfg.get("memory_dir", "memory_store"))

    tm = TaskMemory(mem_dir / "task_traces.jsonl")
    gm = GlobalMemory(mem_dir / "global")

    if connect_models:
        planner_backend = build_backend(cfg["models"]["planner"])
        grounder_cfg = cfg["models"].get("grounder") or cfg["models"]["planner"]
        grounder_backend = (planner_backend if grounder_cfg is cfg["models"]["planner"]
                            else build_backend(grounder_cfg))
        monitor_cfg = cfg["models"].get("monitor")
        monitor_backend = build_backend(monitor_cfg) if monitor_cfg else grounder_backend
    else:
        stub = {"kind": "scripted", "responses": [], "name": "dry-run"}
        planner_backend = build_backend(stub)
        grounder_backend = build_backend(stub)
        monitor_backend = build_backend(stub)

    if robot is None:
        robot_cfg = dict(cfg.get("robot", {"kind": "mock"}))
        if not connect_robot:
            robot_cfg = {"kind": "mock"}
        elif robot_cfg.get("kind") == "stretch_ros2":
            # the semantic map is the robot's waypoint table as well as the
            # planner's, so there is only ever one copy of it
            robot_cfg.setdefault("waypoints",
                                 (cfg.get("semantic_map") or {}).get("waypoints", {}))
        robot = build_robot(robot_cfg)

    planner = Planner(
        planner_backend, task, tm, gm,
        semantic_map=cfg.get("semantic_map"),
        history_window=cfg.get("history_window", 6),
        attach_images=cfg.get("attach_images", 1),
        extra_notes=cfg.get("notes", ""),
        temperature=cfg.get("temperature", 0.0),
        scoped_memory=cfg.get("scoped_memory", True),
        retrieve_on_subgoal_change=cfg.get("retrieve_on_subgoal_change", True),
        retrieve_after_failures=cfg.get("retrieve_after_failures", 3),
    )
    grounder = Grounder(
        grounder_backend,
        mode=cfg.get("grounding_mode", "point"),
        assumed_object_height_m=cfg.get("assumed_object_height_m", 0.08),
    )
    monitor = Monitor(monitor_backend, gm, stall_limit=cfg.get("stall_limit", 3))

    return Conductor(
        robot=robot, planner=planner, grounder=grounder, monitor=monitor,
        task_memory=tm, global_memory=gm,
        mode=cfg.get("mode", "deploy"),
        max_steps=cfg.get("max_steps", 30),
        supervision=cfg.get("supervision") or {},
        log_dir=cfg.get("log_dir"),
    )
