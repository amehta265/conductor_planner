"""The robot seam.

Everything above this file is embodiment-agnostic. Everything below it knows
about Stretch. There is exactly one class to implement per robot, and the
planner never imports a vendor package.

This mirrors the L1 capability API in the two-tier HAL doc: the planner sits at
L4 and talks to L1, never to L0.

Each method returns a `StepResult`. A method that cannot do the thing returns
`Outcome.FAILURE` with a message written *for the planner to read* -- the
message goes straight into the next prompt, so "IK unreachable: the target is
0.94 m from the base and the arm extends to 0.52 m" is a useful message and
"RuntimeError: solver failed" is not.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

from ..grounding import CameraInfo
from ..schema import (Detection, Observation, Outcome, Point3D, Pose2D,
                      PrimitiveName, RobotState, SkillReport, StepResult,
                      StopPredicate)

if TYPE_CHECKING:                       # avoid a cycle: skills import schema
    from ..skills import Skill


class RobotInterface(abc.ABC):
    """Capability API.

    Sensing, four analytic primitives that run to completion in milliseconds,
    and a factory for the three skills, which do not. The split is exactly the
    supervision boundary: analytic primitives are too fast to be worth watching,
    skills are slow enough that watching them is the point.
    """

    # ------------------------------------------------------------ sensing
    @abc.abstractmethod
    def get_state(self) -> RobotState: ...

    @abc.abstractmethod
    def capture(self) -> tuple[str, np.ndarray | None, CameraInfo, np.ndarray | None]:
        """Return (rgb_path, depth_metres_or_None, camera_info, T_cam_to_map)."""

    def scan_summary(self) -> str | None:
        """One line of LiDAR context for the prompt.

        Deliberately a *summary*, not a point cloud. A VLM cannot reason over
        30k points, but "nearest obstacle 0.42 m at -25 deg; clear ahead to
        2.1 m" is exactly what stops it planning a base motion into a sofa.
        Returning None is fine.
        """
        return None

    def semantic_context(self) -> list[str]:
        """Named waypoints near the robot, for the prompt."""
        return []

    # -------------------------------------------------------------- skills
    def skill(self, name: str, **params: Any) -> "Skill":
        """Build a startable skill. Does NOT start it.

        The executor starts it, supervises it while it runs, and stops it.
        That is the whole reason skills are objects rather than method calls:
        a blocking call can only be judged after it finishes.

        The default implementation looks the name up in the skill registry and
        hands over `self` as the robot handle. Override only if a robot needs
        to build its skills differently.
        """
        from ..skills import Skill
        return Skill(name, robot=self, **params)

    # ------------------------------------------------------------ analytic
    @abc.abstractmethod
    def look_at(self, target: str | None = None, point: Point3D | None = None,
                pan_rad: float | None = None,
                tilt_rad: float | None = None) -> StepResult: ...

    @abc.abstractmethod
    def set_gripper(self, state: str, effort: float = 0.4) -> StepResult: ...

    @abc.abstractmethod
    def stow(self, carry: bool = False) -> StepResult: ...

    # -------------------------------------------------------------- hooks
    def emergency_stop(self) -> None:
        """Best-effort halt. Safety proper lives in the vendor's runstop."""

    def close(self) -> None:
        ...


def ok(action: PrimitiveName, call_id: str, message: str = "", **kw: Any) -> StepResult:
    return StepResult(call_id=call_id, action=action, outcome=Outcome.SUCCESS,
                      message=message, **kw)


def fail(action: PrimitiveName, call_id: str, message: str, **kw: Any) -> StepResult:
    return StepResult(call_id=call_id, action=action, outcome=Outcome.FAILURE,
                      message=message, **kw)
