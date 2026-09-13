"""Stretch 4 skills — the skeletons the skills team fills in.

Import this module to register them:

    from conductor_planner.skills import stretch   # noqa: F401

Each class is a `ThreadedSkill`: write `on_start`, `step`, `on_stop`, and get
supervision for free. `docs/SKILL_INTERFACE.md` is the document to read first.

The three rules these must obey, in order of how much damage breaking them does:

1. **`step()` must be short and non-blocking.** It is called at `rate_hz`, and
   the gap between calls is the only moment the skill can be stopped. A
   `step()` that blocks for three seconds is a skill that ignores the stop
   signal for three seconds, while the supervisor has already decided the
   grasp is empty.

2. **`read_state()` must be cheap and honest.** It is polled at control rate,
   so read cached values -- never trigger a sensor read. And report
   `gripper_effort` as measured. The supervisor's free tier is built on that
   number; a nominal value there silently disables empty-grasp detection.

3. **Never report success.** `SkillReport` has no success field. Report how you
   stopped and where the hardware is. The monitor decides what it means.

`on_stop` is always called, whatever happened — a stop request, a timeout, an
exception. Leave the hardware safe there, and assume you may be half-way
through a motion.
"""
from __future__ import annotations

import math
import time
from typing import Any

from ..schema import StopCause
from .base import ThreadedSkill, register


class _StretchSkill(ThreadedSkill):
    """Shared plumbing: the robot handle, state reporting, hardware guards."""

    rate_hz = 20.0

    def read_state(self) -> dict[str, Any]:
        """Cheap snapshot for the supervisor. Cached values only."""
        r = self.robot
        if r is None:
            return {}
        try:
            st = r.get_state()
        except Exception:
            return {}
        return {
            "gripper_aperture": st.gripper_aperture,
            "gripper_effort": st.gripper_effort,     # MEASURED, never nominal
            "lift_m": st.lift_m,
            "arm_m": st.arm_m,
            "base_x": round(st.base_pose.x, 3),
            "base_y": round(st.base_pose.y, 3),
            "base_theta": round(st.base_pose.theta, 3),
        }

    def guard(self) -> str | None:
        """Hardware-level abort, checked every tick before step().

        Distinct from the supervisor: this one knows about the hardware, the
        supervisor knows about the task.
        """
        try:
            if self.robot.get_state().runstop_engaged:
                return "runstop engaged"
        except Exception:
            pass
        return None


@register
class StretchNavigate(_StretchSkill):
    """Drive the base.

    Two paths, and the difference matters for supervision:

    * `relative` — a short local nudge, issued **with the arm extended** after a
      failed pick. Drive it slowly on velocity commands and do not stow first;
      stowing would lose the pose the planner is trying to correct.
    * `target` / `pose` — a transit move. Send a Nav2 goal and poll it.

    The Nav2 path is the one to be careful with: `step()` must POLL the action,
    not wait on it. Blocking on `get_result()` would make the skill
    unstoppable, which defeats the entire design.
    """

    name = "navigate"
    default_timeout_s = 180.0

    def on_start(self) -> None:
        p = self.params
        self.relative = p.get("relative")
        self.tolerance_m = float(p.get("tolerance_m", 0.25))
        if self.relative:
            self._goal = None
            self._t_start = time.time()
            # TODO(you): begin publishing a slow velocity command on /cmd_vel.
            raise NotImplementedError(
                "StretchNavigate: relative nudge. Publish a slow cmd_vel toward "
                "the offset and integrate odometry in step(); stop when within "
                "tolerance. Must work with the arm extended.")
        target, pose = p.get("target"), p.get("pose")
        wp = self.robot.waypoints.get(target) if target else None
        if wp is not None:
            pt = wp.get("pose", wp)
            self._goal = (pt["x"], pt["y"], pt.get("theta", 0.0))
        elif pose:
            self._goal = (pose["x"], pose["y"], pose.get("theta", 0.0))
        else:
            raise ValueError(
                f"unknown waypoint {target!r}. Known: {sorted(self.robot.waypoints)}")
        if not self.robot.get_state().is_stowed:
            raise ValueError("refused: the arm is not stowed. Call stow() first. "
                             "(A small `relative` nudge is allowed with the arm out.)")
        # TODO(you): send the Nav2 goal, keep the goal handle, do not wait.
        raise NotImplementedError(
            "StretchNavigate: send a NavigateToPose goal and store the handle. "
            "step() polls it; it must not block.")

    def step(self) -> bool:
        # TODO(you): poll the Nav2 goal handle.
        #   still running and outside tolerance -> return True
        #   arrived                             -> return False
        #   aborted                             -> raise ValueError with the reason,
        #                                          written for a model to read
        raise NotImplementedError

    def on_stop(self, cause: StopCause) -> None:
        # TODO(you): cancel the Nav2 goal and publish a zero velocity. Always,
        # including on COMPLETED -- a cancel on an already-finished goal is
        # harmless, a base still rolling is not.
        pass

    def read_state(self) -> dict[str, Any]:
        """Report the resolved goal alongside the pose.

        REQUIRED, not optional. The supervisor's `base_at_target` predicate and
        the post-step `arrived_wrong_place` detector both compare the base pose
        against these fields. Omit them and a navigation that reports success
        while stopping in the wrong room goes undetected until something else
        fails for a confusing reason.
        """
        st = super().read_state()
        if getattr(self, "_goal", None) is not None:
            st.update({"goal_x": round(self._goal[0], 3),
                       "goal_y": round(self._goal[1], 3),
                       "goal_theta": round(self._goal[2], 3),
                       "tolerance_m": self.tolerance_m})
        return st

    def phase(self) -> str:
        if self._goal is None:
            return "nudging"
        try:
            b = self.robot.get_state().base_pose
            return f"driving, {math.hypot(self._goal[0] - b.x, self._goal[1] - b.y):.2f} m to go"
        except Exception:
            return "driving"


@register
class StretchPickUp(_StretchSkill):
    """Close on an object that is already in view and within reach.

    Local and contact-rich. It does not navigate and does not know the map: the
    planner guarantees the object is reachable before this starts.

    Structure it as reach → close → lift, and let `step()` return True through
    all of it. The lift phase is what gives the supervisor a window to catch an
    empty grasp while the arm is still over the table — end the skill the moment
    the gripper closes and you take that window away.
    """

    name = "pick_up"
    default_timeout_s = 45.0

    def on_start(self) -> None:
        self.object = self.params.get("object", "")
        self.target = self.params.get("target")
        # TODO(you): plan the approach. If the target is out of the workspace,
        # raise ValueError naming the distance -- the planner can fix staging
        # from that, and cannot from "IK failed".
        raise NotImplementedError(
            "StretchPickUp: plan the approach, then run reach -> close -> lift "
            "across step() calls.")

    def step(self) -> bool:
        # TODO(you): advance one control tick. Return False only after the lift.
        raise NotImplementedError

    def on_stop(self, cause: StopCause) -> None:
        # TODO(you): halt arm motion. Do NOT open the gripper here -- if the
        # supervisor stopped this for an empty grasp the gripper is already
        # empty, and if it stopped for something else you may be holding the
        # object. The planner decides; set_gripper is its tool for that.
        pass


@register
class StretchPutDown(_StretchSkill):
    """Place the held object on a surface already within reach.

    Local only. Structure it as approach → lower → release → retreat, and keep
    `release` honest: when `release=false` the object must still be in the
    gripper when the skill ends.
    """

    name = "put_down"
    default_timeout_s = 45.0

    def on_start(self) -> None:
        self.surface = self.params.get("surface", "")
        self.target = self.params.get("target")
        self.release = bool(self.params.get("release", True))
        # TODO(you): plan the placement; raise ValueError with the distance if
        # the surface is outside the workspace.
        raise NotImplementedError(
            "StretchPutDown: plan the placement, then run approach -> lower -> "
            "release -> retreat across step() calls.")

    def step(self) -> bool:
        raise NotImplementedError

    def on_stop(self, cause: StopCause) -> None:
        # TODO(you): halt. If you were interrupted mid-lower while still
        # holding the object, leave it held -- dropping it is worse than
        # stopping with it.
        pass
