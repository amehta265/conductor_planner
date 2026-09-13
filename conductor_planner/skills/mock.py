"""Mock skills for the scripted world.

These exist so the supervision loop can be tested without a robot: they take
real wall-clock time, advance in small increments, and can genuinely be stopped
half-way. A skill that completed instantly would make every supervision test
vacuous.

They duck-type on `MockRobot` rather than importing it, which keeps
`skills` free of any dependency on `robot`.
"""
from __future__ import annotations

import math
import time
from typing import Any

from ..schema import Point3D, Pose2D, StopCause
from .base import ThreadedSkill, register


class _MockSkill(ThreadedSkill):
    # Faster than any real controller. What supervision actually depends on is
    # the number of TICKS a phase lasts, not its wall-clock duration, so the
    # mock keeps realistic tick counts and compresses the time between them.
    # That keeps the test suite quick without making the tests less honest --
    # but it does mean the mock config raises `supervision.tick_hz` to match.
    rate_hz = 100.0

    def read_state(self) -> dict[str, Any]:
        r = self.robot
        if r is None:
            return {}
        b = r.world.base
        return {"gripper_aperture": r._gripper, "gripper_effort": r._effort,
                "base_x": round(b.x, 3), "base_y": round(b.y, 3),
                "base_theta": round(b.theta, 3),
                "holding": r._holding}

    def guard(self) -> str | None:
        if getattr(self.robot, "runstop", False):
            return "runstop engaged"
        return None

    def _nearest(self, hint: str):
        hint = (hint or "").lower()
        best, best_s = None, 0.0
        for o in self.robot.world.objects:
            if o.held:
                continue
            toks = set(o.name.lower().split())
            s = len(toks & set(hint.split())) / max(len(toks), 1)
            if s > best_s:
                best, best_s = o, s
        return best if best_s > 0 else None


@register
class MockNavigate(_MockSkill):
    """Drives toward the goal at a fixed speed, a little each tick."""

    name = "navigate"
    goal: Pose2D | None = None
    default_timeout_s = 30.0
    # Deliberately faster than a real base. The mock has to take real time --
    # a skill that finished instantly would make every supervision test vacuous
    # -- but only enough of it that the supervisor gets a decent number of ticks.
    # At 8 m/s and 100 Hz, crossing the flat is ~55 ticks and about half a
    # second, which gives the supervisor plenty of looks at it.
    speed_mps = 8.0

    def on_start(self) -> None:
        r = self.robot
        p = self.params
        self._blocked = r.fail_navigations > 0
        if self._blocked:
            r.fail_navigations -= 1
        self.goal: Pose2D | None = None
        if p.get("relative"):
            b = r.world.base
            rel = p["relative"]
            dx, dy = float(rel.get("dx", 0.0)), float(rel.get("dy", 0.0))
            self.goal = Pose2D(
                x=b.x + dx * math.cos(b.theta) - dy * math.sin(b.theta),
                y=b.y + dx * math.sin(b.theta) + dy * math.cos(b.theta),
                theta=b.theta + float(rel.get("dtheta", 0.0)))
        elif p.get("target") in r.world.waypoints:
            x, y, th = r.world.waypoints[p["target"]]
            self.goal = Pose2D(x=x, y=y, theta=th)
        elif p.get("pose"):
            ps = p["pose"]
            self.goal = Pose2D(x=ps["x"], y=ps["y"], theta=ps.get("theta", 0.0))
        else:
            self._error = (f"unknown waypoint {p.get('target')!r}; known: "
                           f"{sorted(r.world.waypoints)}")
            raise ValueError(self._error)
        if not p.get("relative") and not r._stowed:
            raise ValueError("refused: the arm is not stowed. Call stow() first. "
                             "(A small `relative` nudge is allowed with the arm out.)")
        self.tol = float(p.get("tolerance_m", 0.25))

    def read_state(self):
        """Report where we were TRYING to go, not just where we are.

        Without this the supervisor and the post-step detectors can compare the
        base pose against nothing, and a navigation that reports success while
        ending up in the wrong room passes silently.
        """
        st = super().read_state()
        if self.goal is not None:
            st.update({"goal_x": round(self.goal.x, 3),
                       "goal_y": round(self.goal.y, 3),
                       "goal_theta": round(self.goal.theta, 3),
                       "tolerance_m": self.tol})
        return st

    def step(self) -> bool:
        r = self.robot
        if self._blocked:
            raise ValueError("the global planner could not find a path (a chair may "
                             "be blocking the doorway). Try a different waypoint.")
        b, g = r.world.base, self.goal
        d = math.hypot(g.x - b.x, g.y - b.y)
        if d <= max(self.tol, 0.02):
            r.world.base = Pose2D(x=g.x, y=g.y, theta=g.theta)
            return False
        stepd = min(d, self.speed_mps / self.rate_hz)
        r.world.base = Pose2D(x=b.x + stepd * (g.x - b.x) / d,
                              y=b.y + stepd * (g.y - b.y) / d,
                              theta=g.theta)
        return True

    def phase(self) -> str:
        b, g = self.robot.world.base, self.goal
        return f"driving, {math.hypot(g.x - b.x, g.y - b.y):.2f} m to go"

    def progress(self) -> float | None:
        return None


@register
class MockPickUp(_MockSkill):
    """Reach, close, lift. Optionally closes on nothing, on purpose.

    The lift phase is not decoration. A real pick closes and then raises the
    object, and that gap is exactly when an empty grasp becomes detectable and
    still cheap to abort -- the gripper reads closed-with-no-load while the arm
    is still over the table. A mock that returned the instant it closed would
    make the supervision tests vacuous, because there would be nothing to catch.
    """

    name = "pick_up"
    default_timeout_s = 20.0
    reach_ticks = 10
    close_ticks = 6
    lift_ticks = 12

    def on_start(self) -> None:
        r = self.robot
        r._stowed = False
        p = self.params
        tgt = p.get("target")
        found = None
        if tgt:
            pt = tgt if isinstance(tgt, dict) else tgt.model_dump()
            found = min((o for o in r.world.objects if not o.held),
                        key=lambda o: math.hypot(o.x - pt["x"], o.y - pt["y"]),
                        default=None)
            if found and math.hypot(found.x - pt["x"], found.y - pt["y"]) > 0.35:
                found = None
        self.obj = found or self._nearest(p.get("object", ""))
        if self.obj is None:
            raise ValueError(f"nothing matching {p.get('object')!r} is in view")
        b = r.world.base
        d = math.hypot(self.obj.x - b.x, self.obj.y - b.y)
        if d > r.world.reach_m:
            raise ValueError(
                f"IK unreachable: the target is {d:.2f} m from the base and the "
                f"workspace ends at {r.world.reach_m} m. Move the base closer.")
        self.will_miss = r.fail_picks > 0
        if self.will_miss:
            r.fail_picks -= 1
        self._n = 0
        self._total = self.reach_ticks + self.close_ticks + self.lift_ticks

    def step(self) -> bool:
        r = self.robot
        self._n += 1
        if self._n <= self.reach_ticks:
            return True
        closing = self._n - self.reach_ticks
        if closing <= self.close_ticks:
            r._gripper = max(0.0, 1.0 - closing / self.close_ticks)
            if closing == self.close_ticks:
                r._gripper = 0.0
                if self.will_miss:
                    # closed on air. The real number is reported, not a nominal
                    # one -- the supervisor's free tier reads exactly this.
                    r._effort = 0.03
                else:
                    self.obj.held = True
                    r._holding = self.obj.name
                    r._effort = 0.55
            return True
        return self._n < self._total

    def phase(self) -> str:
        if self._n <= self.reach_ticks:
            return "reaching"
        if self._n <= self.reach_ticks + self.close_ticks:
            return "closing"
        return "lifting"

    def progress(self) -> float | None:
        return min(1.0, self._n / self._total)


@register
class MockPutDown(_MockSkill):
    """Moves the held object to the surface, then releases."""

    name = "put_down"
    default_timeout_s = 20.0
    move_ticks = 12

    def on_start(self) -> None:
        r = self.robot
        r._stowed = False
        if not r._holding:
            raise ValueError("the gripper is empty, so there is nothing to put down")
        p = self.params
        surf = self._nearest(p.get("surface", ""))
        tgt = p.get("target")
        if tgt:
            pt = tgt if isinstance(tgt, dict) else tgt.model_dump()
            surf = min(r.world.objects,
                       key=lambda o: math.hypot(o.x - pt["x"], o.y - pt["y"]))
        b = r.world.base
        self.dest = ((surf.x, surf.y, surf.z + 0.02) if surf
                     else (b.x + 0.5 * math.cos(b.theta),
                           b.y + 0.5 * math.sin(b.theta), 0.75))
        self.surf = surf
        if math.hypot(self.dest[0] - b.x, self.dest[1] - b.y) > r.world.reach_m + 0.2:
            raise ValueError(
                f"IK unreachable: the surface is "
                f"{math.hypot(self.dest[0] - b.x, self.dest[1] - b.y):.2f} m away, "
                "beyond the workspace")
        self.release = bool(p.get("release", True))
        self._n = 0

    def step(self) -> bool:
        r = self.robot
        self._n += 1
        if self._n < self.move_ticks:
            return True
        held = next(o for o in r.world.objects if o.name == r._holding)
        held.held = False
        held.x, held.y, held.z = self.dest[0] + 0.05, self.dest[1] + 0.05, self.dest[2]
        held.on = self.surf.name if self.surf else "unknown surface"
        held.room = self.surf.room if self.surf else held.room
        r._holding = None
        if self.release:
            r._gripper, r._effort = 1.0, 0.0
        return False

    def phase(self) -> str:
        return "carrying to the surface" if self._n < self.move_ticks else "releasing"

    def progress(self) -> float | None:
        return min(1.0, self._n / self.move_ticks)
