"""A scripted mock world.

Purpose: make every path through the conductor -- staging, grasping, failure
detection, re-staging, memory abstraction, goal verification -- runnable with
no robot, no GPU and no network, in under a second. You debug the planner here
and you take a working planner to the robot, instead of debugging prompt
formatting while a 25 kg machine waits with its arm out.

The world is deliberately small: rooms, waypoints, objects with 3D positions,
a base pose, a gripper. It renders a crude RGB image per viewpoint so the
grounding path is exercised end to end rather than stubbed.

Failure injection is the point of `fail_picks`: set it to 1 and the first pick
comes up empty, which is the single most important recovery path in the whole
system and the one you cannot test on hardware without wasting an hour.
"""
from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..grounding import CameraInfo, pose2d_to_matrix
from ..schema import (Outcome, Point3D, Pose2D, PrimitiveName, RobotState,
                      StepResult, StopPredicate)
from .base import RobotInterface, fail, ok


@dataclass
class MockObject:
    name: str
    x: float
    y: float
    z: float
    color: tuple[int, int, int] = (170, 200, 230)
    held: bool = False
    room: str = "living_room"
    on: str = "coffee_table"
    extent: tuple[float, float] = (0.0, 0.0)   # a surface has area; a mug does not


@dataclass
class MockWorld:
    waypoints: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        "living_room": (1.0, 0.5, 0.0),
        "coffee_table": (2.2, 0.9, 0.15),
        "kitchen": (5.0, 2.0, 1.57),
        "kitchen_table": (5.6, 2.6, 1.2),
        "charger": (0.0, 0.0, 3.14),
    })
    objects: list[MockObject] = field(default_factory=lambda: [
        MockObject("drinking glass", 2.8, 1.05, 0.74, (200, 225, 240)),
        MockObject("remote control", 2.6, 0.7, 0.72, (40, 40, 45), on="coffee_table"),
        MockObject("kitchen table", 6.0, 2.8, 0.75, (150, 110, 80), room="kitchen",
                   on="floor", extent=(1.40, 0.80)),
        MockObject("coffee table", 2.7, 0.9, 0.45, (140, 100, 70), on="floor",
                   extent=(1.10, 0.60)),
    ])
    base: Pose2D = field(default_factory=lambda: Pose2D(x=0.0, y=0.0, theta=0.0))
    reach_m: float = 0.75


class MockRobot(RobotInterface):
    def __init__(
        self,
        world: MockWorld | None = None,
        workdir: str | Path | None = None,
        fail_picks: int = 0,
        fail_navigations: int = 0,
        seed: int = 0,
    ):
        self.world = world or MockWorld()
        self.dir = Path(workdir or tempfile.mkdtemp(prefix="conductor_mock_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fail_picks = fail_picks
        self.fail_navigations = fail_navigations
        self.rng = np.random.default_rng(seed)
        self._stowed = True
        self._gripper = 1.0            # 1.0 open, 0.0 closed
        self._effort = 0.0
        self._holding: str | None = None
        self._pan, self._tilt = 0.0, -0.3
        self.runstop = False
        self._frame = 0
        self.log: list[str] = []
        self.cam = CameraInfo(fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                              width=640, height=480, frame_id="camera_color_optical_frame")

    # ------------------------------------------------------------ sensing
    def get_state(self) -> RobotState:
        return RobotState(
            base_pose=self.world.base, lift_m=0.6, arm_m=0.05,
            gripper_aperture=self._gripper, gripper_effort=self._effort,
            head_pan_rad=self._pan, head_tilt_rad=self._tilt,
            holding=self._holding, is_stowed=self._stowed,
            battery_soc=0.78, runstop_engaged=False,
        )

    def _camera_matrix(self) -> np.ndarray:
        """camera optical frame -> map, for a head camera at 1.25 m."""
        b = self.world.base
        yaw = b.theta + self._pan
        # optical (x right, y down, z fwd) -> ROS body (x fwd, y left, z up)
        R = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])
        # head tilt is negative-is-down (the Stretch convention), while a
        # positive rotation about +Y pitches the optical axis down -- hence the
        # sign flip. Getting this backwards puts every object out of frame and
        # looks exactly like a detection failure, so it is worth a comment.
        pitch = -self._tilt
        Rp = np.array([[math.cos(pitch), 0.0, math.sin(pitch)],
                       [0.0, 1.0, 0.0],
                       [-math.sin(pitch), 0.0, math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                       [math.sin(yaw), math.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = Ry @ Rp @ R
        T[:3, 3] = [b.x, b.y, 1.25]
        return T

    def visible(self) -> list[MockObject]:
        """Objects in front of the robot, within 4 m and a ~70 deg cone."""
        b = self.world.base
        out = []
        for o in self.world.objects:
            if o.held:
                continue
            dx, dy = o.x - b.x, o.y - b.y
            d = math.hypot(dx, dy)
            if d > 4.0:
                continue
            ang = math.atan2(dy, dx) - (b.theta + self._pan)
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) < math.radians(35):
                out.append(o)
        return out

    def project(self, o: MockObject) -> tuple[int, int] | None:
        T = self._camera_matrix()
        Tinv = np.linalg.inv(T)
        p = Tinv @ np.array([o.x, o.y, o.z, 1.0])
        if p[2] <= 0.1:
            return None
        u = self.cam.fx * p[0] / p[2] + self.cam.cx
        v = self.cam.fy * p[1] / p[2] + self.cam.cy
        if not (0 <= u < self.cam.width and 0 <= v < self.cam.height):
            return None
        return int(u), int(v)

    def _draw_surface(self, d, depth, o: MockObject) -> None:
        """Rasterise a table top by projecting a grid of points on its plane."""
        b = self.world.base
        w, h = o.extent
        nu, nv = 26, 18
        pts = []
        for i in range(nu):
            for j in range(nv):
                x = o.x - w / 2 + w * i / (nu - 1)
                y = o.y - h / 2 + h * j / (nv - 1)
                probe = MockObject(o.name, x, y, o.z, o.color)
                uv = self.project(probe)
                if uv is None:
                    continue
                pts.append((uv, math.hypot(x - b.x, y - b.y)))
        if not pts:
            return
        for (u, v), dist in sorted(pts, key=lambda p: -p[1]):
            r = max(4, int(700 / max(dist * 100, 1)))
            d.rectangle([u - r, v - r, u + r, v + r], fill=o.color)
            depth[max(0, v - r):v + r, max(0, u - r):u + r] = dist
        us = [p[0][0] for p in pts]; vs = [p[0][1] for p in pts]
        d.text((min(us), min(vs) - 12), o.name, fill=(20, 20, 20))

    def capture(self):
        self._frame += 1
        img = Image.new("RGB", (self.cam.width, self.cam.height), (225, 220, 210))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 300, 640, 480], fill=(180, 165, 150))     # floor
        depth = np.full((self.cam.height, self.cam.width), 3.5, np.float32)
        b = self.world.base
        for o in sorted(self.visible(), key=lambda o: -math.hypot(o.x - b.x, o.y - b.y)):
            if o.extent[0] > 0:
                self._draw_surface(d, depth, o)
                continue
            uv = self.project(o)
            if uv is None:
                continue
            u, v = uv
            dist = math.hypot(o.x - b.x, o.y - b.y)
            r = max(8, int(2200 / max(dist * 100, 1)))
            d.rectangle([u - r, v - r, u + r, v + r], fill=o.color, outline=(30, 30, 30))
            d.text((u - r, v - r - 12), o.name, fill=(20, 20, 20))
            depth[max(0, v - r):v + r, max(0, u - r):u + r] = dist
            if "glass" in o.name:
                # transparent objects return invalid depth -- the real failure mode
                depth[max(0, v - r):v + r, max(0, u - r):u + r] = 0.0
        path = self.dir / f"rgb_{self._frame:04d}.png"
        img.save(path)
        np.save(self.dir / f"depth_{self._frame:04d}.npy", depth)
        return str(path), depth, self.cam, self._camera_matrix()

    def scan_summary(self) -> str:
        b = self.world.base
        near = min((math.hypot(o.x - b.x, o.y - b.y) for o in self.world.objects
                    if not o.held), default=9.9)
        return f"nearest obstacle {near:.2f} m; free space ahead {min(near, 3.0):.1f} m"

    def semantic_context(self) -> list[str]:
        b = self.world.base
        items = sorted(self.world.waypoints.items(),
                       key=lambda kv: math.hypot(kv[1][0] - b.x, kv[1][1] - b.y))
        return [f"{n} at ({p[0]:.1f}, {p[1]:.1f}), {math.hypot(p[0]-b.x, p[1]-b.y):.1f} m away"
                for n, p in items[:4]]

    # ---------------------------------------------------------- analytic
    def look_at(self, target=None, point=None, pan_rad=None, tilt_rad=None) -> StepResult:
        if point is not None:
            b = self.world.base
            self._pan = math.atan2(point.y - b.y, point.x - b.x) - b.theta
            self._pan = (self._pan + math.pi) % (2 * math.pi) - math.pi
            self._tilt = -0.45
        if pan_rad is not None:
            self._pan = float(pan_rad)
        if tilt_rad is not None:
            self._tilt = float(tilt_rad)
        if target and point is None and pan_rad is None and tilt_rad is None:
            named = {"table_surface": -0.55, "floor": -0.9, "ahead": -0.1, "up": 0.2}
            self._tilt = named.get(target, -0.4)
            for o in self.world.objects:
                if target.lower() in o.name.lower():
                    b = self.world.base
                    self._pan = ((math.atan2(o.y - b.y, o.x - b.x) - b.theta + math.pi)
                                 % (2 * math.pi) - math.pi)
        self.log.append(f"look_at pan={self._pan:.2f} tilt={self._tilt:.2f}")
        return ok(PrimitiveName.LOOK_AT, "mock",
                  f"head at pan {self._pan:.2f}, tilt {self._tilt:.2f}")

    def set_gripper(self, state: str, effort: float = 0.4) -> StepResult:
        if state == "open":
            if self._holding:
                held = next(o for o in self.world.objects if o.name == self._holding)
                held.held = False
                b = self.world.base
                held.x, held.y = b.x + 0.5 * math.cos(b.theta), b.y + 0.5 * math.sin(b.theta)
                held.z = 0.0
                held.on = "floor"
                self._holding = None
            self._gripper, self._effort = 1.0, 0.0
        else:
            self._gripper = 0.0
            self._effort = 0.6 if self._holding else 0.02
        self.log.append(f"set_gripper {state}")
        return ok(PrimitiveName.SET_GRIPPER, "mock",
                  f"gripper {state}; aperture {self._gripper:.2f}, effort {self._effort:.2f}")

    def stow(self, carry: bool = False) -> StepResult:
        if self._holding and not carry:
            return fail(PrimitiveName.STOW, "mock",
                        f"refused: holding '{self._holding}'. Pass carry=true to stow "
                        "while carrying, or place the object first.")
        self._stowed = True
        self.log.append("stow")
        return ok(PrimitiveName.STOW, "mock", "arm stowed")

    # ----------------------------------------------------------- learned
    # ------------------------------------------------------------- skills
    # navigate / pick_up / put_down live in conductor_planner/skills/mock.py and
    # are built by RobotInterface.skill(). They are startable and stoppable, so
    # the supervision loop has something real to interrupt.

    def _nearest(self, name_hint: str) -> MockObject | None:
        hint = name_hint.lower()
        best, best_s = None, 0.0
        for o in self.world.objects:
            if o.held:
                continue
            toks = set(o.name.lower().split())
            s = len(toks & set(hint.split())) / max(len(toks), 1)
            if s > best_s:
                best, best_s = o, s
        return best if best_s > 0 else None
