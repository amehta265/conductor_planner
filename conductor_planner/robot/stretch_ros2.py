"""Stretch 4 adapter.

This is the ONLY file in the package that imports ROS 2 or knows what a Stretch
is. It is a skeleton in the sense that a handful of methods are marked TODO
where they depend on facts about your particular robot (topic names after your
remapping, which controller your arm is on). Everything structural is here and
runs.

Navigation, picking and placing are NOT methods on this class. They are skills
-- startable, stoppable objects in `conductor_planner/skills/stretch.py` -- because
the supervisor has to be able to stop them mid-motion.

Two Stretch-specific facts are baked in because they will bite you otherwise,
both from the lab's own debug log:

  * `stow['arm'] = 0.0` is unreachable -- the arm bottoms out around 0.0048 m,
    so the stow routine times out at 6 s forever. The fix is a user-params
    override to 0.02; `stow()` below checks for it at startup and says so
    loudly rather than failing mysteriously mid-episode.

  * The head camera may enumerate as an OAK-D-S2 when it is an OAK-FFC-3P,
    which makes the camera TF geometry wrong. Every 3D point this class
    produces inherits that error. `check_camera_frame()` flags it, because a
    systematic 3-5 cm offset in every grasp target is very hard to debug from
    the planner's end and trivial to spot here.

Run `python -m conductor_planner.robot.stretch_ros2 --selftest` on the robot to
exercise the interface without the planner.
"""
from __future__ import annotations

import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..grounding import CameraInfo
from ..schema import (Outcome, Point3D, Pose2D, PrimitiveName, RobotState,
                      StepResult, StopPredicate)
from .base import RobotInterface, fail, ok


# Practical limits for a Stretch 4. Measure yours and correct these once.
LIFT_RANGE_M = (0.15, 1.10)
ARM_RANGE_M = (0.02, 0.52)
REACH_M = 0.80
STOW_ARM_MIN = 0.015          # anything below this cannot be reached; see docstring


class StretchRobot(RobotInterface):
    def __init__(
        self,
        node_name: str = "conductor_planner",
        rgb_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/aligned_depth_to_color/image_raw",
        camera_info_topic: str = "/camera/color/camera_info",
        scan_topic: str = "/scan",
        joint_states_topic: str = "/stretch/joint_states",
        nav_action: str = "/navigate_to_pose",
        head_trajectory_action: str = "/stretch_controller/follow_joint_trajectory",
        map_frame: str = "map",
        base_frame: str = "base_link",
        camera_frame: str = "camera_color_optical_frame",
        waypoints: dict[str, Any] | None = None,
        workdir: str | Path | None = None,
        connect: bool = True,
    ):
        self.map_frame, self.base_frame, self.camera_frame = map_frame, base_frame, camera_frame
        self.waypoints = waypoints or {}
        self.dir = Path(workdir or tempfile.mkdtemp(prefix="conductor_stretch_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._frame = 0
        self._topics = dict(rgb=rgb_topic, depth=depth_topic, info=camera_info_topic,
                            scan=scan_topic, joints=joint_states_topic)
        self._nav_action = nav_action
        self._head_action = head_trajectory_action
        self.node = None
        from ..skills import load_stretch
        load_stretch()          # replaces the mock skills with the real ones
        if connect:
            self._connect(node_name)

    # ------------------------------------------------------------ ROS init
    def _connect(self, node_name: str) -> None:
        import rclpy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo as CameraInfoMsg
        from sensor_msgs.msg import Image, JointState, LaserScan
        from tf2_ros import Buffer, TransformListener

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = Node(node_name)
        self.bridge = CvBridge()
        self._PoseStamped = PoseStamped
        self._NavigateToPose = NavigateToPose

        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()

        def keep(key: str):
            def cb(msg):
                with self._lock:
                    self._latest[key] = msg
            return cb

        self.node.create_subscription(Image, self._topics["rgb"], keep("rgb"), sensor_qos)
        self.node.create_subscription(Image, self._topics["depth"], keep("depth"), sensor_qos)
        self.node.create_subscription(CameraInfoMsg, self._topics["info"], keep("info"), 10)
        self.node.create_subscription(LaserScan, self._topics["scan"], keep("scan"), sensor_qos)
        self.node.create_subscription(JointState, self._topics["joints"], keep("joints"), 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.nav_client = ActionClient(self.node, NavigateToPose, self._nav_action)

        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        self._wait_for_topics()
        self.check_stow_param()
        self.check_camera_frame()

    def _wait_for_topics(self, timeout: float = 15.0) -> None:
        t0 = time.time()
        need = {"rgb", "info", "joints"}
        while time.time() - t0 < timeout:
            with self._lock:
                if need <= set(self._latest):
                    return
            time.sleep(0.2)
        with self._lock:
            missing = need - set(self._latest)
        raise RuntimeError(
            f"no messages on {sorted(missing)} after {timeout} s. "
            "Check `ros2 topic list` and, if /stretch_driver is absent, check that "
            "it did not die at import -- see the lab's NumPy 2 / transforms3d note."
        )

    # ------------------------------------------------------------- checks
    def check_stow_param(self) -> None:
        """Refuse to start with the known-bad factory stow target."""
        try:
            from stretch4_body.core.robot_params import RobotParams
            p = RobotParams().get_params()[1]
            stow = p[p["robot"]["tool"]]["stow"]
            if float(stow.get("arm", 0.0)) < STOW_ARM_MIN:
                raise RuntimeError(
                    f"stow['arm'] = {stow.get('arm')} is below the arm's hard stop "
                    "(~0.0048 m), so every stow() will time out after 6 s. Add\n"
                    "  <tool>:\n    stow:\n      arm: 0.02\n"
                    "to ~/stretch_user/<robot>/stretch_user_params.yaml and RESTART "
                    "stretch_body_server -- params load at server startup only."
                )
        except ImportError:
            pass    # not on the robot; fine

    def check_camera_frame(self) -> None:
        """Warn if the head camera enumerated as the wrong model."""
        try:
            with self._lock:
                info = self._latest.get("info")
            if info is not None and getattr(info, "width", 0) and self.node:
                self.node.get_logger().info(
                    f"head camera {info.width}x{info.height}, frame {info.header.frame_id}. "
                    "If this robot reported 'OAK-FFC-3P not found, falling back to "
                    "OAK-D-S2', the camera TF is wrong and every 3D point from this "
                    "class carries that error."
                )
        except Exception:
            pass

    # ------------------------------------------------------------ sensing
    def get_state(self) -> RobotState:
        with self._lock:
            js = self._latest.get("joints")
        pos = {}
        if js is not None:
            pos = dict(zip(js.name, js.position))
            eff = dict(zip(js.name, js.effort)) if js.effort else {}
        else:
            eff = {}
        base = self._base_pose()
        arm = sum(pos.get(f"joint_arm_l{i}", 0.0) for i in range(4)) or pos.get("wrist_extension", 0.0)
        grip = pos.get("joint_gripper_finger_left", 0.0)
        # Map the Stretch gripper joint (roughly -0.35 closed .. 0.17 open) to 0..1
        aperture = max(0.0, min(1.0, (grip + 0.35) / 0.52))
        effort = abs(eff.get("joint_gripper_finger_left", 0.0)) / 40.0
        return RobotState(
            base_pose=base,
            lift_m=pos.get("joint_lift"),
            arm_m=arm,
            gripper_aperture=aperture,
            gripper_effort=min(1.0, effort),
            head_pan_rad=pos.get("joint_head_pan"),
            head_tilt_rad=pos.get("joint_head_tilt"),
            holding=self._holding,
            is_stowed=(arm is not None and arm < 0.05
                       and (pos.get("joint_lift") or 0.0) < 0.45),
            runstop_engaged=False,      # TODO: subscribe to /is_runstopped
        )

    _holding: str | None = None    # set by the pick_up / put_down skills

    def _base_pose(self) -> Pose2D:
        T = self._lookup(self.map_frame, self.base_frame)
        if T is None:
            return Pose2D(x=0.0, y=0.0, theta=0.0)
        yaw = math.atan2(T[1, 0], T[0, 0])
        return Pose2D(x=float(T[0, 3]), y=float(T[1, 3]), theta=float(yaw))

    def _lookup(self, target: str, source: str) -> np.ndarray | None:
        """4x4 transform that maps a point in `source` into `target`."""
        try:
            from rclpy.time import Time
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def capture(self):
        import cv2
        with self._lock:
            rgb_msg = self._latest.get("rgb")
            depth_msg = self._latest.get("depth")
            info = self._latest.get("info")
        if rgb_msg is None or info is None:
            raise RuntimeError("no camera frame available")
        self._frame += 1
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        path = self.dir / f"rgb_{self._frame:05d}.png"
        cv2.imwrite(str(path), rgb)
        depth = None
        if depth_msg is not None:
            d = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            # 16UC1 is millimetres; 32FC1 is already metres
            depth = (d.astype(np.float32) / 1000.0) if d.dtype == np.uint16 else d.astype(np.float32)
        cam = CameraInfo(fx=info.k[0], fy=info.k[4], cx=info.k[2], cy=info.k[5],
                         width=info.width, height=info.height,
                         frame_id=info.header.frame_id or self.camera_frame)
        T = self._lookup(self.map_frame, cam.frame_id)
        return str(path), depth, cam, T

    def scan_summary(self) -> str | None:
        with self._lock:
            scan = self._latest.get("scan")
        if scan is None:
            return None
        r = np.asarray(scan.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max), r, np.nan)
        if np.all(np.isnan(r)):
            return "lidar returned no valid ranges"
        i = int(np.nanargmin(r))
        ang = math.degrees(scan.angle_min + i * scan.angle_increment)
        n = len(r)
        fwd = r[max(0, n // 2 - 20): n // 2 + 20]
        fwd_min = float(np.nanmin(fwd)) if not np.all(np.isnan(fwd)) else float("nan")
        return (f"nearest obstacle {float(np.nanmin(r)):.2f} m at {ang:+.0f} deg; "
                f"clear ahead to {fwd_min:.2f} m")

    def semantic_context(self) -> list[str]:
        b = self._base_pose()
        out = []
        for name, info in self.waypoints.items():
            p = info.get("pose", info) if isinstance(info, dict) else {}
            if "x" not in p:
                continue
            d = math.hypot(p["x"] - b.x, p["y"] - b.y)
            out.append((d, f"{name} at ({p['x']:.1f}, {p['y']:.1f}), {d:.1f} m away"
                           + (f" -- {info.get('description', '')}" if isinstance(info, dict) else "")))
        out.sort()
        return [s for _, s in out[:4]]

    # ---------------------------------------------------------- analytic
    # ------------------------------------------------------------- skills
    # navigate / pick_up / put_down are NOT methods here. They are startable,
    # stoppable objects in conductor_planner/skills/stretch.py, built through
    # RobotInterface.skill(). That is what lets the supervisor watch them while
    # they run and stop them mid-motion; a blocking method could only ever be
    # judged after it had finished failing.

    def look_at(self, target=None, point=None, pan_rad=None, tilt_rad=None) -> StepResult:
        """TODO(you): send a FollowJointTrajectory goal for joint_head_pan/tilt.

        The geometry is done for you: if `point` is given, we convert it into a
        pan/tilt pair in the base frame. Wire the trajectory goal to
        `self._head_action` and wait for the result.
        """
        if point is not None:
            T = self._lookup(self.base_frame, self.map_frame)
            if T is not None:
                v = T @ np.array([point.x, point.y, point.z, 1.0])
                pan_rad = math.atan2(v[1], v[0])
                tilt_rad = math.atan2(v[2] - 1.25, math.hypot(v[0], v[1]))
        if target and pan_rad is None and tilt_rad is None:
            tilt_rad = {"table_surface": -0.55, "floor": -0.9,
                        "ahead": -0.1, "up": 0.2}.get(target, -0.4)
            pan_rad = 0.0
        raise NotImplementedError(
            "look_at: send pan=%.2f tilt=%.2f to %s" % (pan_rad or 0.0, tilt_rad or 0.0,
                                                        self._head_action))

    def set_gripper(self, state: str, effort: float = 0.4) -> StepResult:
        """TODO(you): FollowJointTrajectory on joint_gripper_finger_left.

        Open is about +0.17 rad, closed about -0.35 rad on a stock PG4.
        """
        raise NotImplementedError("set_gripper")

    def stow(self, carry: bool = False) -> StepResult:
        """TODO(you): call stretch_body's stow routine, or command lift/arm directly.

        If you call the vendor routine, make sure the stow['arm'] override from
        check_stow_param() is in place and the server has been restarted since.
        For carry=true you want a different pose: arm retracted but lift held at
        carrying height, so you do not scrape whatever you are holding.
        """
        raise NotImplementedError("stow")

    def close(self) -> None:
        try:
            self._executor.shutdown()
            self.node.destroy_node()
        except Exception:
            pass


def _spin_until(rclpy, node, future, timeout: float):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if future.done():
            return future.result()
        time.sleep(0.05)
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = StretchRobot()
        print("state:", r.get_state())
        p, d, c, T = r.capture()
        print("rgb:", p, "depth:", None if d is None else d.shape, "cam:", c)
        print("tf map<-camera:", "ok" if T is not None else "MISSING")
        print("scan:", r.scan_summary())
        try:
            from ..skills import available
            print("registered skills:", available())
        except Exception as e:
            print("skill registry: --", e)
