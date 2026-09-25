#!/usr/bin/env python3
"""Record every sensors/* topic for a few seconds, then write an HTML report.

Runs on the WORKSTATION, with RobotNode already running. This is a diagnostic
tool for checking the sensor relays, not part of the stack -- nothing in
agentic_robotics imports it.

    python3 tools/capture_sensors.py --seconds 20 --robot hello-robot@<robot-ip>

Writes sensor_capture_<timestamp>/ next to wherever you ran it, holding the
recorded samples and report.html. Open report.html in a browser: every relayed
topic gets a row saying whether anything arrived, how fast, how late, and what
the data actually looked like.

Latency comes from timestamps the middleware already carries, so nothing in
the stack is instrumented:

    header.stamp           when the driver says the data was captured
    DDS source timestamp   when the publisher wrote the message (its clock)
    DDS reception time     when this machine had the whole message (our clock)

To see the robot's side of each relay it also subscribes to the robot's
original topic, so the robot sends those topics twice while this runs.
--stack-only turns that off, at the cost of the per-stage breakdown.

The robot-to-workstation hop compares two clocks. --robot measures their
offset over SSH (key-based login needed) before and after recording, and the
report corrects for it. Without it the report assumes the clocks agree and
says so.

The topic list comes from RobotNode's own SENSOR_RELAYS table and system.yaml,
so this tool cannot drift away from what the node publishes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import struct
import subprocess
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rclpy.subscription import Subscription
from sensor_msgs.msg import LaserScan

from agentic_robotics_interfaces.msg import Observation, RobotState

from agentic_robotics.robot_node import SENSOR_RELAYS
from agentic_robotics.ros_conversions import bgr_from_image
from agentic_robotics.ros_names import (
    OBSERVATIONS_TOPIC,
    ROBOT_STATE_TOPIC,
    SENSOR_SCAN_TOPIC,
)

FRAMES_PER_CAMERA = 6          # saved image frames, spread over the capture
SCAN_PLOT_BEAMS = 720          # the scan is downsampled to this for plotting
CLOUD_PLOT_POINTS = 2000
THUMBNAIL_PX = 480            # saved camera frames are shrunk to this
DEPTH_NEAR_M, DEPTH_FAR_M = 0.15, 1.50   # wrist camera working range

DISCOVERY_S = 2.0             # let DDS discovery finish before looking at the graph
MATCH_S = 1.0                 # let late subscriptions match before recording
CLOCK_EXCHANGES = 40
# Best effort like every sensor publisher, but a deep queue: if Python falls
# behind, messages wait here (their DDS reception time is already fixed)
# instead of being dropped and skewing the counts.
CAPTURE_QOS = QoSProfile(
    depth=50,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
CONFIG = Path(__file__).resolve().parent.parent / "src/agentic_robotics/config/system.yaml"
# Head cameras the robot publishes that RobotNode does not relay (the center one).
HEAD_CAMERA = re.compile(r"^/cameras_head/[^/]+/image_raw/compressed$")
# The robot prints time.time_ns() for every line it reads: one SSH session,
# many round trips, so the exchange with the shortest trip bounds the offset.
REMOTE_CLOCK = (
    "python3 -u -c 'import sys, time\n"
    "while sys.stdin.readline():\n"
    "    print(time.time_ns(), flush=True)'"
)


def describe(message) -> dict:
    """A small JSON-safe summary of one message, for the report to render."""
    kind = type(message).__name__

    if kind == "JointState":
        return {
            "kind": "joints",
            "joints": [
                {"name": name, "position": round(float(position), 4)}
                for name, position in zip(message.name, message.position)
            ],
        }

    if kind == "Odometry":
        point = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return {
            "kind": "pose",
            "frame": message.header.frame_id,
            "x": round(point.x, 3),
            "y": round(point.y, 3),
            "yaw_deg": round(math.degrees(yaw), 1),
        }

    if kind == "BatteryState":
        return {
            "kind": "battery",
            "voltage": round(float(message.voltage), 2),
            "current": round(float(message.current), 2),
            "percentage": round(float(message.percentage), 3),
        }

    if kind in ("String", "Bool"):
        return {"kind": "value", "value": message.data}

    if kind == "CameraInfo":
        return {
            "kind": "camera_info",
            "frame": message.header.frame_id,
            "width": message.width,
            "height": message.height,
            "fx": round(message.k[0], 2),
            "fy": round(message.k[4], 2),
            "cx": round(message.k[2], 2),
            "cy": round(message.k[5], 2),
        }

    if kind == "CompressedImage":
        return {
            "kind": "image",
            "frame": message.header.frame_id,
            "format": message.format,
            "bytes": len(message.data),
        }

    if kind == "LaserScan":
        return _describe_scan(message)

    if kind == "OccupancyGrid":
        return _describe_costmap(message)

    if kind == "PointCloud2":
        return _describe_cloud(message)

    if kind == "Observation":
        return _describe_observation(message)

    if kind == "RobotState":
        return _describe_robot_state(message)

    return {"kind": "unknown"}


def _describe_scan(message) -> dict:
    ranges = list(message.ranges)
    returns = [
        r for r in ranges
        if math.isfinite(r) and message.range_min <= r <= message.range_max
    ]
    step = max(1, len(ranges) // SCAN_PLOT_BEAMS)
    plot = [
        round(r, 3)
        if math.isfinite(r) and message.range_min <= r <= message.range_max
        else None
        for r in ranges[::step]
    ]
    return {
        "kind": "scan",
        "frame": message.header.frame_id,
        "beams": len(ranges),
        "returns": len(returns),
        "coverage_pct": round(100.0 * len(returns) / max(len(ranges), 1), 1),
        "closest_m": round(min(returns), 3) if returns else None,
        "range_max": round(float(message.range_max), 2),
        "angle_min": float(message.angle_min),
        "angle_step": float(message.angle_increment) * step,
        "plot": plot,
    }


def _describe_costmap(message) -> dict:
    return {
        "kind": "costmap",
        "frame": message.header.frame_id,
        "width": message.info.width,
        "height": message.info.height,
        "resolution_m": round(float(message.info.resolution), 3),
        "extent_m": round(message.info.width * message.info.resolution, 2),
        "occupied_cells": int(sum(1 for value in message.data if value > 50)),
        "unknown_cells": int(sum(1 for value in message.data if value < 0)),
        "image": "costmap.png",
    }


def _describe_cloud(message) -> dict:
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if message.point_step == 0 or raw.size < message.point_step:
        return {"kind": "points", "count": 0, "points": []}
    raw = raw[: (raw.size // message.point_step) * message.point_step]
    raw = raw.reshape(-1, message.point_step)

    offsets = {field.name: field.offset for field in message.fields}
    if not {"x", "y", "z"} <= set(offsets):
        return {"kind": "points", "count": raw.shape[0], "points": []}

    def axis(name):
        start = offsets[name]
        return raw[:, start:start + 4].copy().view(np.float32).ravel()

    points = np.stack([axis("x"), axis("y"), axis("z")], axis=1)
    points = points[np.isfinite(points).all(axis=1)]
    step = max(1, len(points) // CLOUD_PLOT_POINTS)
    return {
        "kind": "points",
        "frame": message.header.frame_id,
        "count": int(len(points)),
        "points": [[round(float(v), 4) for v in p] for p in points[::step]],
    }


def _describe_observation(message) -> dict:
    """The wrist RGB-D that arrives over ZeroMQ, not as a ROS topic."""
    info = message.camera_info
    return {
        "kind": "observation",
        "frame": message.header.frame_id,
        "camera_id": message.camera_id,
        "width": int(message.rgb.width),
        "height": int(message.rgb.height),
        "rgb_encoding": message.rgb.encoding,
        "depth_encoding": message.depth.encoding,
        "fx": round(float(info.k[0]), 2),
        "fy": round(float(info.k[4]), 2),
        "cx": round(float(info.k[2]), 2),
        "cy": round(float(info.k[5]), 2),
        "depth_image": "images/observation_depth.png",
    }


def _describe_robot_state(message) -> dict:
    """RobotState carries the demo telemetry as a JSON string."""
    try:
        payload = json.loads(message.state_json or "{}")
    except ValueError as error:
        return {"kind": "state_json", "fields": [["parse error", str(error)]]}
    fields = []
    for key, value in sorted(payload.items()):
        if isinstance(value, float):
            value = round(value, 4)
        text = str(value)
        fields.append([key, text[:120] + "..." if len(text) > 120 else text])
    return {"kind": "state_json", "fields": fields}


def _depth_metres(message) -> np.ndarray:
    """Depth image as float32 metres, whichever encoding it arrived in."""
    if message.encoding == "16UC1":
        raw = np.frombuffer(bytes(message.data), dtype=np.uint16)
        metres = raw.astype(np.float32) / 1000.0
    elif message.encoding == "32FC1":
        metres = np.frombuffer(bytes(message.data), dtype=np.float32)
    else:
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    return metres[: message.height * message.width].reshape(
        message.height, message.width
    )


def save_depth_png(message, path: Path) -> None:
    """Depth as one blue ramp: near is dark, far is light, no reading is gray."""
    from PIL import Image

    metres = _depth_metres(message)
    near, far = (0x0D, 0x36, 0x6B), (0xCD, 0xE2, 0xFB)
    nothing = (0xE1, 0xE0, 0xD9)

    scaled = np.clip(
        (metres - DEPTH_NEAR_M) / max(DEPTH_FAR_M - DEPTH_NEAR_M, 1e-6), 0.0, 1.0
    )
    picture = np.zeros(metres.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        picture[..., channel] = np.round(
            near[channel] + scaled * (far[channel] - near[channel])
        )
    picture[~(np.isfinite(metres) & (metres > 0))] = nothing

    image = Image.fromarray(picture)
    image.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
    image.save(path)


def save_rgb_png(message, path: Path) -> None:
    """One raw sensor_msgs/Image frame as a PNG."""
    from PIL import Image

    image = Image.fromarray(bgr_from_image(message)[..., ::-1])
    image.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
    image.save(path)


def save_costmap_png(message, path: Path) -> None:
    """Occupancy grid as a small raster: blue ramp for cost, gray for unknown."""
    from PIL import Image

    low, high, unknown = (0xCD, 0xE2, 0xFB), (0x0D, 0x36, 0x6B), (0xE1, 0xE0, 0xD9)
    pixels = []
    for value in message.data:
        if value < 0:
            pixels.append(unknown)
        else:
            t = min(max(value, 0), 100) / 100.0
            pixels.append(tuple(
                int(round(low[i] + t * (high[i] - low[i]))) for i in range(3)
            ))

    width, height = message.info.width, message.info.height
    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)   # ROS origin is bottom-left
    scale = max(1, 320 // max(width, 1))
    image.resize((width * scale, height * scale), Image.NEAREST).save(path)


# ----------------------------------------------------------------- timing
def header_of(raw: bytes) -> tuple[int, str]:
    """(stamp in ns, frame_id) read straight from the serialized bytes.

    Every stamped message here starts with std_msgs/Header, which CDR lays out
    right after the 4-byte encapsulation: int32 sec, uint32 nanosec, then the
    frame_id string. Reading 16 bytes beats deserialising a 2 MB image.
    """
    order = "<" if raw[1] & 1 else ">"
    sec, nanosec, length = struct.unpack_from(order + "iII", raw, 4)
    frame = bytes(raw[16:16 + max(length - 1, 0)]).decode("utf-8", "replace")
    return sec * 1_000_000_000 + nanosec, frame


def starts_with_header(message_type) -> bool:
    return next(iter(message_type.get_fields_and_field_types()), None) == "header"


class MessageInfoExecutor(SingleThreadedExecutor):
    """Humble's executor drops the DDS message info; this one passes it on.

    rclpy from Iron on hands it to any two-argument callback by itself, so
    this override is only used on Humble.
    """

    def _take_subscription(self, sub):
        with sub.handle:
            return sub.handle.take_message(sub.msg_type, sub.raw)

    async def _execute_subscription(self, sub, taken):
        if taken:
            sub.callback(*taken)


def make_executor():
    if hasattr(Subscription, "CallbackType"):        # Iron, Jazzy and later
        return SingleThreadedExecutor()
    return MessageInfoExecutor()


def measure_clock_offset(host: str) -> dict:
    """Robot clock minus this machine's, NTP style, over one SSH session."""
    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        host, REMOTE_CLOCK,
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
    except OSError as error:
        return {"error": str(error)}
    best = None
    try:
        for exchange in range(CLOCK_EXCHANGES):
            sent = time.time_ns()
            process.stdin.write("\n")
            process.stdin.flush()
            wait_s = 10.0 if exchange == 0 else 2.0     # first one pays for login
            if not select.select([process.stdout], [], [], wait_s)[0]:
                break
            line = process.stdout.readline().strip()
            returned = time.time_ns()
            if not line.isdigit():
                break
            round_trip = returned - sent
            if exchange and (best is None or round_trip < best[0]):
                best = (round_trip, int(line) - (sent + returned) // 2)
    except (OSError, ValueError):
        pass
    finally:
        process.kill()
        error = process.communicate()[1].strip()
    if best is None:
        return {"error": error or "no reply from the robot"}
    return {"offset_ns": best[1], "uncertainty_ns": best[0] // 2}


def combine_clock(host, before: dict, after: dict) -> dict:
    good = [c for c in (before, after) if "offset_ns" in c]
    if not good:
        return {"host": host, "error": before.get("error") or after.get("error")}
    return {
        "host": host,
        "offset_ns": sum(c["offset_ns"] for c in good) // len(good),
        "uncertainty_ns": max(c["uncertainty_ns"] for c in good),
        "drift_ns": (
            after["offset_ns"] - before["offset_ns"] if len(good) == 2 else None
        ),
        "before": before,
        "after": after,
    }


def relay_parameters() -> dict:
    """RobotNode's parameters from system.yaml, so source topics match the node's."""
    try:
        import yaml
        return yaml.safe_load(CONFIG.read_text())["robot"]["ros__parameters"]
    except Exception:                               # noqa: BLE001 - use defaults
        return {}


class SensorCapture(Node):
    """Subscribe to everything RobotNode publishes and remember what arrives."""

    def __init__(self, directory: Path, seconds: float, robot_topics: bool):
        super().__init__("sensor_capture")
        self.directory = directory
        self.images = directory / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.image_spacing = seconds / FRAMES_PER_CAMERA
        self.robot_topics = robot_topics
        self.recording = False
        self.started_ns = 0
        self.records: dict[str, dict] = {}
        self.robot_states: list[bytes] = []

        parameters = relay_parameters()
        topics = [
            (stack_topic, message_type, parameters.get(name, default))
            for name, default, message_type, stack_topic in SENSOR_RELAYS
        ]
        topics.append(
            (SENSOR_SCAN_TOPIC, LaserScan, parameters.get("scan_topic", "/scan_filtered"))
        )
        for stack_topic, message_type, source in topics:
            if not source:
                continue                            # relay turned off in system.yaml
            self._add(stack_topic, "/" + stack_topic, message_type, "relay", source)
            self._subscribe(stack_topic, "samples", stack_topic)
            if robot_topics:
                self._subscribe(stack_topic, "source_samples", source)

        # Not relays: RobotNode has always published these, decoded from the
        # demo's ZeroMQ telemetry rather than from any ROS topic.
        for key, message_type, source in (
            (OBSERVATIONS_TOPIC, Observation, "ZeroMQ telemetry (gripper camera)"),
            (ROBOT_STATE_TOPIC, RobotState, "ZeroMQ telemetry (joints, gripper)"),
        ):
            self._add(key, "/" + key, message_type, "zeromq", source)
            self._subscribe(key, "samples", key)

    def _add(self, key, topic, message_type, path, source):
        self.records[key] = {
            "key": key,
            "topic": topic,
            "type": message_type.__module__.split(".")[0] + "/" + message_type.__name__,
            "source": source,
            "path": path,
            "samples": [],
            "source_samples": [],
            "frames": [],
            "detail": None,
            "_message_type": message_type,
            "_stamped": starts_with_header(message_type),
            "_shown": "source_samples" if path == "robot" else "samples",
            "_camera": message_type.__name__ in ("CompressedImage", "Observation"),
            "_frames": [],
            "_last": None,
            "_last_frame_at": -1e9,
        }

    def _subscribe(self, key, side, topic):
        self.create_subscription(
            self.records[key]["_message_type"],
            topic,
            partial(self.on_message, key, side),
            CAPTURE_QOS,
            raw=True,
        )

    def add_robot_only_topics(self, extra: list[str]) -> None:
        """Head cameras RobotNode does not relay, plus anything asked for by name.

        Also notes what type the robot advertises for each relay's source, the
        usual reason a relay is silent.
        """
        from rosidl_runtime_py.utilities import get_message

        graph = dict(self.get_topic_names_and_types())
        for record in self.records.values():
            if record["path"] == "relay":
                record["advertised"] = graph.get(record["source"], [])
        if not self.robot_topics:
            return
        relayed = {r["source"] for r in self.records.values()}
        wanted = [t for t in graph if HEAD_CAMERA.match(t) and t not in relayed]
        for topic in wanted + [t for t in extra if t not in wanted]:
            if topic not in graph:
                self.get_logger().warning(f"{topic} is not on the network; skipped")
                continue
            message_type = get_message(graph[topic][0])
            self._add(topic, topic, message_type, "robot", "robot only (not relayed)")
            self._subscribe(topic, "source_samples", topic)

    def start(self) -> None:
        self.started_ns = time.time_ns()
        self.recording = True

    def on_message(self, key: str, side: str, raw: bytes, info) -> None:
        """Hot path: timestamps and size only. Decoding waits until the end."""
        callback_ns = time.time_ns()
        if not self.recording:
            return
        record = self.records[key]
        stamp, frame = header_of(raw) if record["_stamped"] else (None, None)
        record[side].append([
            stamp, frame, info["source_timestamp"], info["received_timestamp"],
            callback_ns, len(raw),
        ])
        if side != record["_shown"]:
            return
        record["_last"] = raw
        if key == ROBOT_STATE_TOPIC:
            self.robot_states.append(raw)
        at = (callback_ns - self.started_ns) / 1e9
        if (
            record["_camera"]
            and len(record["_frames"]) < FRAMES_PER_CAMERA
            and at - record["_last_frame_at"] >= self.image_spacing
        ):
            record["_last_frame_at"] = at
            record["_frames"].append((at, raw))

    # ------------------------------------------------ after recording stops
    def finish(self) -> None:
        """Decode the few messages the report shows, now that timing is done."""
        for key, record in self.records.items():
            if record["_last"] is None:
                continue
            message = deserialize_message(record["_last"], record["_message_type"])
            record["detail"] = describe(message)
            if type(message).__name__ == "OccupancyGrid":
                try:
                    save_costmap_png(message, self.directory / "costmap.png")
                except Exception as error:           # noqa: BLE001 - diagnostic tool
                    record["detail"]["image_error"] = str(error)
            for at, raw in record["_frames"]:
                self._save_frame(key, record, at, raw)

        telemetry = self.records[ROBOT_STATE_TOPIC]
        telemetry["telemetry"] = [self._sender_time(raw) for raw in self.robot_states]

    def _save_frame(self, key, record, at, raw) -> None:
        message = deserialize_message(raw, record["_message_type"])
        stem = f"{key.strip('/').replace('/', '_')}_{len(record['frames']):02d}"
        try:
            if type(message).__name__ == "Observation":
                name = f"{stem}.png"
                save_rgb_png(message.rgb, self.images / name)
                save_depth_png(message.depth, self.images / "observation_depth.png")
            else:
                suffix = "jpg" if "jpeg" in message.format.lower() else "bin"
                name = f"{stem}.{suffix}"
                (self.images / name).write_bytes(bytes(message.data))
        except Exception as error:                   # noqa: BLE001 - diagnostic tool
            record["detail"]["image_error"] = str(error)
            return
        record["frames"].append({"file": f"images/{name}", "at": round(at, 3)})

    @staticmethod
    def _sender_time(raw: bytes) -> list:
        """[stamp, sender's system_timestamp in ns, image_number] for one message."""
        stamp, _ = header_of(raw)
        try:
            payload = json.loads(deserialize_message(raw, RobotState).state_json)
            sender = payload.get("system_timestamp")
            sender_ns = round(float(sender) * 1e9) if sender is not None else None
            return [stamp, sender_ns, payload.get("image_number")]
        except (ValueError, TypeError):
            return [stamp, None, None]

    def result(self, seconds: float) -> dict:
        topics = []
        for record in self.records.values():
            shown = record[record["_shown"]]
            sizes = [s[5] for s in shown]
            topic = {k: v for k, v in record.items() if not k.startswith("_")}
            topic["times"] = [
                round(((s[3] or s[4]) - self.started_ns) / 1e9, 3) for s in shown
            ]
            topic["count"] = len(shown)
            topic["hz"] = round(len(shown) / seconds, 2)
            topic["avg_bytes"] = int(sum(sizes) / len(sizes)) if sizes else 0
            topic["bytes_per_s"] = int(sum(sizes) / seconds)
            topics.append(topic)
        return {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": seconds,
            "started_ns": self.started_ns,
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
            "rmw": _rmw_name(),
            "robot_topics": self.robot_topics,
            "topics": topics,
        }


def _rmw_name() -> str:
    try:
        from rclpy.utilities import get_rmw_implementation_identifier
        return get_rmw_implementation_identifier()
    except Exception:                               # noqa: BLE001 - just a label
        return os.environ.get("RMW_IMPLEMENTATION", "")


def spin_for(executor, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument(
        "--robot", metavar="USER@HOST",
        help="SSH target used to measure the robot's clock offset",
    )
    parser.add_argument(
        "--stack-only", action="store_true",
        help="Only subscribe to sensors/*; no per-stage breakdown",
    )
    parser.add_argument(
        "--extra", action="append", default=[], metavar="TOPIC",
        help="Also time this robot topic (repeatable)",
    )
    arguments = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(arguments.out or f"sensor_capture_{stamp}")
    directory.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = SensorCapture(directory, arguments.seconds, not arguments.stack_only)
    executor = make_executor()
    executor.add_node(node)
    spin_for(executor, DISCOVERY_S)
    node.add_robot_only_topics(arguments.extra)
    spin_for(executor, MATCH_S)

    clock_before = measure_clock_offset(arguments.robot) if arguments.robot else {}
    print(f"Recording {len(node.records)} topics for {arguments.seconds:g} s ...")
    node.start()
    try:
        spin_for(executor, arguments.seconds)
    except KeyboardInterrupt:
        pass
    node.recording = False
    seconds = (time.time_ns() - node.started_ns) / 1e9
    clock_after = measure_clock_offset(arguments.robot) if arguments.robot else {}

    node.finish()
    result = node.result(seconds)
    result["clock"] = (
        combine_clock(arguments.robot, clock_before, clock_after)
        if arguments.robot else None
    )
    (directory / "capture.json").write_text(json.dumps(result))
    executor.shutdown()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    from build_sensor_report import build_report
    report = build_report(directory)
    print_summary(result, report)


def print_summary(result: dict, report: Path) -> None:
    from sensor_latency import analyse

    latency = analyse(result)
    clock = latency["clock"]
    print(f"\nWrote {report}")
    if clock["measured"]:
        print(
            f"Robot clock offset {clock['offset_ms']:+.2f} ms "
            f"(+/- {clock['uncertainty_ms']:.2f} ms), corrected for"
        )
    elif result.get("clock"):
        print(f"Clock offset NOT measured: {clock['error']}")
    print(f"\n{'topic':<44}{'msgs':>7}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}")
    for topic in result["topics"]:
        age = latency["topics"][topic["key"]]["end_to_end"]
        row = f"{topic['topic']:<44}{topic['count']:>7}"
        if age:
            row += f"{age['p50']:>9.1f}{age['p95']:>9.1f}{age['max']:>9.1f}"
        elif not topic["count"]:
            row += "   silent"
        print(row)


if __name__ == "__main__":
    main()
