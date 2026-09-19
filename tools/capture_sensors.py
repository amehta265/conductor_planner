#!/usr/bin/env python3
"""Record every sensors/* topic for a few seconds, then write an HTML report.

Runs on the WORKSTATION, with RobotNode already running. This is a diagnostic
tool for checking the sensor relays, not part of the stack -- nothing in
agentic_robotics imports it.

    python3 tools/capture_sensors.py --seconds 20

Writes sensor_capture_<timestamp>/ next to wherever you ran it, holding the
recorded samples and report.html. Open report.html in a browser: every relayed
topic gets a row saying whether anything arrived, how fast, and what the data
actually looked like.

The topic list comes from RobotNode's own SENSOR_RELAYS table, so this tool
cannot drift away from what the node publishes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
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


class SensorCapture(Node):
    """Subscribe to everything RobotNode publishes and remember what arrives."""

    def __init__(self, directory: Path, seconds: float):
        super().__init__("sensor_capture")
        self.directory = directory
        self.images = directory / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.image_spacing = seconds / FRAMES_PER_CAMERA
        self.started = time.monotonic()
        self.records: dict[str, dict] = {}

        topics = [
            (stack_topic, message_type, source)
            for _, source, message_type, stack_topic in SENSOR_RELAYS
        ]
        topics.append((SENSOR_SCAN_TOPIC, LaserScan, "/scan_filtered"))
        # Not relays: RobotNode has always published these, decoded from the
        # demo's ZeroMQ telemetry rather than from any ROS topic.
        topics.append(
            (OBSERVATIONS_TOPIC, Observation, "ZeroMQ telemetry (gripper camera)")
        )
        topics.append(
            (ROBOT_STATE_TOPIC, RobotState, "ZeroMQ telemetry (joints, gripper)")
        )

        for stack_topic, message_type, source in topics:
            self.records[stack_topic] = {
                "topic": "/" + stack_topic,
                "type": message_type.__module__.split(".")[0] + "/" + message_type.__name__,
                "source": source,
                "times": [],
                "bytes": [],
                "frames": [],
                "detail": None,
                "_last_frame_at": -1e9,
            }
            self.create_subscription(
                message_type,
                stack_topic,
                lambda message, key=stack_topic: self.on_message(key, message),
                qos_profile_sensor_data,
            )

    def on_message(self, key: str, message) -> None:
        now = time.monotonic() - self.started
        record = self.records[key]
        record["times"].append(round(now, 3))
        record["bytes"].append(len(serialize_message(message)))
        record["detail"] = describe(message)

        kind = type(message).__name__
        if kind in ("CompressedImage", "Observation"):
            self._maybe_save_frame(key, record, message, now)
        elif kind == "OccupancyGrid":
            try:
                save_costmap_png(message, self.directory / "costmap.png")
            except Exception as error:               # noqa: BLE001 - diagnostic tool
                record["detail"]["image_error"] = str(error)

    def _maybe_save_frame(self, key, record, message, now) -> None:
        """Keep a handful of frames, spread across the capture window."""
        if len(record["frames"]) >= FRAMES_PER_CAMERA:
            return
        if now - record["_last_frame_at"] < self.image_spacing:
            return
        record["_last_frame_at"] = now
        stem = f"{key.replace('/', '_')}_{len(record['frames']):02d}"

        try:
            if type(message).__name__ == "Observation":
                name = f"{stem}.png"
                save_rgb_png(message.rgb, self.images / name)
                # Depth changes with the scene, so refresh it alongside the
                # colour frame rather than on every 20 Hz message.
                save_depth_png(message.depth, self.images / "observation_depth.png")
            else:
                suffix = "jpg" if "jpeg" in message.format.lower() else "bin"
                name = f"{stem}.{suffix}"
                (self.images / name).write_bytes(bytes(message.data))
        except Exception as error:                   # noqa: BLE001 - diagnostic tool
            record["detail"]["image_error"] = str(error)
            return

        record["frames"].append({"file": f"images/{name}", "at": round(now, 3)})

    def result(self, seconds: float) -> dict:
        topics = []
        for record in self.records.values():
            record = dict(record)
            record.pop("_last_frame_at", None)
            times, sizes = record["times"], record["bytes"]
            record["count"] = len(times)
            record["hz"] = round(len(times) / seconds, 2) if times else 0.0
            record["avg_bytes"] = int(sum(sizes) / len(sizes)) if sizes else 0
            record["bytes_per_s"] = int(sum(sizes) / seconds) if sizes else 0
            record.pop("bytes")
            topics.append(record)
        return {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": seconds,
            "topics": topics,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", default=None, help="Output directory")
    arguments = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(arguments.out or f"sensor_capture_{stamp}")
    directory.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = SensorCapture(directory, arguments.seconds)
    print(f"Recording {len(node.records)} topics for {arguments.seconds:g} s ...")
    deadline = time.monotonic() + arguments.seconds
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    result = node.result(arguments.seconds)
    (directory / "capture.json").write_text(json.dumps(result, indent=2))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    from build_sensor_report import build_report
    report = build_report(directory)

    silent = [t["topic"] for t in result["topics"] if t["count"] == 0]
    print(f"\nWrote {report}")
    print(f"{len(result['topics']) - len(silent)} topics received data, {len(silent)} silent")
    for topic in silent:
        print(f"  silent: {topic}")


if __name__ == "__main__":
    main()
