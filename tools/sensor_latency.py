"""Latency analysis for a sensor capture. Pure Python: no ROS, no numpy.

capture_sensors.py records, for every message, four clocks' worth of facts:

    stamp     header.stamp -- when the driver says the data was captured
    sent      DDS source timestamp -- when the publisher wrote the message
    received  DDS reception timestamp -- when this machine had all of it
    callback  when the capture tool's Python callback ran

This module joins them into per-stage delays. A relayed topic goes through

    stamp --on robot--> robot publishes --network--> workstation receives
          --RobotNode--> sensors/* published --delivery--> subscriber has it

and the wrist camera, which arrives over ZeroMQ rather than a ROS topic, through

    sender stamps it --network (ZeroMQ)--> RobotNode decodes
          --RobotNode--> observations published --delivery--> subscriber has it

Stages on one machine use one clock and are exact. The two "network" stages
compare the robot's clock with this one, so they are only as good as the clock
offset capture_sensors.py measured (or zero if it did not).
"""
from __future__ import annotations

import math
from collections import defaultdict

ROLES = ("on robot", "network", "RobotNode", "delivery")
SAMPLE = ("stamp", "frame", "sent", "received", "callback", "bytes")
SERIES_POINTS = 400          # latency-over-time points kept per topic
SCATTER_POINTS = 2500        # size-vs-crossing points kept overall
STAMP_TRUST_MS = (-50.0, 1000.0)   # plausible "stamp -> publish" on one clock


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(values_ms) -> dict | None:
    """The summary every table and chart uses. None when there is nothing."""
    values = [v for v in values_ms if v is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "p5": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _ms(ns):
    return None if ns is None else ns / 1e6


def _rows(samples):
    return [dict(zip(SAMPLE, sample)) for sample in samples or []]


def _arrival(row):
    """DDS reception time, or the callback time when the RMW does not give one."""
    return row["received"] or row["callback"]


def _thin(points, limit):
    step = max(1, math.ceil(len(points) / limit))
    return points[::step]


def rate(sent_ns: list[int]) -> dict | None:
    """Publisher-side rate and regularity, from DDS source timestamps."""
    times = sorted(t for t in sent_ns if t)
    if len(times) < 3:
        return None
    gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    period = percentile(gaps, 0.5)
    mean = sum(gaps) / len(gaps)
    return {
        "hz": 1000.0 * (len(times) - 1) / max(sum(gaps), 1e-9),
        "period_ms": period,
        "jitter_ms": math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps)),
        "max_gap_ms": max(gaps),
        # A gap of nearly two periods or more almost certainly lost a message.
        "long_gaps": sum(1 for g in gaps if period and g > 1.8 * period),
    }


def _stamp_check(rows, offset_ns=0):
    """Is header.stamp on the publisher's clock? Some sensors stamp with their own.

    Returns (trusted, median stamp->publish ms). offset_ns is only non-zero
    when all we have are relayed copies, published on this machine's clock.
    """
    delays = [
        (r["sent"] - r["stamp"] + offset_ns) / 1e6
        for r in rows if r["stamp"] and r["sent"]
    ]
    if not delays:
        return False, None
    median = percentile(delays, 0.5)
    low, high = STAMP_TRUST_MS
    return (min(delays) >= low and median <= high), median


def _stage(name, role, values):
    return {"name": name, "role": role, "stats": stats(values)}


def _per_frame(rows, offset_ns, trusted):
    """Split a topic by frame_id when several sensors share it (the two lidars)."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["frame"]].append(row)
    if len(groups) < 2:
        return None
    result = {}
    for frame, members in sorted(groups.items()):
        result[frame] = {
            "count": len(members),
            "on_robot": stats(
                [_ms(r["sent"] - r["stamp"]) for r in members if trusted and r["stamp"]]
            ),
            "network": stats(
                [_ms(_arrival(r) - r["sent"] + offset_ns) for r in members if r["sent"]]
            ),
            "rate": rate([r["sent"] for r in members]),
        }
    return result


def _relay(record, offset_ns, started_ns):
    """A robot topic relayed by RobotNode onto sensors/*, or one only on the robot."""
    source = _rows(record.get("source_samples"))
    stack = _rows(record.get("samples"))
    trusted, stamp_ms = (
        _stamp_check(source) if source else _stamp_check(stack, offset_ns)
    )

    network = [_ms(_arrival(r) - r["sent"] + offset_ns) for r in source if r["sent"]]
    on_robot = [_ms(r["sent"] - r["stamp"]) for r in source if trusted and r["stamp"]]

    # Match each relayed message to the robot's original by its header. The
    # capture tool's own reception of the original stands in for RobotNode's:
    # both processes sit on this machine and get the same datagrams.
    originals = defaultdict(list)
    for row in source:
        if row["stamp"] is not None:
            originals[(row["stamp"], row["frame"])].append(row)
    relay, delivery, end_to_end, series = [], [], [], []
    matched = 0
    for row in stack:
        delivery.append(_ms(_arrival(row) - row["sent"]) if row["sent"] else None)
        candidates = originals.get((row["stamp"], row["frame"]))
        original = candidates.pop(0) if candidates else None
        if original is not None:
            matched += 1
            relay.append(_ms(row["sent"] - _arrival(original)))
        if trusted and row["stamp"]:
            start = row["stamp"]
        elif original is not None and original["sent"]:
            start = original["sent"]
        else:
            continue
        age = _ms(_arrival(row) - start + offset_ns)
        end_to_end.append(age)
        series.append(((_arrival(row) - started_ns) / 1e9, age))

    if not stack:                       # robot-only topic: it ends on arrival here
        for row in source:
            start = row["stamp"] if trusted and row["stamp"] else row["sent"]
            if start:
                age = _ms(_arrival(row) - start + offset_ns)
                end_to_end.append(age)
                series.append(((_arrival(row) - started_ns) / 1e9, age))

    stages = [
        _stage("sensor stamp to robot publish", "on robot", on_robot),
        _stage("robot to workstation (RTPS)", "network", network),
    ]
    if stack:
        stages += [
            _stage("RobotNode relay", "RobotNode", relay),
            _stage("sensors/* to subscriber", "delivery", delivery),
        ]
    return {
        "stages": stages,
        "end_to_end": stats(end_to_end),
        "series": _thin(series, SERIES_POINTS),
        "origin": "sensor stamp" if trusted else "robot publish",
        "stamp_ms": stamp_ms,
        "stamp_trusted": trusted,
        "publish_rate": rate([r["sent"] for r in source]),
        "relay_rate": rate([r["sent"] for r in stack]),
        "counts": {"robot": len(source), "relayed": matched, "stack": len(stack)},
        "per_frame": _per_frame(source, offset_ns, trusted),
        "tool_lag": stats(
            [_ms(r["callback"] - r["received"]) for r in source + stack if r["received"]]
        ),
        "scatter": [(r["bytes"], v) for r, v in zip(
            [r for r in source if r["sent"]], network)],
    }


def _zeromq(record, telemetry, offset_ns, started_ns):
    """observations / robot_state: the sender's timestamp rides inside the payload."""
    rows = _rows(record.get("samples"))
    network, decode, delivery, end_to_end, series = [], [], [], [], []
    for row in rows:
        delivery.append(_ms(_arrival(row) - row["sent"]) if row["sent"] else None)
        sender = telemetry.get(row["stamp"])
        if row["sent"] and row["stamp"]:
            decode.append(_ms(row["sent"] - row["stamp"]))
        if sender is None:
            continue
        network.append(_ms(row["stamp"] - sender + offset_ns))
        age = _ms(_arrival(row) - sender + offset_ns)
        end_to_end.append(age)
        series.append(((_arrival(row) - started_ns) / 1e9, age))
    return {
        "stages": [
            _stage("sensor stamp to robot publish", "on robot", []),
            _stage("robot to RobotNode (ZeroMQ)", "network", network),
            _stage("decode and publish", "RobotNode", decode),
            _stage("topic to subscriber", "delivery", delivery),
        ],
        "end_to_end": stats(end_to_end),
        "series": _thin(series, SERIES_POINTS),
        "origin": "sender timestamp",
        "stamp_ms": None,
        "stamp_trusted": False,
        "publish_rate": None,
        "relay_rate": rate([r["sent"] for r in rows]),
        "counts": {"robot": None, "relayed": len(end_to_end), "stack": len(rows)},
        "per_frame": None,
        "tool_lag": stats(
            [_ms(r["callback"] - r["received"]) for r in rows if r["received"]]
        ),
        "scatter": [],
    }


def _sender_frames(telemetry_rows):
    """How many frames the gripper sender produced vs. how many reached ROS."""
    rows = sorted(
        (r for r in telemetry_rows if r[2] is not None), key=lambda r: r[1]
    )
    if len(rows) < 2:
        return None
    numbers = [r[2] for r in rows]
    skipped = sum(max(0, b - a - 1) for a, b in zip(numbers, numbers[1:]))
    produced = numbers[-1] - numbers[0] + 1
    span_s = (rows[-1][1] - rows[0][1]) / 1e9
    return {
        "produced": produced,
        "received": len(rows),
        "skipped": skipped,
        "sender_hz": (produced - 1) / span_s if span_s > 0 else None,
        "sender_rate": rate([r[1] for r in rows]),
    }


def _fit(points):
    """Least-squares line through per-topic medians: ms = base + bytes / throughput."""
    if len(points) < 2 or len({x for x, _ in points}) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    if slope <= 0:
        return None
    return {
        "base_ms": mean_y - slope * mean_x,
        "ms_per_mb": slope * 1e6,
        "throughput_mb_s": 1e-3 / slope,
    }


def analyse(capture: dict) -> dict:
    """Everything the report draws, keyed by the capture's topic keys."""
    clock = capture.get("clock") or {}
    offset_ns = clock.get("offset_ns") or 0
    started_ns = capture.get("started_ns") or 0

    telemetry = {}
    for record in capture["topics"]:
        for stamp, sender_ns, _ in record.get("telemetry") or []:
            if stamp is not None and sender_ns is not None:
                telemetry[stamp] = sender_ns

    topics, scatter, medians = {}, [], []
    for record in capture["topics"]:
        if record.get("path") == "zeromq":
            result = _zeromq(record, telemetry, offset_ns, started_ns)
            if record.get("telemetry"):
                result["sender"] = _sender_frames(record["telemetry"])
        else:
            result = _relay(record, offset_ns, started_ns)
        points = result.pop("scatter")
        if points:
            scatter += points
            medians.append((
                percentile([b for b, _ in points], 0.5),
                percentile([v for _, v in points], 0.5),
            ))
        topics[record["key"]] = result

    crossings = [v for _, v in scatter]
    zmq = [
        s["stats"]["min"] for t in topics.values() for s in t["stages"]
        if s["role"] == "network" and s["stats"] and "ZeroMQ" in s["name"]
    ]
    lowest = min(crossings + zmq) if crossings or zmq else None
    uncertainty_ms = (clock.get("uncertainty_ns") or 0) / 1e6
    return {
        "topics": topics,
        "clock": {
            "measured": "offset_ns" in clock,
            "offset_ms": offset_ns / 1e6,
            "uncertainty_ms": uncertainty_ms,
            "drift_ms": (clock.get("drift_ns") or 0) / 1e6,
            "host": clock.get("host"),
            "error": clock.get("error"),
            # Fastest RTPS crossing: well above a millisecond or so on a local
            # link hints the robot's clock is behind ours.
            "floor_ms": min(crossings) if crossings else None,
            # Anything below zero arrived before it was sent: the clocks
            # disagree by at least that much.
            "lowest_ms": lowest,
            "impossible": lowest is not None and lowest < -(uncertainty_ms + 0.5),
        },
        "scatter": _thin(scatter, SCATTER_POINTS),
        "fit": _fit(medians),
    }
