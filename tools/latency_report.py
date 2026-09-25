"""The latency half of report.html: sensor_latency.analyse() drawn as HTML.

build_sensor_report.py calls latency_sections() and adds LATENCY_CSS to its
page. Charts are plain HTML bars plus SVG scatter plots whose labels stay in
HTML, so nothing stretches when the page is resized.
"""
from __future__ import annotations

import html
import math

from sensor_latency import ROLES

# Stage colours: the first four categorical slots, validated as a set for
# colour-blind separation in both themes.
LATENCY_CSS = """
:root {
  --role-1: #2a78d6;
  --role-2: #eb6834;
  --role-3: #1baf7a;
  --role-4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --role-1: #3987e5;
    --role-2: #d95926;
    --role-3: #199e70;
    --role-4: #c98500;
  }
}
:root[data-theme="dark"] {
  --role-1: #3987e5;
  --role-2: #d95926;
  --role-3: #199e70;
  --role-4: #c98500;
}
h4 { font-size: 13.5px; margin: 18px 0 6px; font-weight: 600; }
.note { color: var(--ink-2); font-size: 13.5px; margin: 0 0 10px; }
.warn { border-color: var(--critical); }
.row3 { display: grid; grid-template-columns: 210px 1fr 150px; gap: 10px; align-items: center; }
.row3 + .row3 { margin-top: 6px; }
.row3 .name { font-size: 12.5px; color: var(--ink-2); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.row3 .val { font-size: 12.5px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.track { position: relative; height: 18px; }
.stack { display: flex; gap: 2px; height: 18px; }
.seg { height: 100%; min-width: 2px; }
.seg.end { border-radius: 0 4px 4px 0; }
.tick { position: absolute; top: -2px; bottom: -2px; width: 2px; margin-left: -1px;
  background: var(--ink-2); }
.axis { position: relative; height: 16px; font-size: 11px; color: var(--muted); }
.axis span { position: absolute; top: 2px; transform: translateX(-50%); white-space: nowrap; }
.axis span.first { transform: none; }
.axis span.last { transform: translateX(-100%); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px;
  color: var(--ink-2); margin: 0 0 12px; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }
.swatch.bar { width: 2px; height: 14px; border-radius: 0; background: var(--ink-2); }
.dist { position: relative; height: 18px; }
.dist .range { position: absolute; top: 6px; height: 6px; border-radius: 3px;
  background: var(--series); opacity: 0.35; }
.dist .p50 { position: absolute; top: 3px; width: 12px; height: 12px; margin-left: -6px;
  border-radius: 50%; background: var(--series); box-shadow: 0 0 0 2px var(--surface); }
.dist .p99, .dist .max { position: absolute; top: 2px; width: 2px; height: 14px;
  margin-left: -1px; background: var(--ink-2); }
.dist .max { background: var(--muted); }
.plot { display: grid; grid-template-columns: 52px 1fr; }
.plot .y { position: relative; }
.plot .y span { position: absolute; right: 8px; transform: translateY(-50%);
  font-size: 11px; color: var(--muted); white-space: nowrap; }
.plot .area { position: relative; border-bottom: 1px solid var(--axis); }
.plot svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.plot .grid line { vector-effect: non-scaling-stroke; stroke: var(--grid); stroke-width: 1px; }
.plot .dots path { vector-effect: non-scaling-stroke; stroke: var(--series);
  stroke-width: 5px; stroke-linecap: round; }
.plot .fit { vector-effect: non-scaling-stroke; stroke: var(--ink-2); stroke-width: 1.5px;
  fill: none; }
@media (max-width: 640px) {
  .row3 { grid-template-columns: 1fr; gap: 2px; }
}
"""


def esc(value) -> str:
    return html.escape(str(value))


def short(topic: str) -> str:
    return topic.replace("/sensors/", "")


# Which topics make up each physical sensor. Anything not listed (a robot-only
# topic, a new relay) still gets its own row under its topic name.
SENSORS = (
    ("Wrist camera (ZeroMQ)", ("observations", "robot_state")),
    ("Head camera, left (fisheye)", (
        "sensors/head_camera/left/image/compressed",
        "sensors/head_camera/left/camera_info")),
    ("Head camera, right (fisheye)", (
        "sensors/head_camera/right/image/compressed",
        "sensors/head_camera/right/camera_info")),
    ("Head camera, center", (
        "sensors/head_camera/center/image/compressed",
        "sensors/head_camera/center/image/zstd",
        "sensors/head_camera/center/image/compressedDepth",
        "sensors/head_camera/center/camera_info",
        "sensors/head_camera/center/camera_info_luxonis")),
    ("Lidars", ("sensors/scan",)),
    ("Line sensors", ("sensors/line_sensor/points",)),
    ("Joints", ("sensors/joint_states",)),
    ("Wheel odometry", ("sensors/odom",)),
    ("Battery", ("sensors/battery",)),
    ("Driver status", (
        "sensors/mode", "sensors/tool", "sensors/is_homed", "sensors/is_runstopped")),
    ("Local costmap", ("sensors/local_costmap",)),
)
ROLE_COLOR = dict(zip(ROLES, ("var(--role-1)", "var(--role-2)",
                              "var(--role-3)", "var(--role-4)")))
ROLE_LABEL = {
    "on robot": "on the robot",
    "network": "robot to workstation",
    "RobotNode": "inside RobotNode",
    "delivery": "to the subscriber",
}


def ms(value) -> str:
    if value is None:
        return "&ndash;"
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.1f}" if abs(value) < 100 else f"{value:.0f}"


def nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * power:
            return step * power
    return 10 * power


def axis(ticks) -> str:
    """HTML tick labels, so text never stretches with an SVG."""
    spans = []
    for index, (fraction, label) in enumerate(ticks):
        edge = "first" if fraction <= 0.001 else ("last" if fraction >= 0.999 else "")
        spans.append(
            f'<span class="{edge}" style="left:{fraction * 100:.2f}%">{label}</span>'
        )
    return f'<div class="axis">{"".join(spans)}</div>'


def linear_ticks(low, high, count=4, unit=" ms"):
    return [(i / count, f"{low + (high - low) * i / count:g}{unit}") for i in range(count + 1)]


def sensor_groups(capture, latency):
    """[(sensor name, [topic keys that carried data])], in SENSORS order."""
    live = {t["key"] for t in capture["topics"] if t["count"]}
    claimed, groups = set(), []
    for name, keys in SENSORS:
        members = [k for k in keys if k in live]
        claimed.update(keys)
        if members:
            groups.append((name, members))
    for topic in capture["topics"]:
        if topic["key"] in claimed or topic["key"] not in live:
            continue
        # A center topic RobotNode relays still joins the center camera's card.
        name = "Head camera, center" if "/center/" in topic["topic"] else topic["topic"]
        group = next((g for g in groups if g[0] == name), None)
        if group:
            group[1].append(topic["key"])
        else:
            groups.append((name, [topic["key"]]))
    return groups


def legend(extra: str = "") -> str:
    keys = "".join(
        f'<span class="key"><span class="swatch" style="background:{ROLE_COLOR[r]}">'
        f"</span>{ROLE_LABEL[r]}</span>"
        for r in ROLES
    )
    return f'<div class="legend">{keys}{extra}</div>'


def stage_bars(rows) -> str:
    """One stacked bar per sensor: the median of each stage, plus a p95 tick."""
    totals = []
    for _, result in rows:
        medians = sum(max(s["stats"]["p50"], 0) for s in result["stages"] if s["stats"])
        tail = (result["end_to_end"] or {}).get("p95") or 0
        totals.append(max(medians, tail))
    top = nice_ceiling(max(totals + [1.0]))
    lines = []
    for name, result in rows:
        stages = [s for s in result["stages"] if s["stats"]]
        segments = []
        for index, stage in enumerate(stages):
            value = stage["stats"]["p50"]
            width = max(value, 0) / top * 100
            end = " end" if index == len(stages) - 1 else ""
            segments.append(
                f'<div class="seg{end}" style="flex:0 0 {width:.2f}%;'
                f'background:{ROLE_COLOR[stage["role"]]}" title="{esc(stage["name"])}: '
                f'median {ms(value)} ms, p95 {ms(stage["stats"]["p95"])} ms"></div>'
            )
        age = result["end_to_end"]
        tick = (
            f'<div class="tick" style="left:{min(age["p95"] / top, 1) * 100:.2f}%" '
            f'title="end to end p95 {ms(age["p95"])} ms"></div>' if age else ""
        )
        value = (
            f'{ms(age["p50"])} ms &middot; p95 {ms(age["p95"])}' if age
            else "no end-to-end"
        )
        lines.append(
            f'<div class="row3"><div class="name" title="{esc(name)}">{esc(name)}</div>'
            f'<div class="track"><div class="stack">{"".join(segments)}</div>{tick}</div>'
            f'<div class="val">{value}</div></div>'
        )
    lines.append(
        f'<div class="row3"><div></div>{axis(linear_ticks(0, top))}<div></div></div>'
    )
    return "".join(lines)


def distribution_rows(rows) -> str:
    """End-to-end spread per topic on a log axis: p5-p95 bar, median dot, p99, max."""
    values = [r["max"] for _, r in rows]
    low = 0.1
    high = 10 ** math.ceil(math.log10(max(values + [1.0])))

    def at(value):
        value = min(max(value, low), high)
        return (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) * 100

    lines = []
    for name, s in rows:
        lines.append(
            f'<div class="row3"><div class="name" title="{esc(name)}">{esc(name)}</div>'
            f'<div class="dist" title="p50 {ms(s["p50"])} ms, p95 {ms(s["p95"])} ms, '
            f'p99 {ms(s["p99"])} ms, max {ms(s["max"])} ms">'
            f'<div class="range" style="left:{at(s["p5"]):.2f}%;'
            f'width:{max(at(s["p95"]) - at(s["p5"]), 0.5):.2f}%"></div>'
            f'<div class="max" style="left:{at(s["max"]):.2f}%"></div>'
            f'<div class="p99" style="left:{at(s["p99"]):.2f}%"></div>'
            f'<div class="p50" style="left:{at(s["p50"]):.2f}%"></div></div>'
            f'<div class="val">{ms(s["p50"])} / {ms(s["p99"])} ms</div></div>'
        )
    decades = round(math.log10(high / low))
    ticks = [(i / decades, f"{low * 10 ** i:g} ms") for i in range(decades + 1)]
    lines.append(f'<div class="row3"><div></div>{axis(ticks)}<div></div></div>')
    return "".join(lines)


def plot(body: str, y_ticks, x_ticks, height=160) -> str:
    """A chart box: SVG marks stretched to fit, labels in HTML so they stay crisp.

    body uses a 0..1000 square coordinate space, y pointing down.
    """
    grid = "".join(
        f'<line x1="0" x2="1000" y1="{1000 - f * 1000:.1f}" y2="{1000 - f * 1000:.1f}"/>'
        for f, _ in y_ticks
    )
    labels = "".join(
        f'<span style="top:{100 - f * 100:.2f}%">{label}</span>' for f, label in y_ticks
    )
    return (
        f'<div class="plot"><div class="y">{labels}</div>'
        f'<div class="area" style="height:{height}px"><svg viewBox="0 0 1000 1000" '
        f'preserveAspectRatio="none" role="img"><g class="grid">{grid}</g>{body}'
        f"</svg></div><div></div>{axis(x_ticks)}</div>"
    )


def age_over_time(series, seconds) -> str:
    """End-to-end age of every message against when it arrived."""
    if not series:
        return '<p class="empty">No end-to-end samples.</p>'
    values = [v for _, v in series]
    low = min(0.0, -nice_ceiling(-min(values))) if min(values) < 0 else 0.0
    high = nice_ceiling(max(values))
    span = high - low
    dots = "".join(
        f'<path d="M{min(max(t / seconds, 0), 1) * 1000:.1f} '
        f'{1000 - (v - low) / span * 1000:.1f}h0"><title>{t:.1f} s: {ms(v)} ms</title></path>'
        for t, v in series
    )
    return plot(
        f'<g class="dots">{dots}</g>',
        linear_ticks(low, high, 2),
        [(f, f"{f * seconds:.0f} s") for f in (0, 0.5, 1)],
        height=120,
    )


def size_vs_network(points, fit) -> str:
    """Robot-to-workstation time against message size, every message."""
    if not points:
        return '<p class="empty">No robot-side samples (captured with --stack-only).</p>'
    sizes = [max(b, 1) for b, _ in points]
    values = [v for _, v in points]
    x_low = 10 ** math.floor(math.log10(min(sizes)))
    x_high = 10 ** math.ceil(math.log10(max(sizes)))
    y_low = min(0.0, -nice_ceiling(-min(values))) if min(values) < 0 else 0.0
    y_high = nice_ceiling(max(values))

    def x(size):
        return (math.log10(size) - math.log10(x_low)) / (
            math.log10(x_high) - math.log10(x_low)) * 1000

    def y(value):
        return 1000 - (value - y_low) / (y_high - y_low) * 1000

    dots = "".join(
        f'<path d="M{x(max(b, 1)):.1f} {y(v):.1f}h0"><title>{b:,} B: {ms(v)} ms'
        "</title></path>"
        for b, v in points
    )
    line = ""
    if fit:
        steps = [x_low * (x_high / x_low) ** (i / 60) for i in range(61)]
        path = " ".join(
            f"{x(size):.1f},{y(min(fit['base_ms'] + size * fit['ms_per_mb'] / 1e6, y_high)):.1f}"
            for size in steps
        )
        line = f'<polyline class="fit" points="{path}"/>'
    names = {1: "1 B", 10: "10 B", 100: "100 B", 1000: "1 kB", 10**4: "10 kB",
             10**5: "100 kB", 10**6: "1 MB", 10**7: "10 MB"}
    decades = round(math.log10(x_high / x_low))
    x_ticks = [
        (i / decades, names.get(int(x_low * 10 ** i), f"{x_low * 10 ** i:g} B"))
        for i in range(decades + 1)
    ]
    return plot(f'<g class="dots">{dots}</g>{line}',
                linear_ticks(y_low, y_high), x_ticks, height=220)


def stage_table(result) -> str:
    rows = ""
    for stage in result["stages"]:
        s = stage["stats"]
        if not s:
            continue
        rows += (
            f'<tr><td><span class="key"><span class="swatch" style="background:'
            f'{ROLE_COLOR[stage["role"]]}"></span>{esc(stage["name"])}</span></td>'
            f'<td class="num">{s["n"]}</td><td class="num">{ms(s["p50"])}</td>'
            f'<td class="num">{ms(s["p95"])}</td><td class="num">{ms(s["p99"])}</td>'
            f'<td class="num">{ms(s["max"])}</td></tr>'
        )
    age = result["end_to_end"]
    if age:
        rows += (
            f'<tr><td><b>end to end</b> <span class="note">from '
            f'{esc(result["origin"])}</span></td><td class="num">{age["n"]}</td>'
            f'<td class="num"><b>{ms(age["p50"])}</b></td><td class="num">'
            f'<b>{ms(age["p95"])}</b></td><td class="num">{ms(age["p99"])}</td>'
            f'<td class="num">{ms(age["max"])}</td></tr>'
        )
    if not rows:
        return '<p class="empty">No timing samples.</p>'
    return (
        '<div class="scroll"><table><thead><tr><th>stage (ms)</th>'
        '<th style="text-align:right">msgs</th><th style="text-align:right">p50</th>'
        '<th style="text-align:right">p95</th><th style="text-align:right">p99</th>'
        f'<th style="text-align:right">max</th></tr></thead><tbody>{rows}'
        "</tbody></table></div>"
    )


def topic_notes(topic, result) -> str:
    notes = []
    counts = result["counts"]
    if topic["path"] == "relay" and counts["robot"]:
        share = 100.0 * counts["relayed"] / counts["robot"]
        notes.append(
            f'The robot published {counts["robot"]} messages that arrived here; '
            f'RobotNode relayed {counts["relayed"]} of them ({share:.0f}%).'
        )
    if topic["path"] in ("relay", "robot") and result["stamp_ms"] is not None \
            and not result["stamp_trusted"]:
        notes.append(
            f'header.stamp is not on the robot&rsquo;s clock (stamp to publish '
            f'median {ms(result["stamp_ms"])} ms), so the sensor stamps with its own '
            "clock. End to end is measured from the robot&rsquo;s publish instead."
        )
    sender = result.get("sender")
    if sender:
        notes.append(
            f'The gripper sender produced {sender["produced"]} frames'
            + (f' ({sender["sender_hz"]:.1f} Hz)' if sender["sender_hz"] else "")
            + f'; RobotNode published {sender["received"]} and skipped '
            f'{sender["skipped"]}. Its ZeroMQ socket keeps only the newest frame and '
            "is read on a timer, so skipped frames are expected when the sender is "
            "faster than the timer."
        )
    frames = result.get("per_frame")
    table = ""
    if frames:
        rows = ""
        for frame, f in frames.items():
            hz = f'{f["rate"]["hz"]:.1f}' if f["rate"] else "&ndash;"
            rows += (
                f'<tr><td><code>{esc(frame)}</code></td><td class="num">{f["count"]}</td>'
                f'<td class="num">{hz}</td>'
                f'<td class="num">{ms((f["on_robot"] or {}).get("p50"))}</td>'
                f'<td class="num">{ms((f["network"] or {}).get("p50"))}</td>'
                f'<td class="num">{ms((f["network"] or {}).get("p95"))}</td></tr>'
            )
        table = (
            '<h4>Split by frame_id (one row per physical sensor)</h4><div class="scroll">'
            "<table><thead><tr><th>frame_id</th><th style=\"text-align:right\">msgs</th>"
            "<th style=\"text-align:right\">Hz</th>"
            "<th style=\"text-align:right\">on robot p50</th>"
            "<th style=\"text-align:right\">network p50</th>"
            "<th style=\"text-align:right\">network p95</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    text = "".join(f'<p class="note">{n}</p>' for n in notes)
    return text + table


def sensor_card(name, members, by_key, latency, seconds) -> str:
    parts = [f"<h3>{esc(name)}</h3>"]
    for key in members:
        topic, result = by_key[key], latency["topics"][key]
        parts.append(
            f'<h4><code>{esc(topic["topic"])}</code> <span class="note">from '
            f'<code>{esc(topic["source"])}</code></span></h4>'
            + topic_notes(topic, result) + stage_table(result)
        )
    primary = latency["topics"][members[0]]
    if primary["series"]:
        parts.append(
            f'<h4>End-to-end age of each <code>{esc(by_key[members[0]]["topic"])}</code>'
            " message over the capture (ms)</h4>"
            + age_over_time(primary["series"], seconds)
        )
    return f'<div class="card">{"".join(parts)}</div>'


def clock_card(clock, capture) -> str:
    if clock["measured"]:
        drift = (
            f' It moved {ms(clock["drift_ms"])} ms between the checks before and '
            "after recording." if clock["drift_ms"] else ""
        )
        text = (
            f'The robot&rsquo;s clock is <b>{clock["offset_ms"]:+.2f} ms</b> from this '
            f'machine&rsquo;s (&plusmn;{ms(clock["uncertainty_ms"])} ms, measured over '
            f'SSH to <code>{esc(clock["host"])}</code>).{drift} Every robot-to-'
            "workstation number below is corrected for it."
        )
    elif capture.get("clock"):
        text = (
            f'Tried to measure the robot&rsquo;s clock over SSH and could not: '
            f'<code>{esc(str(clock["error"]).rstrip("."))}</code>. Robot-to-workstation numbers assume '
            "the two clocks agree."
        )
    else:
        text = (
            "Not measured. Robot-to-workstation numbers assume the two clocks agree; "
            "run with <code>--robot user@robot-ip</code> to measure and correct."
        )
    floor, check = clock["floor_ms"], ""
    if clock["impossible"]:
        check = (
            f'<p class="note"><b class="bad">Some messages arrived '
            f'{ms(-clock["lowest_ms"])} ms before they were sent, so the clocks '
            "disagree by at least that much and every network and end-to-end number "
            "is off by it.</b></p>"
        )
    elif floor is not None:
        check = f'<p class="note">Fastest robot-to-workstation message: {ms(floor)} ms.' + (
            " That is slow for a small message on a local link, which usually means "
            "the robot&rsquo;s clock is behind this machine&rsquo;s by roughly that much."
            if floor > 5 else
            " Nothing arrived before it was sent, so nothing contradicts the clock "
            "offset used."
        ) + "</p>"
    style = " warn" if clock["impossible"] or not clock["measured"] else ""
    return (
        f'<div class="card{style}"><h3>Robot and workstation clocks</h3>'
        f'<p class="note" style="margin-bottom:6px">{text}</p>{check}</div>'
    )


def health_card(capture, latency) -> str:
    """How much to trust the capture itself."""
    topics = capture["topics"]
    fallback = [
        t["topic"] for t in topics
        if any(not s[3] for s in t["samples"] + t["source_samples"])
    ]
    lags = [
        r["tool_lag"]["p99"] for r in latency["topics"].values() if r["tool_lag"]
    ]
    extra = sum(
        sum(s[5] for s in t["source_samples"]) for t in topics if t["path"] == "relay"
    ) / max(capture["seconds"], 1e-9)
    lines = [
        f'Middleware: <code>{esc(capture.get("rmw") or "unknown")}</code>, ROS '
        f'<code>{esc(capture.get("ros_distro") or "unknown")}</code>.',
    ]
    if fallback:
        lines.append(
            "This middleware gave no reception timestamp for "
            f'{len(fallback)} topic(s), so arrival there is when the capture&rsquo;s '
            "Python callback ran. That adds the tool&rsquo;s own queueing to those numbers."
        )
    else:
        lines.append(
            "Arrival times are the middleware&rsquo;s own reception timestamps, taken "
            "when the last byte arrived, so they do not include this tool&rsquo;s "
            "Python overhead."
        )
    if lags:
        lines.append(
            f"The capture&rsquo;s callbacks ran up to {ms(max(lags))} ms (worst topic "
            "p99) after reception. Its 50-message queues absorb that without dropping."
        )
    if capture.get("robot_topics", True):
        lines.append(
            f"To time the robot side, this capture subscribed to the robot&rsquo;s own "
            f"topics too, so the robot sent them twice: about {extra / 1e6:.1f} MB/s "
            "extra on the link. To check that did not slow things down, compare the "
            "end-to-end numbers with a <code>--stack-only</code> capture."
        )
    return (
        '<div class="card"><h3>How far to trust this capture</h3>'
        + "".join(f'<p class="note">{line}</p>' for line in lines) + "</div>"
    )


def latency_sections(capture, latency) -> str:
    seconds = capture["seconds"]
    by_key = {t["key"]: t for t in capture["topics"]}
    groups = sensor_groups(capture, latency)

    bars = [(name, latency["topics"][members[0]]) for name, members in groups]
    spread = [
        (by_key[k]["topic"].replace("/sensors/", ""), latency["topics"][k]["end_to_end"])
        for _, members in groups for k in members if latency["topics"][k]["end_to_end"]
    ]

    rows = ""
    for name, members in groups:
        for key in members:
            topic, result = by_key[key], latency["topics"][key]
            age = result["end_to_end"] or {}
            publish = result["publish_rate"] or result["relay_rate"] or {}
            counts = result["counts"]
            relayed = (
                f'{100.0 * counts["relayed"] / counts["robot"]:.0f}%'
                if topic["path"] == "relay" and counts["robot"] else "&ndash;"
            )
            hz = f'{publish["hz"]:.1f}' if publish else "&ndash;"
            rows += (
                f'<tr><td><code>{esc(short(topic["topic"]))}</code></td>'
                f'<td>{esc(result["origin"])}</td>'
                f'<td class="num">{ms(age.get("p50"))}</td><td class="num">{ms(age.get("p95"))}</td>'
                f'<td class="num">{ms(age.get("p99"))}</td><td class="num">{ms(age.get("max"))}</td>'
                f'<td class="num">{hz}</td>'
                f'<td class="num">{ms(publish.get("jitter_ms"))}</td>'
                f'<td class="num">{ms(publish.get("max_gap_ms"))}</td>'
                f'<td class="num">{publish.get("long_gaps", "&ndash;")}</td>'
                f'<td class="num">{relayed}</td></tr>'
            )

    fit = latency["fit"]
    fit_note = (
        f'<p class="note">Line: a straight fit through each topic&rsquo;s median, '
        f'{ms(fit["base_ms"])} ms plus {ms(fit["ms_per_mb"])} ms per MB, which is an '
        f'effective {fit["throughput_mb_s"]:.1f} MB/s for one message crossing the link. '
        "Big messages sit above small ones because they take longer to send; points "
        "far above the line waited behind other traffic.</p>" if fit else ""
    )
    cards = "".join(
        sensor_card(name, members, by_key, latency, seconds) for name, members in groups
    )
    return f"""
{clock_card(latency["clock"], capture)}

<h2>Where the time goes, per sensor</h2>
<div class="card">
<p class="note">Each bar is the median time a sensor&rsquo;s main topic spends in each
stage; the thin tick is its end-to-end 95th percentile. End to end runs from the
sensor&rsquo;s own timestamp (or the robot&rsquo;s publish, where the stamp cannot be
trusted) to the moment a subscriber on <code>sensors/*</code> has the whole message.</p>
{legend('<span class="key"><span class="swatch bar"></span>end to end p95</span>')}
{stage_bars(bars)}
</div>

<h2>How much it varies, per topic</h2>
<div class="card">
<div class="legend"><span class="key"><span class="swatch" style="background:var(--series);opacity:.35"></span>5th to 95th percentile</span>
<span class="key"><span class="swatch" style="background:var(--series);border-radius:50%"></span>median</span>
<span class="key"><span class="swatch bar"></span>99th percentile</span>
<span class="key"><span class="swatch bar" style="background:var(--muted)"></span>worst</span></div>
{distribution_rows(spread) if spread else '<p class="empty">No end-to-end samples.</p>'}
</div>

<h2>Latency and rate for every topic</h2>
<div class="card"><div class="scroll"><table><thead><tr>
<th>topic</th><th>measured from</th>
<th style="text-align:right">p50 ms</th><th style="text-align:right">p95</th>
<th style="text-align:right">p99</th><th style="text-align:right">max</th>
<th style="text-align:right">publish Hz</th><th style="text-align:right">jitter ms</th>
<th style="text-align:right">longest gap</th><th style="text-align:right">missed</th>
<th style="text-align:right">relayed</th>
</tr></thead><tbody>{rows}</tbody></table></div>
<p class="note" style="margin-top:10px">Publish Hz, jitter and gaps come from the
publisher&rsquo;s own send times. <i>missed</i> counts gaps longer than 1.8 periods,
each most likely a lost message. <i>relayed</i> is how many of the robot&rsquo;s
messages that reached this machine RobotNode passed on.</p></div>

<h2>Crossing the link: time against message size</h2>
<div class="card">
<p class="note">Every message the robot published, from its send to its arrival here.</p>
{size_vs_network(latency["scatter"], fit)}
{fit_note}
</div>

<h2>Each sensor, stage by stage</h2>
{cards}

{health_card(capture, latency)}
"""
