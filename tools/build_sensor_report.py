#!/usr/bin/env python3
"""Turn a sensor_capture_* folder into a single self-contained report.html.

Normally run for you by capture_sensors.py. Run it directly to re-render a
report without re-recording:

    python3 tools/build_sensor_report.py sensor_capture_20260919_143022

Pure Python, no ROS and no plotting library: the charts are inline SVG.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --series: #2a78d6;
  --good: #0ca30c;
  --critical: #d03b3b;
  --ring: rgba(11, 11, 11, 0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --series: #3987e5;
    --ring: rgba(255, 255, 255, 0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --series: #3987e5;
  --ring: rgba(255, 255, 255, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px 80px;
  background: var(--page);
  color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 40px 0 12px; }
h3 { font-size: 15px; margin: 0 0 8px; }
.sub { color: var(--ink-2); margin: 0 0 28px; }
.card {
  background: var(--surface);
  border: 1px solid var(--ring);
  border-radius: 10px;
  padding: 18px;
  margin-bottom: 16px;
}
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }
.stat { flex: 1 1 160px; }
.stat .value { font-size: 30px; font-weight: 600; letter-spacing: -0.01em; }
.stat .label { color: var(--muted); font-size: 13px; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { white-space: nowrap; }
th {
  text-align: left; color: var(--muted); font-weight: 600;
  border-bottom: 1px solid var(--axis); padding: 6px 8px;
}
td { padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12.5px; }
.state { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.ok .dot { background: var(--good); }
.bad .dot { background: var(--critical); }
.ok { color: var(--good); }
.bad { color: var(--critical); }
.row { display: grid; grid-template-columns: 250px 1fr; gap: 10px; align-items: center; }
.row + .row { margin-top: 3px; }
.row .name { font-size: 12.5px; color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gallery { display: flex; flex-wrap: wrap; gap: 10px; }
.depth { max-width: 320px; }
.depth img { width: 100%; border-radius: 6px; border: 1px solid var(--ring);
  display: block; image-rendering: auto; }
.ramp { height: 8px; border-radius: 4px; margin-top: 6px;
  background: linear-gradient(90deg, #0d366b, #cde2fb); }
.ramp-ends { display: flex; justify-content: space-between;
  color: var(--muted); font-size: 11px; margin-top: 2px; }
.frame { width: 200px; }
.frame img { width: 100%; border-radius: 6px; border: 1px solid var(--ring); display: block; }
.cap { color: var(--muted); font-size: 12px; margin-top: 4px; }
.plots { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 18px; font-size: 13.5px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); font-style: italic; }
@media (max-width: 640px) {
  .row { grid-template-columns: 1fr; }
  .row .name { margin-bottom: 2px; }
}
"""


def esc(value) -> str:
    return html.escape(str(value))


def short(topic: str) -> str:
    return topic.replace("/sensors/", "")


# ---------------------------------------------------------------- timeline
def timeline_row(times: list[float], seconds: float) -> str:
    """One strip of ticks: when messages arrived over the capture window."""
    if not times:
        return (
            '<svg viewBox="0 0 1000 14" preserveAspectRatio="none" height="14" '
            'style="width:100%"><rect x="0" y="6" width="1000" height="2" '
            'fill="var(--grid)"/></svg>'
        )
    marks = []
    for moment in times:
        x = max(0.0, min(1.0, moment / seconds)) * 996
        marks.append(f'<rect x="{x:.1f}" y="1" width="4" height="12" rx="1.5"/>')
    return (
        '<svg viewBox="0 0 1000 14" preserveAspectRatio="none" height="14" '
        'style="width:100%"><rect x="0" y="6" width="1000" height="2" '
        'fill="var(--grid)"/><g fill="var(--series)">'
        + "".join(marks)
        + "</g></svg>"
    )


def timeline_axis(seconds: float) -> str:
    ticks = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = fraction * 996
        anchor = "start" if fraction == 0 else ("end" if fraction == 1 else "middle")
        ticks.append(
            f'<text x="{x:.1f}" y="12" text-anchor="{anchor}" font-size="11" '
            f'fill="var(--muted)">{fraction * seconds:.0f}s</text>'
        )
    return (
        '<svg viewBox="0 0 1000 16" preserveAspectRatio="none" height="16" '
        'style="width:100%">' + "".join(ticks) + "</svg>"
    )


# ------------------------------------------------------------- polar plots
def _top_down(points_xy, limit, size=300, radius=130):
    """Shared top-down frame: robot at centre, +x up, +y left, range rings."""
    centre = size / 2
    scale = radius / limit if limit > 0 else 1.0
    rings = []
    for fraction in (1 / 3, 2 / 3, 1.0):
        rings.append(
            f'<circle cx="{centre}" cy="{centre}" r="{radius * fraction:.1f}" '
            f'fill="none" stroke="var(--grid)" stroke-width="1"/>'
        )
    rings.append(                                   # only the outer ring is labelled
        f'<text x="{centre + radius - 2}" y="{centre - 6}" text-anchor="end" '
        f'font-size="10" fill="var(--muted)">{limit:.1f} m</text>'
    )
    axes = (
        f'<line x1="{centre}" y1="{centre - radius}" x2="{centre}" '
        f'y2="{centre + radius}" stroke="var(--axis)" stroke-width="1"/>'
        f'<line x1="{centre - radius}" y1="{centre}" x2="{centre + radius}" '
        f'y2="{centre}" stroke="var(--axis)" stroke-width="1"/>'
        f'<text x="{centre}" y="{centre - radius - 5}" text-anchor="middle" '
        f'font-size="11" fill="var(--muted)">front</text>'
    )
    marks = []
    for forward, left, dot in points_xy:
        x = centre - left * scale
        y = centre - forward * scale
        if 0 <= x <= size and 0 <= y <= size:
            marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot}"/>')
    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'role="img">{"".join(rings)}{axes}'
        f'<g fill="var(--series)">{"".join(marks)}</g>'
        f'<circle cx="{centre}" cy="{centre}" r="3" fill="var(--ink-2)"/></svg>'
    )


def scan_plot(detail: dict) -> str:
    returns = [r for r in detail["plot"] if r is not None]
    if not returns:
        return '<p class="empty">No returns in this scan.</p>'
    limit = max(1.0, round(sorted(returns)[int(len(returns) * 0.95)] + 0.4, 1))
    points = []
    angle = detail["angle_min"]
    for value in detail["plot"]:
        if value is not None and value <= limit:
            points.append((value * math.cos(angle), value * math.sin(angle), 1.6))
        angle += detail["angle_step"]
    return _top_down(points, limit)


def cloud_plot(detail: dict) -> str:
    points = detail.get("points") or []
    if not points:
        return '<p class="empty">No points in this cloud.</p>'
    limit = max(0.3, round(max(max(abs(p[0]), abs(p[1])) for p in points) * 1.1, 2))
    return _top_down([(p[0], p[1], 3) for p in points], limit, size=260, radius=110)


# ----------------------------------------------------------- detail panels
def kv(pairs) -> str:
    items = "".join(
        f"<dt>{esc(key)}</dt><dd>{esc(value)}</dd>" for key, value in pairs
    )
    return f'<dl class="kv">{items}</dl>'


def gallery(record: dict) -> str:
    """The saved frames for one camera, in capture order."""
    frames = record.get("frames") or []
    if not frames:
        return '<p class="empty">Frames arrived but none were saved.</p>'
    cards = "".join(
        f'<figure class="frame"><img src="{esc(f["file"])}" alt="frame at '
        f'{f["at"]:.1f} s" loading="lazy">'
        f'<figcaption class="cap">t = {f["at"]:.1f} s</figcaption></figure>'
        for f in frames
    )
    return f'<div class="gallery" style="margin-top:12px">{cards}</div>'


def depth_figure(detail: dict) -> str:
    """The depth channel, with the ramp spelled out so it reads unaided."""
    return (
        '<figure class="depth" style="margin:14px 0 0">'
        f'<img src="{esc(detail["depth_image"])}" alt="depth channel" loading="lazy">'
        '<div class="ramp"></div>'
        '<div class="ramp-ends"><span>near</span><span>far</span></div>'
        '<figcaption class="cap">depth &middot; gray means no reading</figcaption>'
        "</figure>"
    )


def detail_panel(record: dict) -> str:
    detail = record.get("detail") or {}
    kind = detail.get("kind")

    if kind == "image":
        return (
            kv([("format", detail.get("format")), ("frame_id", detail.get("frame")),
                ("bytes/frame", f'{detail.get("bytes", 0):,}')])
            + gallery(record)
        )

    if kind == "observation":
        return (
            kv([
                ("camera_id", detail.get("camera_id")),
                ("frame_id", detail.get("frame")),
                ("size", f'{detail["width"]} x {detail["height"]}'),
                ("encodings", f'{detail["rgb_encoding"]} / {detail["depth_encoding"]}'),
                ("fx, fy", f'{detail["fx"]}, {detail["fy"]}'),
                ("cx, cy", f'{detail["cx"]}, {detail["cy"]}'),
            ])
            + gallery(record)
            + depth_figure(detail)
        )

    if kind == "state_json":
        rows = "".join(
            f"<tr><td><code>{esc(name)}</code></td><td>{esc(value)}</td></tr>"
            for name, value in detail.get("fields", [])
        )
        if not rows:
            return '<p class="empty">Telemetry carried no state fields.</p>'
        return (
            '<div class="scroll"><table><thead><tr><th>field</th><th>value</th>'
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    if kind == "scan":
        return (
            '<div class="plots">' + scan_plot(detail)
            + kv([
                ("frame_id", detail.get("frame")),
                ("beams", f'{detail["beams"]:,}'),
                ("with a return", f'{detail["returns"]:,} ({detail["coverage_pct"]}%)'),
                ("closest", f'{detail["closest_m"]} m' if detail["closest_m"] else "-"),
                ("range_max", f'{detail["range_max"]} m'),
            ])
            + "</div>"
        )

    if kind == "costmap":
        image = (
            f'<img src="{esc(detail["image"])}" alt="local costmap" '
            f'style="image-rendering:pixelated;border-radius:6px;'
            f'border:1px solid var(--ring)">'
        )
        return (
            '<div class="plots">' + image
            + kv([
                ("frame_id", detail.get("frame")),
                ("grid", f'{detail["width"]} x {detail["height"]} cells'),
                ("resolution", f'{detail["resolution_m"]} m/cell'),
                ("extent", f'{detail["extent_m"]} m across'),
                ("occupied", f'{detail["occupied_cells"]:,} cells'),
                ("unknown", f'{detail["unknown_cells"]:,} cells'),
            ])
            + "</div>"
        )

    if kind == "points":
        return (
            '<div class="plots">' + cloud_plot(detail)
            + kv([("frame_id", detail.get("frame")),
                  ("points", f'{detail.get("count", 0):,}')])
            + "</div>"
        )

    if kind == "joints":
        rows = "".join(
            f'<tr><td><code>{esc(j["name"])}</code></td>'
            f'<td class="num">{j["position"]:.4f}</td></tr>'
            for j in detail["joints"]
        )
        return (
            "<table><thead><tr><th>joint</th><th style=\"text-align:right\">"
            f"position</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if kind == "pose":
        return kv([("frame_id", detail.get("frame")), ("x", f'{detail["x"]} m'),
                   ("y", f'{detail["y"]} m'), ("yaw", f'{detail["yaw_deg"]} deg')])

    if kind == "battery":
        return kv([("voltage", f'{detail["voltage"]} V'),
                   ("current", f'{detail["current"]} A'),
                   ("charge", f'{detail["percentage"] * 100:.0f}%')])

    if kind == "camera_info":
        return kv([("frame_id", detail.get("frame")),
                   ("size", f'{detail["width"]} x {detail["height"]}'),
                   ("fx, fy", f'{detail["fx"]}, {detail["fy"]}'),
                   ("cx, cy", f'{detail["cx"]}, {detail["cy"]}')])

    if kind == "value":
        return kv([("latest value", detail.get("value"))])

    return '<p class="empty">Nothing decoded for this message type.</p>'


# ------------------------------------------------------------------ report
def build_report(directory: Path) -> Path:
    directory = Path(directory)
    data = json.loads((directory / "capture.json").read_text())
    seconds = data["seconds"]
    topics = data["topics"]
    live = [t for t in topics if t["count"] > 0]
    silent = [t for t in topics if t["count"] == 0]
    total_rate = sum(t["bytes_per_s"] for t in topics)

    strips = "".join(
        f'<div class="row"><div class="name" title="{esc(t["topic"])}">'
        f'{esc(short(t["topic"]))}</div><div>{timeline_row(t["times"], seconds)}</div></div>'
        for t in topics
    )

    rows = ""
    for topic in topics:
        ok = topic["count"] > 0
        state = (
            '<span class="state ok"><span class="dot"></span>receiving</span>'
            if ok else
            '<span class="state bad"><span class="dot"></span>silent</span>'
        )
        rows += (
            f'<tr><td><code>{esc(short(topic["topic"]))}</code></td>'
            f'<td>{state}</td>'
            f'<td class="num">{topic["count"]}</td>'
            f'<td class="num">{topic["hz"]:.2f}</td>'
            f'<td class="num">{topic["avg_bytes"]:,}</td>'
            f'<td class="num">{topic["bytes_per_s"] / 1e6:.2f}</td>'
            f'<td><code>{esc(topic["source"])}</code></td></tr>'
        )

    panels = ""
    for topic in live:
        panels += (
            f'<div class="card"><h3><code>{esc(topic["topic"])}</code></h3>'
            f'{detail_panel(topic)}</div>'
        )

    silent_note = ""
    if silent:
        names = ", ".join(f'<code>{esc(t["topic"])}</code>' for t in silent)
        silent_note = (
            f'<div class="card"><h3 class="bad">Silent topics</h3><p>{names}</p>'
            "<p class=\"sub\" style=\"margin:0\">Either the robot-side driver is not "
            "running, the source topic name in <code>system.yaml</code> is wrong, or "
            "the message type in <code>SENSOR_RELAYS</code> does not match what the "
            "publisher actually sends. Check with <code>ros2 topic type &lt;source&gt;"
            "</code>.</p></div>"
        )

    return _write(
        directory,
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sensor relay capture</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>Sensor relay capture</h1>
<p class="sub">{esc(data["captured_at"])} &middot; {seconds:g} second window &middot;
every topic RobotNode publishes under <code>/sensors/</code></p>

<div class="card"><div class="stats">
  <div class="stat"><div class="value">{len(live)} / {len(topics)}</div>
    <div class="label">topics receiving data</div></div>
  <div class="stat"><div class="value">{len(silent)}</div>
    <div class="label">silent</div></div>
  <div class="stat"><div class="value">{total_rate / 1e6:.1f}</div>
    <div class="label">MB/s across all relays</div></div>
</div></div>

{silent_note}

<h2>When messages arrived</h2>
<div class="card">{strips}<div class="row"><div class="name"></div>
<div>{timeline_axis(seconds)}</div></div></div>

<h2>Every topic</h2>
<div class="card"><div class="scroll"><table><thead><tr>
<th>topic (under <code>/sensors/</code>)</th><th>state</th>
<th style="text-align:right">msgs</th>
<th style="text-align:right">Hz</th><th style="text-align:right">bytes/msg</th>
<th style="text-align:right">MB/s</th><th>relayed from</th>
</tr></thead><tbody>{rows}</tbody></table></div></div>

<h2>What the data looks like</h2>
{panels}
</div></body></html>
""",
    )


def _write(directory: Path, markup: str) -> Path:
    path = directory / "report.html"
    path.write_text(markup, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="A sensor_capture_* folder")
    print(build_report(Path(parser.parse_args().directory)))


if __name__ == "__main__":
    main()
