"""Phrase -> pixel -> metric 3D point in the map frame.

This module is the reason the planner can say "the glass" and a skill can
receive a coordinate. Three stages:

  1. ask the grounder VLM to point at the phrase in the RGB image
  2. read the depth at that pixel (median of a small patch, because a single
     depth pixel on a transparent glass rim is a lottery ticket)
  3. deproject with the camera intrinsics, then transform camera -> map

Stage 2 is where transparent objects hurt. A drinking glass is the classic
failure case for active-stereo depth: the D435-class sensor on the Stretch
head will often return zero or garbage on the glass body. `estimate_depth`
handles that explicitly by falling back to the supporting surface around the
object and adding a height offset, rather than silently returning a point
inside the coffee table. Read `DEPTH_FALLBACK_NOTE` before you trust a number.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .backends import ChatMessage, VLMBackend
from .schema import Detection, Point3D

DEPTH_FALLBACK_NOTE = (
    "depth was invalid at the detected pixel (common on glass and other "
    "transparent or specular objects); the point was estimated from the "
    "surrounding surface plus an assumed object height"
)

# --------------------------------------------------------------------------
# Prompts. Kept separate per model family because the pointing output formats
# genuinely differ -- this is the one place where swapping grounder models is
# not free.
# --------------------------------------------------------------------------
POINT_PROMPT = """\
Point at: {query}

Reply with ONLY a JSON object:
{{"points": [{{"point": [x, y], "label": "<what you see>", "confidence": 0.0-1.0}}],
  "found": true/false,
  "note": "<one short sentence; say so if it is occluded, ambiguous or absent>"}}

Coordinates are pixel positions in the image, x to the right, y down.
If the object is not visible, return {{"points": [], "found": false, "note": "..."}}.
Do not guess a location for something you cannot see -- a false point is far
worse than an honest "not found", because a robot will drive its hand there."""

BBOX_PROMPT = """\
Find: {query}

Reply with ONLY a JSON object:
{{"boxes": [{{"bbox": [x0, y0, x1, y1], "label": "...", "confidence": 0.0-1.0}}],
  "found": true/false, "note": "..."}}
Coordinates are pixels in the image. If absent, return an empty list."""


@dataclass
class CameraInfo:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame_id: str = "camera_color_optical_frame"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraInfo":
        if "K" in d and len(d["K"]) == 9:
            K = d["K"]
            return cls(fx=K[0], fy=K[4], cx=K[2], cy=K[5],
                       width=d.get("width", 0), height=d.get("height", 0),
                       frame_id=d.get("frame_id", "camera_color_optical_frame"))
        return cls(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
                   width=d.get("width", 0), height=d.get("height", 0),
                   frame_id=d.get("frame_id", "camera_color_optical_frame"))


# --------------------------------------------------------------------------
# Output parsing. Models normalise coordinates differently; normalise back.
# --------------------------------------------------------------------------
def parse_points(raw: str, width: int, height: int) -> list[dict[str, Any]]:
    """Parse a pointing response into absolute pixel coordinates.

    Handles the three conventions in the wild:
      absolute pixels            (RoboBrain, most fine-tunes)
      normalised 0-1             (some LLaVA derivatives)
      normalised 0-1000          (Qwen-VL lineage, and models derived from it)

    Disambiguation is by magnitude, which is heuristic but safe: a value in
    (0, 1] cannot be a useful pixel, and a value > max(width, height) cannot
    be an absolute one.
    """
    from .backends.base import extract_json
    try:
        obj = extract_json(raw)
    except ValueError:
        # last resort: bare coordinate pairs in prose
        nums = re.findall(r"[-+]?\d*\.?\d+", raw)
        if len(nums) >= 2:
            obj = {"points": [{"point": [float(nums[0]), float(nums[1])],
                               "label": "", "confidence": 0.3}], "found": True}
        else:
            return []
    items = obj.get("points") or obj.get("boxes") or []
    out: list[dict[str, Any]] = []
    for it in items:
        if "point" in it:
            xy = it["point"]
        elif "bbox" in it and len(it["bbox"]) == 4:
            x0, y0, x1, y1 = it["bbox"]
            xy = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
            it = {**it, "bbox_raw": it["bbox"]}
        else:
            continue
        if len(xy) != 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        scale = _infer_scale(x, y, width, height)
        px, py = int(round(x * scale[0])), int(round(y * scale[1]))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        rec = {"pixel": (px, py), "label": it.get("label", ""),
               "confidence": float(it.get("confidence", 0.5)),
               "note": obj.get("note", "")}
        if "bbox_raw" in it:
            b = it["bbox_raw"]
            rec["bbox"] = (int(b[0] * scale[0]), int(b[1] * scale[1]),
                           int(b[2] * scale[0]), int(b[3] * scale[1]))
        out.append(rec)
    return out


def _infer_scale(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    m = max(abs(x), abs(y))
    if m <= 1.0:
        return (width, height)
    if m <= 1000.0 and m > max(width, height):
        return (width / 1000.0, height / 1000.0)
    return (1.0, 1.0)


# --------------------------------------------------------------------------
# Depth -> 3D
# --------------------------------------------------------------------------
def estimate_depth(
    depth_m: np.ndarray,
    pixel: tuple[int, int],
    patch: int = 7,
    valid_range: tuple[float, float] = (0.15, 6.0),
) -> tuple[float | None, bool]:
    """Robust depth at a pixel. Returns (metres, used_fallback)."""
    h, w = depth_m.shape[:2]
    u, v = pixel
    half = patch // 2
    sub = depth_m[max(0, v - half): min(h, v + half + 1),
                  max(0, u - half): min(w, u + half + 1)].astype(np.float32)
    ok = sub[(sub > valid_range[0]) & (sub < valid_range[1]) & np.isfinite(sub)]
    if ok.size >= max(3, patch):
        return float(np.median(ok)), False
    # widen: the object itself is see-through, so read the surface around it
    for r in (25, 60, 120):
        sub = depth_m[max(0, v - r): min(h, v + r + 1),
                      max(0, u - r): min(w, u + r + 1)].astype(np.float32)
        ok = sub[(sub > valid_range[0]) & (sub < valid_range[1]) & np.isfinite(sub)]
        if ok.size >= 30:
            # 25th percentile, not the median: a window wide enough to escape a
            # transparent object also catches the far wall, and the supporting
            # surface -- the thing we actually want -- is the NEARER population.
            # Taking the middle of a bimodal window lands the point in mid-air.
            return float(np.percentile(ok, 25)), True
    return None, True


def deproject(pixel: tuple[int, int], depth: float, cam: CameraInfo) -> tuple[float, float, float]:
    """Pixel + depth -> point in the camera optical frame (x right, y down, z forward)."""
    u, v = pixel
    z = depth
    x = (u - cam.cx) * z / cam.fx
    y = (v - cam.cy) * z / cam.fy
    return (x, y, z)


def transform_point(p: Sequence[float], T: np.ndarray) -> tuple[float, float, float]:
    """Apply a 4x4 homogeneous transform."""
    v = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    out = T @ v
    return (float(out[0]), float(out[1]), float(out[2]))


def pose2d_to_matrix(x: float, y: float, theta: float, z: float = 0.0) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0, x],
                     [s,  c, 0.0, y],
                     [0.0, 0.0, 1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# The grounder
# --------------------------------------------------------------------------
class Grounder:
    """Wraps a VLM in the pointing role and produces `Detection`s."""

    def __init__(
        self,
        backend: VLMBackend,
        mode: str = "point",                 # "point" | "bbox"
        assumed_object_height_m: float = 0.08,
        z_range: tuple[float, float] = (-0.05, 2.20),
        system_prompt: str | None = None,
    ):
        self.backend = backend
        self.mode = mode
        self.assumed_object_height_m = assumed_object_height_m
        self.z_range = z_range
        self.system_prompt = system_prompt or (
            "You are the visual grounding module of a household robot. You locate "
            "objects precisely in images and you never invent a location for "
            "something you cannot see."
        )

    def ground(
        self,
        query: str,
        rgb_path: str | Path,
        depth: np.ndarray | None,
        cam: CameraInfo,
        T_cam_to_map: np.ndarray | None,
        max_results: int = 3,
    ) -> list[Detection]:
        prompt = (POINT_PROMPT if self.mode == "point" else BBOX_PROMPT).format(query=query)
        raw = self.backend.chat(
            [ChatMessage("system", self.system_prompt),
             ChatMessage("user", prompt, images=[str(rgb_path)])],
            max_tokens=512, temperature=0.0,
        )
        hits = parse_points(raw, cam.width, cam.height)[:max_results]
        dets: list[Detection] = []
        for h in hits:
            d = Detection(query=query, label=h.get("label") or query,
                          confidence=h.get("confidence", 0.5),
                          pixel=h["pixel"], bbox=h.get("bbox"),
                          notes=h.get("note", ""))
            if depth is not None:
                z, fallback = estimate_depth(depth, h["pixel"])
                if z is not None:
                    d.depth_m = z
                    cam_pt = deproject(h["pixel"], z, cam)
                    if T_cam_to_map is not None:
                        x, y, zz = transform_point(cam_pt, T_cam_to_map)
                        if fallback:
                            zz += self.assumed_object_height_m
                            d.notes = (d.notes + " | " + DEPTH_FALLBACK_NOTE).strip(" |")
                            d.confidence *= 0.7
                        if not (self.z_range[0] <= zz <= self.z_range[1]):
                            # The classic transparent-object failure: depth fell
                            # through the glass onto the far wall, and the ray now
                            # terminates below the floor or up near the ceiling.
                            # Handing that to a skill sends the arm somewhere
                            # dangerous, so we refuse the point and say why.
                            d.notes = (d.notes + f" | REJECTED a 3D point at height "
                                       f"{zz:.2f} m, which is not a plausible place for a "
                                       "household object -- the depth ray passed through the "
                                       "object onto the background. Get closer, or view the "
                                       "object against its supporting surface.").strip(" |")
                            d.confidence *= 0.3
                        else:
                            d.point_3d = Point3D(x=x, y=y, z=zz, frame="map")
                else:
                    d.notes = (d.notes + " | no valid depth anywhere near the pixel; "
                               "point not computed").strip(" |")
                    d.confidence *= 0.4
            dets.append(d)
        return dets
