"""Oracle planner and grounder for offline testing.

These stand in for the two VLMs so the whole conductor can be exercised without
a model. They are not a simulation of a VLM's *intelligence* -- they are a
fixed, correct policy -- which is exactly what you want when the thing under
test is the conductor: if an episode fails here, the bug is in the conductor, not
in the model.

The oracle planner deliberately reacts to monitor feedback (it recovers from
an empty grasp rather than blundering on), so the recovery path is covered.
"""
from __future__ import annotations

import json
import math
from typing import Any

from .backends.base import ChatMessage, VLMBackend
from .robot.mock import MockRobot


class OracleGrounder(VLMBackend):
    """Answers pointing queries by projecting the mock world into the image."""

    def __init__(self, robot: MockRobot):
        self.robot = robot
        self.name = "oracle-grounder"
        self.supports_guided_json = False

    def chat(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None) -> str:
        text = messages[-1].text
        # verification questions come from the Monitor, not the Grounder
        if "is that task fully and visibly complete" in text or "QUESTION:" in text:
            return self._verify(text)
        query = ""
        for line in text.splitlines():
            if line.startswith("Point at:") or line.startswith("Find:"):
                query = line.split(":", 1)[1].strip()
        best, best_s = None, 0.0
        qt = set(query.lower().replace(",", " ").split())
        for o in self.robot.visible():
            ot = set(o.name.lower().split())
            s = len(ot & qt) / max(len(ot), 1)
            if s > best_s:
                best, best_s = o, s
        if best is None or best_s == 0:
            return json.dumps({"points": [], "found": False,
                               "note": "not visible from this viewpoint"})
        uv = self.robot.project(best)
        if uv is None:
            return json.dumps({"points": [], "found": False, "note": "outside the frame"})
        return json.dumps({"points": [{"point": [uv[0], uv[1]], "label": best.name,
                                       "confidence": 0.88}],
                           "found": True, "note": ""})

    def _verify(self, text: str) -> str:
        """Ground truth: is the glass on the kitchen table and the gripper empty?"""
        glass = next((o for o in self.robot.world.objects if "glass" in o.name), None)
        done = bool(glass and not glass.held and "kitchen table" in (glass.on or ""))
        return json.dumps({
            "answer": done, "confidence": 0.9 if done else 0.85,
            "evidence": (f"the glass is on the {glass.on}" if glass else "no glass in the scene"),
        })


class OraclePlanner(VLMBackend):
    """A correct, hand-written policy for the glass task.

    It reads two things out of the prompt: whether the monitor complained, and
    whether the gripper is holding something. That is enough to drive the whole
    task and to recover from an injected empty grasp.
    """

    def __init__(self, robot: MockRobot):
        self.robot = robot
        self.name = "oracle-planner"
        self.supports_guided_json = False
        self.phase = 0
        self.pick_attempts = 0

    def chat(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None) -> str:
        text = messages[-1].text
        holding = "holding        : nothing" not in text
        complained = "MONITOR:" in text or "empty_grasp" in text
        rejected = "REJECTED" in text
        det = self._parse_detection(text)

        def out(action, args, sub="", rationale="", expect=""):
            return json.dumps({"action": action, "args": args, "subgoal": sub,
                               "rationale": rationale or "oracle policy",
                               "expect": expect})

        if rejected and "not stowed" in text.lower():
            return out("stow", {"carry": holding}, "recover",
                       "the arm must be stowed before navigating to a waypoint")

        if self._placed() and not holding:
            return out("done", {"status": "success",
                                "rationale": "observed the glass on the kitchen table"},
                       "verify")

        if complained and not holding and self.phase >= 4:
            # Empty grasp: the supervisor already stopped the pick mid-lift, so
            # the arm is over the table with an empty closed gripper. Clear it,
            # back off, re-ground, retry with a more specific feature. This is
            # the recovery path the whole design exists for.
            self.pick_attempts += 1
            step = self.pick_attempts % 4
            if step == 1:
                return out("set_gripper", {"state": "open"}, "recover from empty grasp",
                           "the gripper closed on air; clear it first")
            if step == 2:
                return out("navigate", {"relative": {"dx": -0.15}},
                           "recover from empty grasp",
                           "back off so the object is fully in view again")
            if step == 3:
                return out("observe", {"query": "the drinking glass on the coffee table"},
                           "recover from empty grasp",
                           "re-ground the glass after the miss")
            return out("pick_up",
                       {"object": "the side wall of the clear drinking glass",
                        "target": det or {"x": 2.8, "y": 1.05, "z": 0.74},
                        "stop": {"kind": "object_in_gripper", "argument": "glass"}},
                       "pick up the glass",
                       "the gripper closes on the glass and carries load")

        seq = [
            ("stow", {}, "prepare to drive"),
            ("navigate", {"target": "coffee_table"}, "reach the living room"),
            ("look_at", {"target": "table_surface", "tilt_rad": -0.55}, "find the glass"),
            ("observe", {"query": "the drinking glass on the coffee table"}, "find the glass"),
        ]
        if self.phase < len(seq):
            a, args, sub = seq[self.phase]
            self.phase += 1
            return out(a, args, sub)

        if not holding:
            self.phase = max(self.phase, 4)
            return out("pick_up",
                       {"object": "the clear drinking glass, by its side wall",
                        "target": det or {"x": 2.8, "y": 1.05, "z": 0.74},
                        "stop": {"kind": "object_in_gripper", "argument": "glass"}},
                       "pick up the glass",
                       "the gripper closes on the glass and carries load")

        # holding the glass
        if "arm stowed     : True" not in text:
            return out("stow", {"carry": True}, "carry to the kitchen",
                       "stow while holding so the base can drive")
        if not self._near_kitchen(text):
            return out("navigate", {"target": "kitchen_table"}, "carry to the kitchen")
        return out("put_down",
                   {"surface": "the empty area on the kitchen table",
                    "release": True,
                    "stop": {"kind": "gripper_empty"}},
                   "place the glass",
                   "the gripper opens and the glass rests on the table")

    # -- helpers ---------------------------------------------------------
    def _near_kitchen(self, text: str) -> bool:
        b = self.robot.world.base
        return math.hypot(b.x - 5.6, b.y - 2.6) < 1.2

    def _placed(self) -> bool:
        g = next((o for o in self.robot.world.objects if "glass" in o.name), None)
        return bool(g and not g.held and "kitchen table" in (g.on or ""))

    @staticmethod
    def _parse_detection(text: str) -> dict[str, float] | None:
        import re
        m = re.search(r"map point \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", text)
        if not m:
            return None
        return {"x": float(m.group(1)), "y": float(m.group(2)), "z": float(m.group(3))}
