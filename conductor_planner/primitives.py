"""The fixed primitive vocabulary.

The planner cannot invent primitives at deployment time. It composes a closed
set, declared once here and rendered into the system prompt automatically, so a
prompt that describes a primitive you deleted is structurally impossible.

Eight primitives, in three groups:

  analytic  deterministic, ours, no learning involved
  skill     local, startable and stoppable; supervised while it runs
  control   episode termination

The skill group is `navigate`, `pick_up`, `put_down`. These names ARE the
contract: they are what the planner emits and what the skills team registers in
`conductor_planner/skills/`. Skills differ from analytic primitives in one way
that shapes everything else -- they take seconds, so they are objects that get
started, watched and stopped, not calls that block. See
`docs/SKILL_INTERFACE.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import Observation, PrimitiveName, RobotState

Group = str  # "analytic" | "skill" | "control"


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str                      # rendered into the prompt / JSON schema
    required: bool = True
    default: Any = None
    doc: str = ""
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PrimitiveSpec:
    name: PrimitiveName
    group: Group
    summary: str
    args: tuple[ArgSpec, ...]
    postcondition: str             # prose, checked by monitor.py
    preconditions: tuple[str, ...] = ()
    # (state, args) -> None if allowed, else a reason addressed to the planner.
    # Guards see the arguments because the same primitive can be safe or unsafe
    # depending on them -- crossing a room with the arm out is dangerous, nudging
    # 20 cm to re-stage a failed pick is not, and both are `navigate`.
    guard: Callable[[RobotState, dict[str, Any]], str | None] | None = None
    cost_hint: str = ""
    # stop kinds that make sense for this primitive; the first is suggested
    stop_kinds: tuple[str, ...] = ()
    examples: tuple[dict[str, Any], ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Guards -- cheap, local, deterministic checks run before we ever move.
# These exist because the most expensive failure on a real robot is the one
# you could have refused. They are NOT safety; the vendor's runstop is safety.
# --------------------------------------------------------------------------
def _guard_navigate(s: RobotState, args: dict[str, Any]) -> str | None:
    if s.runstop_engaged:
        return "runstop engaged; release it before commanding the base"
    if args.get("relative") and not (args.get("target") or args.get("pose")):
        # a short re-stage nudge is allowed with the arm out: it is slow, local,
        # and it is the move you need precisely when the arm is extended after a
        # failed pick. Requiring a stow here would force stow -> nudge -> unstow
        # and lose the pose you were trying to correct.
        return None
    if not s.is_stowed:
        return ("arm is not stowed; call stow() before navigating to a waypoint "
                "or pose (a small `relative` nudge is allowed with the arm out)")
    return None


def _guard_manipulate(s: RobotState, args: dict[str, Any]) -> str | None:
    if s.runstop_engaged:
        return "runstop engaged; release it before commanding the arm"
    return None


def _guard_pick_up(s: RobotState, args: dict[str, Any]) -> str | None:
    g = _guard_manipulate(s, args)
    if g:
        return g
    if s.holding:
        return (f"already holding '{s.holding}'; put it down or release it "
                "before picking up something else")
    return None


def _guard_put_down(s: RobotState, args: dict[str, Any]) -> str | None:
    g = _guard_manipulate(s, args)
    if g:
        return g
    if not s.holding:
        return "the gripper is empty; there is nothing to put down"
    return None


# --------------------------------------------------------------------------
# The library
# --------------------------------------------------------------------------
REGISTRY: dict[PrimitiveName, PrimitiveSpec] = {}


def _reg(spec: PrimitiveSpec) -> None:
    REGISTRY[spec.name] = spec


# ---------------------------- skill group --------------------------------
_reg(PrimitiveSpec(
    name=PrimitiveName.NAVIGATE,
    group="skill",
    summary=(
        "Drive the mobile base. Three ways to say where: a named waypoint from "
        "the semantic map (preferred), an explicit map-frame pose, or a small "
        "relative offset from where the robot is standing now. Blocks until the "
        "base arrives or the skill gives up. Obstacle avoidance is handled for you."
    ),
    args=(
        ArgSpec("target", "string", required=False, doc=(
            "A waypoint name from the semantic map, e.g. 'living_room', "
            "'kitchen_table'. Prefer this: waypoints were placed at poses from "
            "which the relevant surface is actually reachable, which a "
            "geometrically valid pose often is not."
        )),
        ArgSpec("pose", "object {x, y, theta}", required=False,
                doc="Explicit map-frame goal, when no waypoint fits."),
        ArgSpec("relative", "object {dx, dy, dtheta}", required=False, doc=(
            "A small offset in the robot's own frame: dx forward, dy left, "
            "dtheta counter-clockwise. This is your re-staging tool -- after a "
            "failed pick, back off with {\"dx\": -0.2} and look again. Keep it "
            "under about half a metre; use a waypoint to cross a room."
        )),
        ArgSpec("tolerance_m", "number", required=False, default=0.25,
                doc="Acceptance radius in metres."),
        ArgSpec("stop", "object {kind, argument, threshold}", required=False,
                doc="Abort condition, checked during the drive and verified after."),
    ),
    postcondition="base is within tolerance of the goal and has stopped",
    preconditions=("arm is stowed, unless this is a small `relative` nudge",),
    guard=_guard_navigate,
    cost_hint="seconds to a minute; by far the slowest primitive",
    stop_kinds=("base_at_target", "object_in_view", "timeout"),
    examples=(
        {"action": "navigate", "args": {"target": "living_room"}},
        {"action": "navigate", "args": {"relative": {"dx": -0.2},
                                        "stop": {"kind": "timeout"}}},
    ),
))

_reg(PrimitiveSpec(
    name=PrimitiveName.PICK_UP,
    group="skill",
    summary=(
        "Pick up an object that is already in view and within reach. This skill "
        "is local and contact-rich: it does NOT navigate and it does NOT know "
        "the map. Stage it first -- navigate, look_at, observe -- so that "
        "closing on the object is all that is left to do."
    ),
    args=(
        ArgSpec("object", "string", doc=(
            "What to pick up, phrased as the skill was trained to hear it: "
            "short, concrete, one object. Name the graspable feature where you "
            "can -- 'the side wall of the clear drinking glass' beats 'the glass'."
        )),
        ArgSpec("target", "object {x, y, z}", required=False, doc=(
            "Map-frame point of the object, from a fresh `observe`. Strongly "
            "recommended: without it the skill has to find the object itself."
        )),
        ArgSpec("stop", "object {kind, argument, threshold}", required=False,
                doc="Declare how you will know it worked. Use object_in_gripper."),
        ArgSpec("timeout_s", "number", required=False, default=45.0,
                doc="Give up after this long and hand control back."),
    ),
    postcondition=("the gripper is closed with load, and the object is no longer "
                   "detected where it was"),
    preconditions=("object is in view and within reach", "gripper is empty"),
    guard=_guard_pick_up,
    cost_hint="tens of seconds",
    stop_kinds=("object_in_gripper", "timeout"),
    examples=({"action": "pick_up",
               "args": {"object": "the side wall of the clear drinking glass",
                        "target": {"x": 2.81, "y": 1.07, "z": 0.74},
                        "stop": {"kind": "object_in_gripper", "argument": "glass"}}},),
))

_reg(PrimitiveSpec(
    name=PrimitiveName.PUT_DOWN,
    group="skill",
    summary=(
        "Put the held object down on a surface that is already within reach. "
        "Local only -- it cannot cross a room, so navigate to the destination "
        "first and confirm the surface is in view."
    ),
    args=(
        ArgSpec("surface", "string", doc=(
            "Where to put it, phrased concretely: 'the empty area on the kitchen "
            "table', not 'down'."
        )),
        ArgSpec("target", "object {x, y, z}", required=False,
                doc="Map-frame point on the surface, from a fresh `observe`."),
        ArgSpec("release", "boolean", required=False, default=True,
                doc="Open the gripper once the object is supported."),
        ArgSpec("stop", "object {kind, argument, threshold}", required=False,
                doc="Declare how you will know it worked. Use object_at_target."),
        ArgSpec("timeout_s", "number", required=False, default=45.0),
    ),
    postcondition="the gripper is empty and the object is detected on the surface",
    preconditions=("holding something", "the surface is within reach"),
    guard=_guard_put_down,
    cost_hint="tens of seconds",
    stop_kinds=("object_at_target", "gripper_empty", "timeout"),
    examples=({"action": "put_down",
               "args": {"surface": "the empty area on the kitchen table",
                        "stop": {"kind": "object_at_target", "argument": "glass"}}},),
))

# ---------------------------- analytic group ------------------------------
_reg(PrimitiveSpec(
    name=PrimitiveName.LOOK_AT,
    group="analytic",
    summary=(
        "Aim the head camera: at a named region ('table_surface', 'floor', "
        "'ahead'), at a map-frame point, or at explicit pan/tilt angles. Cheap. "
        "Call this before `observe` whenever what you want is probably out of frame."
    ),
    args=(
        ArgSpec("target", "string", required=False,
                doc="Named view direction, or an object phrase to centre."),
        ArgSpec("point", "object {x, y, z}", required=False,
                doc="Map-frame point to centre in the view."),
        ArgSpec("pan_rad", "number", required=False, doc="Explicit head pan."),
        ArgSpec("tilt_rad", "number", required=False,
                doc="Explicit head tilt. Negative looks down."),
    ),
    postcondition="head has reached the commanded angles",
    cost_hint="~1 s",
    examples=({"action": "look_at", "args": {"target": "table_surface", "tilt_rad": -0.6}},),
))

_reg(PrimitiveSpec(
    name=PrimitiveName.OBSERVE,
    group="analytic",
    summary=(
        "Capture a fresh RGB-D frame and ground a natural-language query in it. "
        "Returns detections with a pixel location, a depth and a map-frame 3D "
        "point. This is your only way to turn a phrase like 'the glass' into "
        "coordinates. Observe immediately before a skill that must touch a "
        "specific object, and again afterwards to find out what actually happened."
    ),
    args=(
        ArgSpec("query", "string", doc=(
            "What to find, phrased concretely and visually: 'the drinking glass "
            "on the coffee table', not 'the object'. One target per call."
        )),
        ArgSpec("expect_count", "integer", required=False, default=1,
                doc="How many you expect; a mismatch is reported, not an error."),
    ),
    postcondition="detections returned for the query, or an explicit empty result",
    cost_hint="one grounder call, ~1-3 s",
    examples=({"action": "observe",
               "args": {"query": "the drinking glass on the coffee table"}},),
))

_reg(PrimitiveSpec(
    name=PrimitiveName.SET_GRIPPER,
    group="analytic",
    summary=(
        "Drive the gripper open or closed directly, bypassing the skills. Use it "
        "for recovery -- clearing a failed grasp -- and for a deliberate release. "
        "Do not use it to grasp: `pick_up` owns contact."
    ),
    args=(
        ArgSpec("state", "string", enum=("open", "close"), doc="Target state."),
        ArgSpec("effort", "number", required=False, default=0.4,
                doc="0..1 closing effort."),
    ),
    postcondition="gripper aperture matches the commanded state",
    guard=_guard_manipulate,
    cost_hint="~2 s",
    examples=({"action": "set_gripper", "args": {"state": "open"}},),
))

_reg(PrimitiveSpec(
    name=PrimitiveName.STOW,
    group="analytic",
    summary=(
        "Retract arm and wrist into the safe transit configuration. Required "
        "before every `navigate`. Refuses if something is in the gripper unless "
        "you pass carry=true."
    ),
    args=(
        ArgSpec("carry", "boolean", required=False, default=False,
                doc="Stow into the carry pose while holding an object."),
    ),
    postcondition="robot reports stowed and the arm is inside the base footprint",
    guard=_guard_manipulate,
    cost_hint="~7 s",
    examples=({"action": "stow", "args": {"carry": True}},),
))

# ----------------------------- control group ------------------------------
_reg(PrimitiveSpec(
    name=PrimitiveName.DONE,
    group="control",
    summary=(
        "End the episode. Only call this after an `observe` whose detections "
        "positively confirm the goal state. Reporting success you have not seen "
        "is the single most costly error available to you: it is recorded as a "
        "false success and it poisons memory for every future run."
    ),
    args=(
        ArgSpec("status", "string", enum=("success", "failure"), doc="Your verdict."),
        ArgSpec("rationale", "string", doc="What you observed that justifies it."),
    ),
    postcondition="episode terminates",
    examples=({"action": "done", "args": {
        "status": "success",
        "rationale": "observe returned the glass upright on the kitchen table at "
                     "(5.9, 2.7) and the gripper is empty"}},),
))


STOP_PREDICATE_DOC = """\
object_in_gripper   the gripper is closed and carrying load
                    argument = object phrase. The right stop for pick_up.
gripper_empty       the gripper is open, or closed with no load
object_at_target    the named object is detected at this call's `target`
                    argument = object phrase. The right stop for put_down.
object_in_view      the named object appears in a fresh observation
base_at_target      the base is within `threshold` metres of this call's goal
vlm_predicate       argument is a yes/no question about the scene. Correct but
                    slow and paid -- prefer a structured kind where one fits.
timeout             asserts nothing; the skill runs to its own completion

A stop predicate is checked TWICE: the supervisor evaluates it every tick while
the skill runs and stops the skill as soon as it holds, and the monitor checks
it again against a fresh observation afterwards. Most kinds cost nothing, so
declaring one is close to free and buys you an early return plus a free
correctness check."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
class ValidationError(ValueError):
    pass


def validate_call(action: PrimitiveName, args: dict[str, Any]) -> list[str]:
    """Return a list of readable problems; empty means valid.

    Problems are fed back to the planner as a REJECTED StepResult, so the
    wording is addressed to the model, not to a developer.
    """
    spec = REGISTRY.get(action)
    if spec is None:
        return [f"unknown action '{action}'"]
    problems: list[str] = []

    known = {a.name for a in spec.args}
    for key in args:
        if key not in known:
            problems.append(
                f"'{key}' is not an argument of {action.value}; valid arguments "
                f"are {sorted(known)}")
    for a in spec.args:
        if a.required and a.name not in args:
            problems.append(
                f"{action.value} requires the argument '{a.name}' ({a.doc.strip()})")
        if a.enum and a.name in args and args[a.name] not in a.enum:
            problems.append(
                f"{action.value}.{a.name} must be one of {list(a.enum)}, "
                f"got {args[a.name]!r}")

    # --- stop predicate: parsed and scoped, never just passed through ------
    if "stop" in args and args["stop"] is not None:
        from .schema import PrimitiveCall
        try:
            pred = PrimitiveCall(action=action, args=args).stop_predicate()
        except ValueError as e:
            problems.append(str(e))
            pred = None
        if pred is not None and spec.stop_kinds and pred.kind not in spec.stop_kinds:
            problems.append(
                f"stop.kind '{pred.kind}' does not apply to {action.value}; "
                f"useful kinds here are {list(spec.stop_kinds)}")
        if pred is not None and pred.kind in (
                "object_in_gripper", "object_at_target", "object_in_view",
                "vlm_predicate") and not pred.argument:
            problems.append(
                f"stop.kind '{pred.kind}' needs 'argument' -- the object phrase "
                "(or, for vlm_predicate, the yes/no question) it refers to")

    # --- primitive-specific sanity ----------------------------------------
    if action is PrimitiveName.NAVIGATE and not (
            {"target", "pose", "relative"} & set(args)):
        problems.append(
            "navigate needs one of 'target' (a waypoint name), 'pose', or "
            "'relative' (a small offset for re-staging)")
    if action is PrimitiveName.NAVIGATE and isinstance(args.get("relative"), dict):
        r = args["relative"]
        reach = (abs(float(r.get("dx", 0.0))) ** 2 + abs(float(r.get("dy", 0.0))) ** 2) ** 0.5
        if reach > 1.0:
            problems.append(
                f"relative offset of {reach:.2f} m is too large for re-staging; "
                "use a waypoint or an explicit pose to move that far")
    if action is PrimitiveName.LOOK_AT and not (
            {"target", "point", "pan_rad", "tilt_rad"} & set(args)):
        problems.append("look_at needs one of 'target', 'point', 'pan_rad' or 'tilt_rad'")
    return problems


def check_guard(action: PrimitiveName, state: RobotState,
                args: dict[str, Any] | None = None) -> str | None:
    spec = REGISTRY.get(action)
    if spec is None or spec.guard is None:
        return None
    return spec.guard(state, args or {})


def apply_defaults(action: PrimitiveName, args: dict[str, Any]) -> dict[str, Any]:
    spec = REGISTRY[action]
    out = dict(args)
    for a in spec.args:
        if not a.required and a.name not in out and a.default is not None:
            out[a.name] = a.default
    return out


def render_library(include: tuple[Group, ...] = ("analytic", "skill", "control")) -> str:
    """Render the vocabulary into prompt text. Single source of truth."""
    lines: list[str] = []
    titles = {
        "analytic": "ANALYTIC PRIMITIVES  (deterministic, reliable, cheap)",
        "skill": "SKILLS               (learned policies -- local, fallible, slow)",
        "control": "CONTROL",
    }
    for group in include:
        lines.append(f"\n### {titles[group]}\n")
        for spec in REGISTRY.values():
            if spec.group != group:
                continue
            lines.append(f"{spec.name.value}")
            lines.append(f"  {spec.summary}")
            for a in spec.args:
                req = "required" if a.required else f"optional, default={a.default!r}"
                enum = f" one of {list(a.enum)}" if a.enum else ""
                lines.append(f"    - {a.name} : {a.type} ({req}){enum}")
                if a.doc:
                    lines.append(f"        {a.doc.strip()}")
            if spec.preconditions:
                lines.append(f"  precondition: {'; '.join(spec.preconditions)}")
            lines.append(f"  postcondition: {spec.postcondition}")
            if spec.stop_kinds:
                lines.append(f"  stop kinds: {', '.join(spec.stop_kinds)}"
                             f"   (prefer '{spec.stop_kinds[0]}')")
            if spec.cost_hint:
                lines.append(f"  cost: {spec.cost_hint}")
            lines.append("")
    return "\n".join(lines)


ANALYTIC = tuple(n for n, s in REGISTRY.items() if s.group == "analytic")
SKILLS = tuple(n for n, s in REGISTRY.items() if s.group == "skill")
