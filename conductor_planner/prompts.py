"""Prompt construction.

One principle governs this file: the prompt is *generated*, never hand-kept.
The primitive list, the stop-predicate list and the memory sections are all
rendered from the code and data that actually run, so a prompt that describes
a primitive you deleted is structurally impossible.

Two layers, split by how often they change, because that split is now worth
real money.

SYSTEM PROMPT -- built once per episode, byte-identical on every turn, and
marked cacheable by the Anthropic backend:

    role and task
    primitive vocabulary          <- primitives.render_library()
    stop predicates               <- primitives.STOP_PREDICATE_DOC
    output contract
    semantic map
    memory catalogue              <- ALL success rules and failure models
    retrieved trace               <- the parameterised trace for this task

PER-TURN MESSAGE -- cheap, and different every turn:

    memory for this step          <- rules/failures scoped to what just happened
    trace progress                <- where you are in the retrieved trace
    live observation              <- state, detections, monitor verdict, image

Anything volatile in the system prompt would invalidate the prompt cache on
every turn and multiply the input bill by roughly the number of steps. So the
catalogue is static and the *scoping* is dynamic -- see `planner.memory_block`.
"""
from __future__ import annotations

import json
from typing import Any

from .primitives import STOP_PREDICATE_DOC, render_library
from .schema import Observation, PrimitiveCall, StepResult

SYSTEM_CORE = """\
You are the planner of a household mobile manipulator (a Hello Robot Stretch 4:
a wheeled base, a telescoping arm on a vertical lift, a one-DoF gripper, and a
head-mounted RGB-D camera). You have a prebuilt SLAM map of the home with named
waypoints.

You do not output motor commands. You output ONE primitive call at a time from
a fixed vocabulary, you observe the result, and you decide the next one. Three
of those primitives -- navigate, pick_up, put_down -- are skills: they take
real time, and a supervisor watches them while they run and can stop them part
way. The rest are yours and complete immediately.

Hold these four things in mind, because they are what the architecture is for:

1. THE SKILLS ARE LOCAL SPECIALISTS, NOT AGENTS. pick_up and put_down see the
   current camera frame and nothing else. They have no map, no memory, no idea
   what the overall task is, and they cannot drive the base to reach something.
   They are good at the last thirty centimetres and bad at everything else.
   Your job is to put them in a position where the last thirty centimetres is
   all that is left.

2. STAGE, THEN CALL, THEN LOOK. Before a skill that must touch a specific
   object: navigate so it is reachable, look_at so it is in frame, observe so
   you have a live 3D point. Afterwards: observe again. Your one real advantage
   over a skill is that you can see before and after it runs -- spend it.

3. DECLARE A STOP PREDICATE. Every skill call that has a checkable outcome
   should say, in the `stop` argument, how you will know it worked. It is
   verified against the scene for free after the call. A skill that reports no
   error while its stop predicate is false is a failure you get to catch now
   instead of three steps later. Coordinates from two minutes ago are fiction;
   so is a skill's silence.

4. NEVER REPORT SUCCESS YOU HAVE NOT SEEN. done(success) requires a fresh
   observe showing the goal state. Claiming success from belief is recorded as
   a false success and is the worst error available to you.
"""

OUTPUT_CONTRACT = """\
### OUTPUT CONTRACT

Reply with exactly ONE JSON object and nothing else -- no prose before or
after, no markdown fence:

{
  "rationale": "<one or two sentences: what you believe the state is and why this call>",
  "subgoal":   "<which phase of the task this serves, e.g. 'reach the glass'>",
  "action":    "<primitive name>",
  "args":      { ... },
  "expect":    "<what will be true if this works -- it will be checked against reality>"
}

`expect` is not decoration. A monitor compares it against the next observation,
and a mismatch is fed back to you. Write something falsifiable.
"""


def call_json_schema() -> dict[str, Any]:
    """JSON schema for guided decoding. Small models need this."""
    from .schema import PrimitiveName
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "subgoal": {"type": "string"},
            "action": {"type": "string", "enum": [p.value for p in PrimitiveName]},
            "args": {"type": "object"},
            "expect": {"type": "string"},
        },
        "required": ["action", "args", "rationale"],
        "additionalProperties": False,
    }


def build_system_prompt(task: str, global_memory_text: str,
                        semantic_map: dict[str, Any] | None = None,
                        extra_notes: str = "") -> str:
    parts = [SYSTEM_CORE, f"\n### THE TASK\n\n{task}\n",
             "### PRIMITIVE VOCABULARY",
             "You may not invent primitives. These are all of them.",
             render_library(),
             "### STOP PREDICATES (the `stop` argument of a skill call)\n",
             STOP_PREDICATE_DOC, "", OUTPUT_CONTRACT]
    if semantic_map:
        parts += ["### SEMANTIC MAP (waypoints you may navigate to by name)\n",
                  _render_map(semantic_map), ""]
    parts += ["### MEMORY: WHAT PAST RUNS TAUGHT US\n", global_memory_text, ""]
    if extra_notes:
        parts += ["### NOTES FOR THIS DEPLOYMENT\n", extra_notes, ""]
    return "\n".join(parts)


def _render_map(m: dict[str, Any]) -> str:
    lines = []
    for name, info in (m.get("waypoints") or {}).items():
        if isinstance(info, dict):
            pose = info.get("pose", {})
            desc = info.get("description", "")
            room = info.get("room", "")
            lines.append(f"  {name}  ({pose.get('x', '?')}, {pose.get('y', '?')}, "
                         f"theta {pose.get('theta', '?')})"
                         + (f"  [{room}]" if room else "")
                         + (f"  -- {desc}" if desc else ""))
        else:
            lines.append(f"  {name}  {info}")
    return "\n".join(lines) or "  (no waypoints defined)"


def render_observation(obs: Observation, step_budget_left: int) -> str:
    s = obs.state
    lines = [f"### STEP {obs.step}   ({step_budget_left} steps left in the budget)", ""]
    lines.append("ROBOT STATE")
    lines.append(f"  base pose      : ({s.base_pose.x:.2f}, {s.base_pose.y:.2f}, "
                 f"theta {s.base_pose.theta:.2f}) in the map frame")
    if s.lift_m is not None:
        lines.append(f"  lift / arm     : {s.lift_m:.2f} m / {s.arm_m:.2f} m")
    if s.gripper_aperture is not None:
        lines.append(f"  gripper        : aperture {s.gripper_aperture:.2f} "
                     f"(1 open, 0 closed), effort {s.gripper_effort:.2f}")
    lines.append(f"  holding        : {s.holding or 'nothing'}")
    lines.append(f"  arm stowed     : {s.is_stowed}")
    if s.head_pan_rad is not None:
        lines.append(f"  head           : pan {s.head_pan_rad:.2f}, tilt {s.head_tilt_rad:.2f}")
    if s.runstop_engaged:
        lines.append("  RUNSTOP ENGAGED -- no motion is possible until it is released")
    if obs.scan_summary:
        lines.append(f"  lidar          : {obs.scan_summary}")
    if obs.semantic_context:
        lines.append("\nNEARBY MAP WAYPOINTS")
        for c in obs.semantic_context:
            lines.append(f"  {c}")
    if obs.detections:
        lines.append("\nLIVE DETECTIONS (from your most recent observe)")
        for d in obs.detections:
            p = d.point_3d
            where = (f"map point ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})" if p
                     else "NO 3D POINT -- depth was unusable here")
            lines.append(f"  '{d.label}' for query '{d.query}': {where}, "
                         f"confidence {d.confidence:.2f}")
            if d.notes:
                lines.append(f"      note: {d.notes}")
    else:
        lines.append("\nLIVE DETECTIONS: none. Nothing has been grounded in the current view.")
    if obs.last_result is not None:
        r = obs.last_result
        lines.append(f"\nRESULT OF YOUR LAST CALL ({r.action.value}) -> {r.outcome.value}")
        lines.append(f"  {r.message}")
        if r.stop_reason:
            lines.append(f"  the skill stopped because: {r.stop_reason}")
        if r.stop_predicate_met is False:
            lines.append("  your declared stop predicate was checked and is FALSE")
        if r.monitor_verdict:
            lines.append(f"  MONITOR: {r.monitor_verdict}")
    lines.append("\nThe attached image is the robot's current camera view.")
    lines.append("\nEmit the next primitive call.")
    return "\n".join(lines)


def render_rejection(problems: list[str]) -> str:
    return ("Your last call was REJECTED before execution and the robot did not "
            "move. Fix it and emit a corrected call:\n"
            + "\n".join(f"  - {p}" for p in problems))


# --------------------------------------------------------------------------
# Per-turn memory block
# --------------------------------------------------------------------------
def render_memory_block(
    rules: list,
    failures: list,
    trace_progress: str = "",
    recent_detectors: list[str] | None = None,
) -> str:
    """The dynamic half of memory, rebuilt every turn.

    The catalogue already sits in the (cached) system prompt, so this block does
    not repeat it. It narrows: the handful of rules and failure modes that bear
    on what the planner just did and is about to do, plus where it has got to in
    the retrieved trace. Narrowing is the whole value -- a model that is shown
    eight failure modes every turn stops reading them.
    """
    out: list[str] = ["### MEMORY FOR THIS STEP", ""]
    if recent_detectors:
        out.append("Just fired, and therefore the most relevant thing on this page:")
        for d in dict.fromkeys(recent_detectors):
            out.append(f"  ! {d}")
        out.append("")
    if rules:
        out.append("Rules that apply here:")
        out += [f"  - {r.rule}" for r in rules]
        out.append("")
    if failures:
        out.append("Failure modes that apply here:")
        for f in failures:
            out.append(f"  - {f.name}: {f.signature}")
            out.append(f"      if it happens: {f.recovery}")
        out.append("")
    if trace_progress:
        out.append(trace_progress)
        out.append("")
    if len(out) <= 2:
        return ""
    return "\n".join(out)


def render_trace_progress(trace, cursor: int, unbound: list[str]) -> str:
    """Where the planner has got to in the retrieved trace.

    Deliberately phrased as position, not instruction. The trace is a
    suggestion; a scene that differs from the one it was recorded in should
    override it, and saying "you are at step 5" invites that judgement far more
    than repeating the whole trace would.
    """
    if trace is None:
        return ""
    lines = [f"Retrieved trace {trace.trace_id} "
             f"(worked {trace.n_success}/{trace.n_attempts} times) -- you are at "
             f"step {min(cursor + 1, len(trace.steps))} of {len(trace.steps)}."]
    done = ", ".join(s.action.value for s in trace.steps[:cursor]) or "nothing yet"
    lines.append(f"  done so far, per the trace: {done}")
    if cursor < len(trace.steps):
        nxt = trace.steps[cursor]
        lines.append(f"  the trace would do next: {nxt.action.value} "
                     f"{json.dumps(nxt.args)}")
    else:
        lines.append("  the trace is exhausted; you are past the end of it.")
    lines.append("  It is a suggestion from a different scene, not a script. If what "
                 "you can see disagrees with it, believe what you can see.")
    if unbound:
        lines.append(f"  still unresolved in the trace: {unbound} -- each needs an "
                     "`observe` before you can use the step that refers to it.")
    return "\n".join(lines)
