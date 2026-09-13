"""Typed contracts for the Conductor planner.

Everything that crosses a module boundary is one of these models. The planner
emits `PrimitiveCall`, the robot returns `StepResult`, memory stores `Trace`.
Nothing else is allowed across the seam -- that is what makes the VLM
swappable and the skills team's work independent of ours.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Spatial types
# --------------------------------------------------------------------------
class Point3D(BaseModel):
    """A point in a named frame. Default frame is the SLAM map frame."""
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    z: float = 0.0
    frame: str = "map"


class Pose2D(BaseModel):
    """A base pose on the floor plane, in the map frame."""
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    theta: float = 0.0
    frame: str = "map"


class Detection(BaseModel):
    """One grounded object instance produced by `observe`."""
    model_config = ConfigDict(extra="forbid")
    query: str                      # the phrase that produced it
    label: str                      # what the grounder called it
    confidence: float = 0.0
    pixel: tuple[int, int] | None = None      # (u, v) in the RGB image
    bbox: tuple[int, int, int, int] | None = None   # (x0, y0, x1, y1)
    point_3d: Point3D | None = None           # deprojected, map frame
    depth_m: float | None = None
    notes: str = ""


# --------------------------------------------------------------------------
# Observation: what the planner sees each turn
# --------------------------------------------------------------------------
class RobotState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_pose: Pose2D
    lift_m: float | None = None
    arm_m: float | None = None
    gripper_aperture: float | None = None   # 0.0 closed .. 1.0 open
    gripper_effort: float | None = None     # proxy for "am I holding something"
    head_pan_rad: float | None = None
    head_tilt_rad: float | None = None
    holding: str | None = None              # best guess at what is in the gripper
    is_stowed: bool = False
    battery_soc: float | None = None
    runstop_engaged: bool = False


class Observation(BaseModel):
    """One multimodal snapshot. `rgb_path` is what actually reaches the VLM.

    We pass file paths rather than arrays so observations are serialisable,
    loggable and replayable -- every episode can be re-run offline against a
    different planner model without the robot.
    """
    model_config = ConfigDict(extra="forbid")
    step: int
    t: float = Field(default_factory=time.time)
    rgb_path: str | None = None
    depth_path: str | None = None            # .npy, float32 metres
    camera_info: dict[str, Any] | None = None  # fx, fy, cx, cy, frame_id, width, height
    scan_summary: str | None = None          # one-line LiDAR digest, see robot/base.py
    state: RobotState
    detections: list[Detection] = Field(default_factory=list)
    semantic_context: list[str] = Field(default_factory=list)  # nearby map waypoints
    last_result: "StepResult | None" = None


# --------------------------------------------------------------------------
# Primitive calls: the only thing the planner is allowed to emit
# --------------------------------------------------------------------------
class PrimitiveName(str, Enum):
    # --- analytic (deterministic, ours) ---
    LOOK_AT = "look_at"
    OBSERVE = "observe"
    SET_GRIPPER = "set_gripper"
    STOW = "stow"
    # --- skills (local, startable/stoppable, supervised while they run) ---
    NAVIGATE = "navigate"
    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    # --- control ---
    DONE = "done"


# Names that used to exist, mapped to what replaced them. The planner may emit
# an old name from a stale memory trace or from its own priors; mapping it and
# telling it what happened is far better than a crash or a silent no-op.
RENAMED: dict[str, str] = {
    "navigate_to": "navigate",
    "vla_grab": "pick_up",
    "vla_pick_and_place": "put_down",
    "vla_move": "navigate",
    "grab": "pick_up",
    "place": "put_down",
    "pick": "pick_up",
}


class StopCause(str, Enum):
    """Why a skill stopped. Never "did it work" -- that is the monitor's call."""
    COMPLETED = "completed"     # the skill reached its own natural end
    PREDICATE = "predicate"     # a declared stop predicate fired mid-flight
    MONITOR = "monitor"         # the supervisor decided to cut it short
    TIMEOUT = "timeout"         # ran out its clock
    GUARD = "guard"             # a state guard tripped (runstop, joint limit)
    PLANNER = "planner"         # explicitly aborted from above
    ERROR = "error"             # the skill raised


class SkillStatus(BaseModel):
    """A cheap, non-blocking snapshot of a running skill.

    Polled at control rate by the supervisor, so it must cost nothing: read
    whatever the skill already has in memory, never trigger a sensor read or a
    computation. Anything expensive belongs in the supervisor's slower tiers.
    """
    model_config = ConfigDict(extra="forbid")
    running: bool = True
    phase: str = ""                  # free text: "approaching", "closing", ...
    elapsed_s: float = 0.0
    ticks: int = 0
    progress: float | None = None    # 0..1 if the skill can estimate it
    state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SkillReport(BaseModel):
    """What a skill did. Note the absence of a success field: that is deliberate.

    A skill reports mechanism -- how it stopped, how long it ran, where the
    hardware ended up. Whether the task worked is decided by the monitor from a
    fresh observation. A skill that grades its own homework produces the
    false-success failure mode, where the episode ends, the log says done, and
    the glass is still on the sofa.
    """
    model_config = ConfigDict(extra="forbid")
    skill: str
    stopped_by: StopCause = StopCause.COMPLETED
    reason: str = ""
    elapsed_s: float = 0.0
    ticks: int = 0
    final_state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


StopKind = Literal[
    "object_in_gripper",   # gripper closed AND effort above threshold
    "gripper_empty",       # gripper open, or closed with no load
    "object_at_target",    # the named object is detected at the call's target
    "object_in_view",      # the named object appears in a fresh observation
    "base_at_target",      # base within `threshold` metres of the call's goal
    "vlm_predicate",       # a yes/no question for the monitor VLM (a paid call)
    "timeout",             # no condition; run to the skill's own completion
]


class StopPredicate(BaseModel):
    """The machine-checkable form of a call's expected outcome.

    It does two jobs, and the second is the one that matters most here.

    **While the call runs** it is an abort condition. The supervisor evaluates
    it every tick against the skill's reported state and stops the skill the
    moment it is satisfied, so a skill returns as soon as it has done its job
    instead of grinding on to its own timeout. Most kinds cost nothing to
    check -- see `monitor.Supervisor`.

    **After the call** it is checked against reality by the monitor. This is
    what makes it worth having. `PrimitiveCall.expect` is prose, so verifying it
    costs a VLM round trip. A stop predicate is structured, so verifying it is
    free and deterministic -- and a skill that ends without its declared stop
    condition holding is caught immediately, which is exactly the false-success
    failure mode that ends episodes with the task undone.

    `kind` defaults to "timeout", which asserts nothing. Prefer a real condition
    whenever the call has one; the planner is told this in its prompt.
    """
    model_config = ConfigDict(extra="forbid")
    kind: StopKind = "timeout"
    argument: str | None = None       # object phrase, or the predicate question
    threshold: float | None = None    # metres, or an effort/aperture fraction

    def describe(self) -> str:
        bits = [self.kind]
        if self.argument:
            bits.append(f"({self.argument})")
        if self.threshold is not None:
            bits.append(f"threshold={self.threshold}")
        return " ".join(bits)


class PrimitiveCall(BaseModel):
    """A single planner action. This is the JSON the VLM must produce."""
    model_config = ConfigDict(extra="forbid")
    action: PrimitiveName
    # free-form arguments, validated per-primitive in primitives.py
    args: dict[str, Any] = Field(default_factory=dict)
    # planner's own reasoning, kept for logs, memory abstraction and debugging
    rationale: str = ""
    # what the planner expects to be true afterwards; the monitor checks it
    expect: str = ""
    subgoal: str = ""      # which phase of the long-horizon task this serves
    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])

    def stop_predicate(self) -> "StopPredicate | None":
        """Parse `args["stop"]` into a typed predicate.

        Returns None when the call declares no stop condition. Raises
        ValueError with a message addressed to the planner when the predicate
        is malformed -- `primitives.validate_call` turns that into a REJECTED
        result so the model gets to fix it without the robot moving.
        """
        raw = self.args.get("stop")
        if raw is None:
            return None
        if isinstance(raw, StopPredicate):
            return raw
        if isinstance(raw, str):
            raw = {"kind": raw}
        if not isinstance(raw, dict):
            raise ValueError("'stop' must be an object like "
                             '{"kind": "object_in_gripper", "argument": "glass"}')
        try:
            return StopPredicate.model_validate(raw)
        except Exception as e:
            raise ValueError(f"'stop' is not a valid stop predicate: {e}") from None


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    REJECTED = "rejected"      # precondition or validation failed, never executed
    ABORTED = "aborted"        # safety / runstop / operator


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    call_id: str
    action: PrimitiveName
    outcome: Outcome
    message: str = ""
    duration_s: float = 0.0
    stop_reason: str | None = None           # why the skill returned
    stopped_by: StopCause | None = None      # who stopped it
    skill_state: dict[str, Any] = Field(default_factory=dict)
    supervisor_ticks: int = 0                # how many times it was checked
    vlm_checks: int = 0                      # how many of those cost a model call
    # set by the monitor when the call's declared stop predicate is checked
    stop_predicate_met: bool | None = None
    detections: list[Detection] = Field(default_factory=list)
    postcondition_met: bool | None = None
    # populated by the monitor when it disagrees with the primitive's own report
    monitor_verdict: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.SUCCESS


# --------------------------------------------------------------------------
# Memory records
# --------------------------------------------------------------------------
class TraceStep(BaseModel):
    """A memory-abstracted step: concrete coordinates replaced by queries."""
    model_config = ConfigDict(extra="forbid")
    action: PrimitiveName
    args: dict[str, Any]          # may contain {"$query": "..."} placeholders
    subgoal: str = ""
    note: str = ""


class Trace(BaseModel):
    """A successful primitive composition, parameterised for reuse."""
    model_config = ConfigDict(extra="forbid")
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str
    steps: list[TraceStep]
    n_success: int = 1
    n_attempts: int = 1
    created_at: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)

    @property
    def reliability(self) -> float:
        return self.n_success / max(self.n_attempts, 1)


class SuccessRule(BaseModel):
    """A prompting / sequencing strategy that was observed to work."""
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    scope: str                    # primitive name, subgoal, or "global"
    rule: str
    evidence: int = 1


class FailureModel(BaseModel):
    """A recognised failure mode and its prescribed recovery.

    `signature` is matched against the observation + last result by
    memory/global_memory.py. `recovery` is injected into the planner prompt as
    an instruction, not executed automatically -- the planner stays in charge.
    """
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    scope: str = "global"
    signature: str                # human-readable trigger description
    detector: str | None = None   # key into monitor.DETECTORS, if automatable
    recovery: str                 # what the planner should do instead
    seen: int = 1


# --------------------------------------------------------------------------
# Episode log
# --------------------------------------------------------------------------
class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str
    mode: Literal["bootstrap", "deploy"] = "deploy"
    started_at: float = Field(default_factory=time.time)
    calls: list[PrimitiveCall] = Field(default_factory=list)
    results: list[StepResult] = Field(default_factory=list)
    succeeded: bool = False
    termination_reason: str = ""
    replans: int = 0


Observation.model_rebuild()
