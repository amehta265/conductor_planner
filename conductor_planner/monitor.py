"""Progress monitoring, post-conditions and goal verification.

Nobody is standing next to the robot to say whether the glass made it, so the
monitor has to manufacture the success signal. It does so in two modes.

**While a skill runs** -- `Supervisor`, below. Skills are local objects that can
be stopped from outside, so the monitor watches them tick by tick and cuts them
short the moment something is wrong, instead of waiting for a blocking call to
finish failing.

**After every step** -- the detectors and checks below, against a fresh
observation.

Both use the same four layers, cheapest first:

  1. **State detectors** -- pure functions over joint state and detections.
     Free, deterministic, and they catch the failures that matter most (empty
     grasp, out of reach) without spending a single token.
  2. **Stop-predicate checks** -- the call declared a structured, checkable
     outcome; verify it actually holds. Also free.
  3. **Post-conditions** -- did the primitive do what its spec says it does?
  4. **VLM verification** -- only for the goal predicate and for explicit
     `vlm_predicate` stops, because each one is a paid round trip.

Layers 1 and 2 are the ones people skip, and they are the ones that make the
difference. An empty grasp is detectable from gripper effort in microseconds;
leave it to the VLM and you pay a second per check for a worse answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .backends import ChatMessage, VLMBackend
from .memory import GlobalMemory
from .schema import (Detection, Observation, Outcome, PrimitiveCall,
                     PrimitiveName, RobotState, SkillStatus, StepResult,
                     StopCause, StopPredicate)

# Thresholds. Measure these on the robot once, and change them only here.
EMPTY_GRASP_EFFORT = 0.10       # below this, the gripper closed on air
CLOSED_APERTURE = 0.25
REACH_LIMIT_M = 0.80            # Stretch 4 telescoping arm, practical limit
AT_TARGET_M = 0.15              # how close counts as "the object is there"
APERTURE_SETTLED = 0.01         # aperture change below this counts as "stopped"
CONFIRM_SAMPLES = 3             # consecutive settled samples before judging a grasp


@dataclass
class Verdict:
    ok: bool
    detector: str | None = None
    detail: str = ""


DetectorFn = Callable[[PrimitiveCall, StepResult, Observation, Observation], Verdict]
DETECTORS: dict[str, DetectorFn] = {}


def detector(name: str) -> Callable[[DetectorFn], DetectorFn]:
    def deco(fn: DetectorFn) -> DetectorFn:
        DETECTORS[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------
# Layer 1: state detectors
# --------------------------------------------------------------------------
def _gripper(obs: Observation) -> tuple[float | None, float | None]:
    s = obs.state
    return s.gripper_aperture, s.gripper_effort


@detector("empty_grasp")
def _empty_grasp(call, res, before, after) -> Verdict:
    """The single most important check in the system.

    A gripper that closed on nothing reports the same 'closed' as one holding a
    glass. The discriminator is effort: squeezing an object draws sustained
    current, closing on air does not.

    No debounce here, unlike the supervisor's live version: this runs after the
    skill has ended, so the gripper is stationary and one sample is the truth.
    """
    if call.action is not PrimitiveName.PICK_UP:
        return Verdict(True)
    ap, eff = _gripper(after)
    if ap is None or eff is None:
        # fall back to the final state the skill reported
        st = res.skill_state or {}
        ap, eff = st.get("gripper_aperture"), st.get("gripper_effort")
    if ap is None or eff is None:
        return Verdict(True, detail="gripper feedback unavailable; cannot check")
    if ap <= CLOSED_APERTURE and eff < EMPTY_GRASP_EFFORT:
        return Verdict(False, "empty_grasp",
                       f"the gripper closed to {ap:.2f} but effort is only {eff:.2f} "
                       f"(below {EMPTY_GRASP_EFFORT}) -- it is holding air")
    if ap > CLOSED_APERTURE:
        return Verdict(False, "empty_grasp",
                       f"the gripper did not close (aperture {ap:.2f})")
    return Verdict(True)


@detector("out_of_reach")
def _out_of_reach(call, res, before, after) -> Verdict:
    tgt = call.args.get("target")
    if not isinstance(tgt, dict) or "x" not in tgt:
        return Verdict(True)
    if call.action not in (PrimitiveName.PICK_UP, PrimitiveName.PUT_DOWN):
        return Verdict(True)
    b = before.state.base_pose
    d = ((tgt["x"] - b.x) ** 2 + (tgt["y"] - b.y) ** 2) ** 0.5
    if d > REACH_LIMIT_M:
        return Verdict(False, "out_of_reach",
                       f"the target is {d:.2f} m from the base, beyond the "
                       f"{REACH_LIMIT_M:.2f} m workspace; this is a staging problem, "
                       "not a manipulation problem")
    return Verdict(True)


@detector("not_stowed")
def _not_stowed(call, res, before, after) -> Verdict:
    if (call.action is PrimitiveName.NAVIGATE
            and not call.args.get("relative")
            and not before.state.is_stowed):
        return Verdict(False, "not_stowed",
                       "navigate to a waypoint was issued with the arm extended")
    return Verdict(True)


@detector("empty_observation")
def _empty_observation(call, res, before, after) -> Verdict:
    if call.action is PrimitiveName.OBSERVE and not res.detections:
        return Verdict(False, "empty_observation",
                       f"nothing matched {call.args.get('query')!r} in this view")
    return Verdict(True)


@detector("arrived_wrong_place")
def _arrived_wrong_place(call, res, before, after) -> Verdict:
    """Navigation reported success, but the base is not where it was sent.

    This is the quiet failure: Nav2 says "goal reached", localisation has
    drifted, and the robot is confidently in the wrong room. Nothing downstream
    catches it directly -- the next `observe` just fails to find the object, and
    the planner concludes the object is missing rather than that it is in the
    wrong place. Comparing the reported goal against the reported pose costs
    two subtractions and turns a confusing failure into a legible one.
    """
    if call.action is not PrimitiveName.NAVIGATE:
        return Verdict(True)
    st = res.skill_state or {}
    if "goal_x" not in st:
        return Verdict(True, detail="the navigate skill did not report its goal")
    b = after.state.base_pose
    tol = float(st.get("tolerance_m", 0.25))
    d = ((b.x - st["goal_x"]) ** 2 + (b.y - st["goal_y"]) ** 2) ** 0.5
    if d <= max(tol * 2.0, 0.3):
        return Verdict(True)
    return Verdict(False, "arrived_wrong_place",
                   f"navigate reported finishing, but the base is at "
                   f"({b.x:.2f}, {b.y:.2f}) and the goal was "
                   f"({st['goal_x']:.2f}, {st['goal_y']:.2f}) -- {d:.2f} m away, "
                   f"well outside the {tol:.2f} m tolerance")


@detector("skill_timeout")
def _skill_timeout(call, res, before, after) -> Verdict:
    if res.outcome is Outcome.TIMEOUT or res.stopped_by is StopCause.TIMEOUT:
        return Verdict(False, "skill_timeout",
                       f"{call.action.value} ran out its clock without reaching its "
                       "own stopping point, so the scene may be half-changed")
    return Verdict(True)


@detector("stopped_by_supervisor")
def _stopped_by_supervisor(call, res, before, after) -> Verdict:
    """The supervisor cut the skill short mid-flight.

    Worth its own detector because the recovery differs from a skill that ran
    to completion and failed: the scene is half-changed at an arbitrary point,
    so the first move is always to look, not to retry.
    """
    if res.stopped_by is StopCause.MONITOR:
        return Verdict(False, "stopped_by_supervisor",
                       f"{call.action.value} was stopped mid-flight: {res.message}")
    return Verdict(True)


@detector("object_did_not_move")
def _object_did_not_move(call, res, before, after) -> Verdict:
    """After a pick, the object should no longer be where it was."""
    if call.action is not PrimitiveName.PICK_UP:
        return Verdict(True)
    tgt = call.args.get("target")
    if not isinstance(tgt, dict) or "x" not in tgt:
        return Verdict(True)
    for d in after.detections:
        if d.point_3d is None:
            continue
        dist = ((d.point_3d.x - tgt["x"]) ** 2 + (d.point_3d.y - tgt["y"]) ** 2) ** 0.5
        if dist < 0.12:
            return Verdict(False, "empty_grasp",
                           f"'{d.query}' is still detected {dist:.2f} m from the pick "
                           "target -- the grasp did not take")
    return Verdict(True)


# --------------------------------------------------------------------------
# Layer 2: stop predicates
# --------------------------------------------------------------------------
def evaluate_stop_predicate(
    pred: StopPredicate,
    call: PrimitiveCall,
    res: StepResult,
    after: Observation,
    verify: Callable[[str, str | None], tuple[bool, float, str]] | None = None,
) -> Verdict:
    """Check a declared stop predicate against reality after the call.

    This is what turns `stop` from decoration into a contract. The planner
    states, in structured form, how it will know the call worked; we check it
    for free; a skill that ends without its declared stop condition holding is
    caught on the spot rather than three steps later.
    """
    kind, arg = pred.kind, (pred.argument or "").lower().strip()
    ap, eff = _gripper(after)
    st = res.skill_state or {}
    if ap is None:
        ap = st.get("gripper_aperture")
    if eff is None:
        eff = st.get("gripper_effort")

    if kind == "timeout":
        return Verdict(True)

    if kind == "object_in_gripper":
        thr = pred.threshold if pred.threshold is not None else EMPTY_GRASP_EFFORT
        if ap is None or eff is None:
            return Verdict(True, detail="no gripper feedback; stop predicate unchecked")
        if ap <= CLOSED_APERTURE and eff >= thr:
            return Verdict(True)
        return Verdict(False, "stop_predicate_false",
                       f"you declared stop=object_in_gripper({pred.argument}), but the "
                       f"gripper is at aperture {ap:.2f} / effort {eff:.2f}. "
                       "You are not holding it.")

    if kind == "gripper_empty":
        if ap is None:
            return Verdict(True, detail="no gripper feedback; stop predicate unchecked")
        if ap > CLOSED_APERTURE or (eff or 0.0) < EMPTY_GRASP_EFFORT:
            return Verdict(True)
        return Verdict(False, "stop_predicate_false",
                       f"you declared stop=gripper_empty, but the gripper is closed "
                       f"at {ap:.2f} with effort {eff:.2f} -- something is still in it")

    if kind in ("object_at_target", "object_in_view"):
        hits = [d for d in after.detections
                if arg in d.query.lower() or arg in (d.label or "").lower()]
        if not hits:
            return Verdict(False, "stop_predicate_unverified",
                           f"you declared stop={kind}({pred.argument}), but no live "
                           "detection covers it. Call observe before trusting this step.")
        if kind == "object_in_view":
            return Verdict(True)
        tgt = call.args.get("target")
        if not isinstance(tgt, dict) or "x" not in tgt:
            return Verdict(True, detail="no target point to compare against")
        thr = pred.threshold if pred.threshold is not None else AT_TARGET_M
        for d in hits:
            if d.point_3d is None:
                continue
            dist = ((d.point_3d.x - tgt["x"]) ** 2 + (d.point_3d.y - tgt["y"]) ** 2) ** 0.5
            if dist <= thr:
                return Verdict(True)
        return Verdict(False, "stop_predicate_false",
                       f"you declared stop=object_at_target({pred.argument}), but the "
                       f"nearest detection of it is not within {thr:.2f} m of the target")

    if kind == "base_at_target":
        goal = call.args.get("pose")
        if not (isinstance(goal, dict) and "x" in goal) and "goal_x" in st:
            # the call named a waypoint; the skill resolved it to a pose and
            # reported it, so the claim is checkable after all
            goal = {"x": st["goal_x"], "y": st["goal_y"]}
        if not isinstance(goal, dict) or "x" not in goal:
            return Verdict(True, detail="no goal pose available to compare against")
        b = after.state.base_pose
        thr = pred.threshold if pred.threshold is not None else 0.3
        dist = ((b.x - goal["x"]) ** 2 + (b.y - goal["y"]) ** 2) ** 0.5
        if dist <= thr:
            return Verdict(True)
        return Verdict(False, "stop_predicate_false",
                       f"you declared stop=base_at_target, but the base finished "
                       f"{dist:.2f} m from the goal (tolerance {thr:.2f} m)")

    if kind == "vlm_predicate":
        if verify is None:
            return Verdict(True, detail="no verification backend; predicate unchecked")
        ok_, conf, ev = verify(pred.argument or "", after.rgb_path)
        if ok_ and conf >= 0.6:
            return Verdict(True)
        return Verdict(False, "stop_predicate_false",
                       f"you declared stop=vlm_predicate({pred.argument!r}); looking at "
                       f"the scene, that is not true ({ev})")

    return Verdict(True)


# --------------------------------------------------------------------------
# Layers 3/4: the Monitor
# --------------------------------------------------------------------------
VERIFY_PROMPT = """\
You are the verification module of a household robot. You are shown what the
robot's camera sees right now. Answer one question about the physical scene.

QUESTION: {question}

Reply with ONLY:
{{"answer": true/false, "confidence": 0.0-1.0, "evidence": "<what in the image decides it>"}}

Be strict. If you cannot see the thing the question is about, the answer is
false with low confidence and evidence saying what is missing. A wrong "true"
here makes the robot stop working and report success while the task is
unfinished, which is the worst outcome available to you."""


class Monitor:
    def __init__(self, backend: VLMBackend | None = None,
                 global_memory: GlobalMemory | None = None,
                 stall_limit: int = 3):
        self.backend = backend
        self.gm = global_memory
        self.stall_limit = stall_limit
        self._recent: list[tuple[PrimitiveName, str]] = []
        self.fired: list[str] = []          # detector names, for memory consolidation
        self._supervision_frame: str | None = None

    # --------------------------------------------------------- layers 1-3
    def check(self, call: PrimitiveCall, res: StepResult,
              before: Observation, after: Observation) -> list[Verdict]:
        """Run every detector and the declared stop predicate."""
        out: list[Verdict] = []
        for name, fn in DETECTORS.items():
            try:
                v = fn(call, res, before, after)
            except Exception as e:            # a detector must never kill an episode
                v = Verdict(True, detail=f"detector {name} errored: {e}")
            if not v.ok:
                out.append(v)

        try:
            pred = call.stop_predicate()
        except ValueError:
            pred = None
        if pred is not None:
            v = evaluate_stop_predicate(pred, call, res, after, self.verify)
            res.stop_predicate_met = v.ok
            if not v.ok:
                out.append(v)

        for v in out:
            if v.detector:
                self.fired.append(v.detector)
                if self.gm:
                    self.gm.note_failure(v.detector)
        return out

    def advice_for(self, verdicts: list[Verdict]) -> list[str]:
        """Turn fired checks into planner-readable recovery instructions."""
        out: list[str] = []
        for v in verdicts:
            fm = self.gm.by_detector(v.detector) if (self.gm and v.detector) else None
            if fm:
                out.append(f"[{fm.name}] {v.detail}. {fm.recovery}")
            else:
                out.append(v.detail)
        return out

    # ----------------------------------------------------------- layer 4
    def verify(self, question: str, rgb_path: str | None) -> tuple[bool, float, str]:
        """Ask the VLM a yes/no question about the current scene."""
        if self.backend is None or rgb_path is None:
            return (False, 0.0, "no verification backend or image available")
        msgs = [ChatMessage("user", VERIFY_PROMPT.format(question=question),
                            images=[rgb_path])]
        try:
            obj = self.backend.chat_json(msgs, max_tokens=300, temperature=0.0, retries=1)
        except Exception as e:
            return (False, 0.0, f"verification call failed: {e}")
        return (bool(obj.get("answer", False)),
                float(obj.get("confidence", 0.0)),
                str(obj.get("evidence", "")))

    def verify_goal(self, task: str, obs: Observation) -> tuple[bool, str]:
        """The goal predicate. Never trust the planner's own claim."""
        q = (f'The robot was told: "{task}". Looking at the scene right now, '
             "is that task fully and visibly complete?")
        ok_, conf, ev = self.verify(q, obs.rgb_path)
        return (ok_ and conf >= 0.6,
                f"visual check: {'complete' if ok_ else 'not complete'} "
                f"(confidence {conf:.2f}) -- {ev}")

    # ------------------------------------------------------- supervision
    def supervisor(self, call: PrimitiveCall,
                   predicate: StopPredicate | None = None, **kw) -> "Supervisor":
        """Build a supervisor for one skill run."""
        return Supervisor(self, call, predicate, **kw)

    # --------------------------------------------------------- stall check
    def note(self, call: PrimitiveCall, res: StepResult) -> str | None:
        """Detect the planner going in circles.

        A VLM planner's characteristic failure is not doing something wrong --
        it is doing the same reasonable thing over and over. Three identical
        calls in a row means the loop has lost the thread and has to be told so
        explicitly, because it cannot notice from inside the prompt.
        """
        sig = (call.action, repr(sorted(call.args.items()))[:160])
        self._recent.append(sig)
        self._recent = self._recent[-self.stall_limit:]
        if len(self._recent) == self.stall_limit and len(set(self._recent)) == 1:
            self._recent.clear()
            return (f"You have now issued {self.stall_limit} identical "
                    f"{call.action.value} calls with no change in the result. "
                    "Repeating it will not work. Change something concrete: a "
                    "different viewpoint (look_at / navigate), a different phrasing "
                    "for the skill, or a different subgoal ordering. If you believe "
                    "the task is impossible in this scene, call done(failure) and "
                    "say why.")
        return None

    def reset(self) -> None:
        self._recent.clear()
        self.fired.clear()


# --------------------------------------------------------------------------
# Supervision of a running skill
# --------------------------------------------------------------------------
@dataclass
class Decision:
    """What the supervisor concluded on one tick."""
    stop: bool = False
    cause: StopCause | None = None
    reason: str = ""
    detector: str | None = None
    cost: str = "free"          # "free" | "grounding" | "vlm"


class Supervisor:
    """Watches one running skill and decides when to stop it.

    THE COST PROBLEM, AND THE SHAPE OF THE ANSWER

    The obvious design -- feed every frame to a VLM and ask "is this going
    well?" -- does not survive contact with reality. A vision call costs a
    second or two of latency and real money, while a control loop wants an
    answer in milliseconds. Run it at 10 Hz against a 20-second pick and you
    have spent two hundred model calls to supervise one grasp.

    So supervision is tiered by cost, and the cheap tiers do almost all the
    work:

      FREE      every tick (default 10 Hz). Gripper aperture and effort, base
                pose, runstop, elapsed time, structural stop predicates, and a
                no-progress stall check. This catches the failures that
                actually happen mid-skill: the gripper closing on air, the base
                overshooting, the arm stalling against something.

      GROUNDING every `ground_every_s` (default 1.0 s), and only when the
                declared stop predicate needs to know where an object is.
                One grounder call.

      VLM       every `vlm_every_s` (default 5.0 s), and ONLY when the planner
                explicitly declared a `vlm_predicate`, or a cheap tier saw
                something ambiguous it could not settle. Off by default.

    The VLM tier is an exception handler, not the loop. If you find yourself
    wanting it every tick, what you actually want is a new free detector.

    A note on the images: frames captured while the robot is moving are motion
    blurred and often mid-occlusion. That is another reason the slow tiers are
    slow -- a blurred frame is a bad input to an expensive judgement.
    """

    def __init__(
        self,
        monitor: "Monitor",
        call: PrimitiveCall,
        predicate: StopPredicate | None = None,
        *,
        tick_hz: float = 10.0,
        ground_every_s: float = 1.0,
        vlm_every_s: float = 5.0,
        enable_vlm: bool | None = None,
        stall_after_s: float = 4.0,
        min_ticks_for_stall: int = 10,
        confirm_samples: int = CONFIRM_SAMPLES,
    ):
        self.m = monitor
        self.call = call
        self.pred = predicate
        self.tick_s = 1.0 / max(tick_hz, 1e-6)
        self.ground_every_s = ground_every_s
        self.vlm_every_s = vlm_every_s
        # VLM supervision is opt-in. Default: on only when the planner asked
        # for it by declaring a vlm_predicate.
        self.enable_vlm = (enable_vlm if enable_vlm is not None
                           else bool(predicate and predicate.kind == "vlm_predicate"))
        self.stall_after_s = stall_after_s
        self.min_ticks_for_stall = min_ticks_for_stall
        self.confirm_samples = confirm_samples
        self._prev_aperture: float | None = None
        self._closed_samples = 0
        self.ticks = 0
        self.vlm_checks = 0
        self.ground_checks = 0
        self._last_ground = 0.0
        self._last_vlm = 0.0
        self._last_state: dict[str, Any] = {}
        self._last_change = 0.0
        self._ticks_at_change = 0
        self.log: list[str] = []

    # ------------------------------------------------------------------
    def tick(self, status, ground=None) -> Decision:
        """One supervision step. `ground` is a callable that costs a model call.

        `ground(query) -> list[Detection]` is only invoked on the grounding
        tier, so passing it does not by itself cost anything.
        """
        self.ticks += 1
        t = status.elapsed_s

        d = self._free(status, t)
        if d.stop:
            return d
        if self.pred is not None:
            d = self._predicate_tier(status, t, ground)
            if d.stop:
                return d
        if self.enable_vlm and t - self._last_vlm >= self.vlm_every_s:
            self._last_vlm = t
            d = self._vlm(status)
            if d.stop:
                return d
        return Decision()

    # ------------------------------------------- tier 1: free, every tick
    def _free(self, status, t: float) -> Decision:
        st = status.state or {}
        if status.error:
            return Decision(True, StopCause.ERROR, status.error, cost="free")

        # The gripper closing on nothing is the single most valuable thing to
        # catch mid-skill: the sooner the planner knows, the less it has to
        # undo. Effort is the discriminator; aperture alone cannot tell a glass
        # from air.
        #
        # But it must be DEBOUNCED. A gripper does not snap shut, it ramps, and
        # on the way down it passes through "aperture below threshold, effort
        # still zero" -- which is indistinguishable from an empty grasp in a
        # single sample. Judging on one sample aborts perfectly good picks
        # mid-close, and then the planner backs off and retries something that
        # was working. So: only judge once the aperture has stopped moving.
        if self.call.action is PrimitiveName.PICK_UP:
            ap, eff = st.get("gripper_aperture"), st.get("gripper_effort")
            if ap is not None and eff is not None:
                settled = (self._prev_aperture is not None
                           and abs(ap - self._prev_aperture) <= APERTURE_SETTLED)
                if ap <= CLOSED_APERTURE and settled:
                    self._closed_samples += 1
                else:
                    self._closed_samples = 0
                self._prev_aperture = ap
                if (self._closed_samples >= self.confirm_samples
                        and eff < EMPTY_GRASP_EFFORT):
                    return Decision(
                        True, StopCause.MONITOR,
                        f"the gripper has been closed at {ap:.2f} for "
                        f"{self._closed_samples} samples with effort {eff:.2f} -- it "
                        "is holding air. Stopping now rather than lifting nothing.",
                        detector="empty_grasp", cost="free")

        # Stalled: the skill is RUNNING but achieving nothing -- pushing against
        # something, or reaching for a target that is not there.
        #
        # Two conditions, and the second is not optional. Wall-clock time alone
        # is not evidence of a stall: on a loaded machine a skill thread can go
        # seconds without being scheduled, and killing it then would be wrong
        # and maddening to debug. So we also require that the skill has actually
        # executed ticks in that window. Ticking with no state change is a
        # stall; not ticking at all is starvation, and the timeout handles that.
        if st != self._last_state:
            self._last_state = dict(st)
            self._last_change = t
            self._ticks_at_change = status.ticks
        elif (t - self._last_change > self.stall_after_s
              and status.ticks - self._ticks_at_change >= self.min_ticks_for_stall
              and status.progress is None and t > 1.0):
            return Decision(
                True, StopCause.MONITOR,
                f"{self.call.action.value} has executed "
                f"{status.ticks - self._ticks_at_change} control ticks over "
                f"{t - self._last_change:.1f}s with no change in hardware state "
                "-- it is stalled",
                detector="skill_stalled", cost="free")
        return Decision()

    # ------------------------ tier 2: structural predicate, mostly free
    def _predicate_tier(self, status, t: float, ground) -> Decision:
        p = self.pred
        st = status.state or {}
        ap, eff = st.get("gripper_aperture"), st.get("gripper_effort")

        if p.kind == "object_in_gripper" and ap is not None and eff is not None:
            thr = p.threshold if p.threshold is not None else EMPTY_GRASP_EFFORT
            if ap <= CLOSED_APERTURE and eff >= thr:
                return Decision(True, StopCause.PREDICATE,
                                f"stop predicate satisfied: holding {p.argument} "
                                f"(effort {eff:.2f})", cost="free")
        if p.kind == "gripper_empty" and ap is not None:
            if ap > CLOSED_APERTURE:
                return Decision(True, StopCause.PREDICATE,
                                "stop predicate satisfied: the gripper is open",
                                cost="free")
        if p.kind == "base_at_target":
            goal = self.call.args.get("pose")
            if not (isinstance(goal, dict) and "x" in goal) and "goal_x" in st:
                goal = {"x": st["goal_x"], "y": st["goal_y"]}
            if isinstance(goal, dict) and "x" in goal and "base_x" in st:
                thr = p.threshold if p.threshold is not None else 0.3
                d = ((st["base_x"] - goal["x"]) ** 2
                     + (st["base_y"] - goal["y"]) ** 2) ** 0.5
                if d <= thr:
                    return Decision(True, StopCause.PREDICATE,
                                    f"stop predicate satisfied: base within "
                                    f"{d:.2f} m of the goal", cost="free")

        # These two need to know where an object is, which costs a grounder
        # call -- so they are rate limited hard.
        if p.kind in ("object_in_view", "object_at_target") and ground is not None:
            if t - self._last_ground >= self.ground_every_s:
                self._last_ground = t
                self.ground_checks += 1
                dets = ground(p.argument or "")
                if p.kind == "object_in_view" and dets:
                    return Decision(True, StopCause.PREDICATE,
                                    f"stop predicate satisfied: {p.argument} is in view",
                                    cost="grounding")
                tgt = self.call.args.get("target")
                if p.kind == "object_at_target" and isinstance(tgt, dict) and dets:
                    thr = p.threshold if p.threshold is not None else AT_TARGET_M
                    for det in dets:
                        if det.point_3d is None:
                            continue
                        dist = ((det.point_3d.x - tgt["x"]) ** 2
                                + (det.point_3d.y - tgt["y"]) ** 2) ** 0.5
                        if dist <= thr:
                            return Decision(
                                True, StopCause.PREDICATE,
                                f"stop predicate satisfied: {p.argument} is at the "
                                f"target ({dist:.2f} m)", cost="grounding")
        return Decision()

    # ---------------------------------- tier 3: the model, rarely, opt-in
    def _vlm(self, status) -> Decision:
        rgb = getattr(self.m, "_supervision_frame", None)
        if rgb is None:
            return Decision()
        self.vlm_checks += 1
        if self.pred is not None and self.pred.kind == "vlm_predicate":
            question = self.pred.argument or ""
            ok_, conf, ev = self.m.verify(question, rgb)
            if ok_ and conf >= 0.6:
                return Decision(True, StopCause.PREDICATE,
                                f"stop predicate satisfied: {question} ({ev})",
                                cost="vlm")
            return Decision(cost="vlm")
        # generic safety net, only when explicitly enabled
        q = (f"The robot is running the skill '{self.call.action.value}' in order to: "
             f"{self.call.expect or self.call.subgoal or 'complete the current step'}. "
             "Is something visibly going wrong -- an object knocked over, the arm in "
             "the wrong place, the robot pushing against something?")
        ok_, conf, ev = self.m.verify(q, rgb)
        if ok_ and conf >= 0.7:
            return Decision(True, StopCause.MONITOR,
                            f"visual supervision flagged a problem: {ev}",
                            detector="visual_anomaly", cost="vlm")
        return Decision(cost="vlm")

    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (f"{self.ticks} supervision ticks, {self.ground_checks} grounder "
                f"calls, {self.vlm_checks} model calls")
