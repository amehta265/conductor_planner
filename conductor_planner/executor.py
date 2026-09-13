"""The Conductor: the turn-based loop that mediates planner and robot.

This is the REPL. Every turn:

    observe -> retrieve memory -> planner emits a call -> validate ->
    guard -> execute -> observe again -> monitor -> feed the verdict back

Two modes:

  bootstrap   generous step budget, failures are cheap, the point is to
              *discover* a working composition. At the end, a successful
              episode is abstracted into Task-Specific Memory and the
              failures it survived become Global Memory failure models.

  deploy      strict budget, no reset. The planner retrieves the trace it
              learned and grounds it against the live scene.

The thing to notice in `step()` is how much happens *before* the robot moves:
schema validation, then an embodiment guard, then -- only then -- execution.
Every call the conductor rejects is a robot motion that never had to be
recovered from, and rejection costs one cheap round trip instead of thirty
seconds and a knocked-over glass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backends import VLMBackend
from .grounding import CameraInfo, Grounder
from .memory import GlobalMemory, TaskMemory
from .monitor import Monitor, Verdict
from .planner import Planner, finalize, validate
from .robot.base import RobotInterface
from .schema import (Detection, EpisodeRecord, Observation, Outcome, Point3D,
                     Pose2D, PrimitiveCall, PrimitiveName, StepResult, StopCause,
                     StopPredicate, Trace)


class Conductor:
    def __init__(
        self,
        robot: RobotInterface,
        planner: Planner,
        grounder: Grounder,
        monitor: Monitor,
        task_memory: TaskMemory,
        global_memory: GlobalMemory,
        *,
        mode: str = "deploy",
        max_steps: int = 30,
        detection_ttl: int = 3,
        supervision: dict[str, Any] | None = None,
        log_dir: str | Path | None = None,
        on_step: Callable[[PrimitiveCall, StepResult], None] | None = None,
    ):
        self.robot = robot
        self.planner = planner
        self.grounder = grounder
        self.monitor = monitor
        self.tm = task_memory
        self.gm = global_memory
        self.mode = mode
        self.max_steps = max_steps
        self.detection_ttl = detection_ttl
        # passed straight to Monitor.supervisor(); see monitor.Supervisor for
        # what each knob costs
        self.supervision = supervision or {}
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.on_step = on_step
        self._detections: list[tuple[int, Detection]] = []
        self._step = 0

    # ------------------------------------------------------------ observe
    def observe(self, last: StepResult | None = None) -> Observation:
        rgb, depth, cam, T = self.robot.capture()
        live = [d for age, d in self._detections if self._step - age <= self.detection_ttl]
        return Observation(
            step=self._step,
            rgb_path=rgb,
            depth_path=None,
            camera_info={"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                         "width": cam.width, "height": cam.height, "frame_id": cam.frame_id},
            scan_summary=self.robot.scan_summary(),
            state=self.robot.get_state(),
            detections=live,
            semantic_context=self.robot.semantic_context(),
            last_result=last,
        )

    # ------------------------------------------------------------ dispatch
    def execute(self, call: PrimitiveCall) -> StepResult:
        t0 = time.time()
        a, args = call.action, call.args
        try:
            pred = call.stop_predicate()
        except ValueError:
            pred = None          # validation already rejected it; belt and braces
        try:
            if a in (PrimitiveName.NAVIGATE, PrimitiveName.PICK_UP,
                     PrimitiveName.PUT_DOWN):
                res = self.run_skill(call, pred)
            elif a is PrimitiveName.LOOK_AT:
                res = self.robot.look_at(args.get("target"),
                                         _as_point(args.get("point")),
                                         args.get("pan_rad"), args.get("tilt_rad"))
            elif a is PrimitiveName.OBSERVE:
                res = self._do_observe(args.get("query", ""))
            elif a is PrimitiveName.SET_GRIPPER:
                res = self.robot.set_gripper(args["state"], float(args.get("effort", 0.4)))
            elif a is PrimitiveName.STOW:
                res = self.robot.stow(bool(args.get("carry", False)))
            elif a is PrimitiveName.DONE:
                res = StepResult(call_id=call.call_id, action=a, outcome=Outcome.SUCCESS,
                                 message=f"planner declared {args.get('status')}: "
                                         f"{args.get('rationale', '')}")
            else:
                res = StepResult(call_id=call.call_id, action=a, outcome=Outcome.REJECTED,
                                 message=f"no dispatch for {a}")
        except KeyError as e:
            res = StepResult(call_id=call.call_id, action=a, outcome=Outcome.REJECTED,
                             message=f"missing required argument {e}")
        except Exception as e:              # a driver fault must not kill the episode
            res = StepResult(call_id=call.call_id, action=a, outcome=Outcome.FAILURE,
                             message=f"the robot raised an error: {type(e).__name__}: {e}")
        res = res.model_copy(update={"call_id": call.call_id, "action": a,
                                     "duration_s": time.time() - t0})
        return res

    # --------------------------------------------------------- supervision
    def run_skill(self, call: PrimitiveCall,
                  pred: StopPredicate | None) -> StepResult:
        """Start a skill, watch it, stop it when the supervisor says so.

        This is the loop the whole local-skill design exists for. A blocking
        call can only be judged once it has finished failing; a running skill
        is judged every tick and cut short the moment something is wrong.

        The supervisor's tiers are described in `monitor.Supervisor`. Note that
        the expensive tiers are pulled, not pushed: `ground` and the
        supervision frame are only produced when a tier actually asks, so a
        skill whose predicate is structural costs nothing to watch.
        """
        params = {k: v for k, v in call.args.items() if k not in ("stop",)}
        skill = self.robot.skill(call.action.value, **params)
        sup = self.monitor.supervisor(call, pred, **self.supervision)

        last_frame_t = 0.0
        try:
            skill.start()
        except Exception as e:
            return StepResult(call_id=call.call_id, action=call.action,
                              outcome=Outcome.FAILURE,
                              message=f"{call.action.value} could not start: {e}")

        decision = None
        while skill.is_running():
            status = skill.poll()

            # Only refresh the supervision frame when the VLM tier is armed and
            # due: capturing an image is not free, and a frame taken while the
            # robot is moving is blurred anyway.
            if sup.enable_vlm and status.elapsed_s - last_frame_t >= sup.vlm_every_s:
                last_frame_t = status.elapsed_s
                try:
                    rgb, _, _, _ = self.robot.capture()
                    self.monitor._supervision_frame = rgb
                except Exception:
                    self.monitor._supervision_frame = None

            d = sup.tick(status, ground=self._grounder_for_supervision)
            if d.stop:
                decision = d
                break
            time.sleep(sup.tick_s)

        if decision is not None:
            report = skill.stop(decision.reason, decision.cause)
        else:
            report = skill.join(timeout=5.0)
        self.monitor._supervision_frame = None

        return self._result_from_report(call, report, sup, decision)

    def _grounder_for_supervision(self, query: str):
        """Ground an object mid-skill. Costs one grounder call; rate limited
        by the supervisor, never called from the free tier."""
        try:
            rgb, depth, cam, T = self.robot.capture()
            return self.grounder.ground(query, rgb, depth, cam, T)
        except Exception:
            return []

    @staticmethod
    def _result_from_report(call: PrimitiveCall, report, sup, decision) -> StepResult:
        """Translate a skill report into a step result.

        The skill reported mechanism only. The outcome assigned here says
        whether the STEP is something the planner can build on -- not whether
        the task succeeded, which the post-step detectors settle against a
        fresh observation.
        """
        cause = report.stopped_by
        if cause in (StopCause.COMPLETED, StopCause.PREDICATE):
            outcome = Outcome.SUCCESS
        elif cause is StopCause.TIMEOUT:
            outcome = Outcome.TIMEOUT
        elif cause in (StopCause.ERROR, StopCause.GUARD):
            outcome = Outcome.FAILURE
        elif cause is StopCause.MONITOR:
            # the supervisor cut it short: never something to build on
            outcome = Outcome.FAILURE
        else:
            outcome = Outcome.ABORTED
        msg = (f"{call.action.value} stopped by {cause.value} after "
               f"{report.elapsed_s:.1f}s: {report.reason}")
        if report.error:
            msg += f" | error: {report.error}"
        return StepResult(
            call_id=call.call_id, action=call.action, outcome=outcome, message=msg,
            stop_reason=cause.value, stopped_by=cause,
            skill_state=report.final_state,
            supervisor_ticks=sup.ticks, vlm_checks=sup.vlm_checks,
            duration_s=report.elapsed_s)

    def _do_observe(self, query: str) -> StepResult:
        rgb, depth, cam, T = self.robot.capture()
        dets = self.grounder.ground(query, rgb, depth, cam, T)
        for d in dets:
            self._detections.append((self._step, d))
        found = [d for d in dets if d.point_3d is not None]
        if not dets:
            msg = (f"no detections for {query!r} in this view. The object may be out "
                   "of frame rather than absent -- change viewpoint before concluding.")
            outcome = Outcome.SUCCESS      # an honest empty result is not a failure
        elif not found:
            msg = (f"found {query!r} in the image but could not compute a 3D point "
                   "(depth invalid at the pixel -- typical for glass and shiny "
                   "surfaces). Move closer or view it against its supporting surface.")
            outcome = Outcome.SUCCESS
        else:
            msg = "; ".join(
                f"{d.label} at ({d.point_3d.x:.2f}, {d.point_3d.y:.2f}, {d.point_3d.z:.2f}) "
                f"conf {d.confidence:.2f}" for d in found)
        return StepResult(call_id="", action=PrimitiveName.OBSERVE,
                          outcome=Outcome.SUCCESS, message=msg, detections=dets)

    # ---------------------------------------------------------------- run
    def run(self, task: str) -> EpisodeRecord:
        ep = EpisodeRecord(task=task, mode="bootstrap" if self.mode == "bootstrap" else "deploy")
        self._step = 0
        self.monitor.reset()
        last: StepResult | None = None  # step result from the previous iteration
        injected = "" # Failure or monitor
        failure_notes: list[str] = []

        while self._step < self.max_steps:
            before = self.observe(last)

            # ---- plan ------------------------------------------------
            try:
                # VLM call
                call = self.planner.propose(before, self.max_steps - self._step, injected)
            except ValueError as e:
                injected = self.planner.reject([str(e)])
                self._step += 1
                continue
            except Exception as e:
                ep.termination_reason = f"planner backend failure: {e}"
                break
            injected = ""

            # ---- validate --------------------------------------------
            problems = validate(call, before)
            if problems:
                res = StepResult(call_id=call.call_id, action=call.action,
                                 outcome=Outcome.REJECTED, message="; ".join(problems))
                ep.calls.append(call)
                ep.results.append(res)
                self.planner.record(call, res, ["rejected_call"])
                injected = self.planner.reject(problems)
                last = res
                self._step += 1
                if self.on_step:
                    self.on_step(call, res)
                continue
            call = finalize(call)

            # ---- terminate? ------------------------------------------
            if call.action is PrimitiveName.DONE:
                res = self.execute(call)
                ep.calls.append(call)
                ep.results.append(res)
                claimed = call.args.get("status") == "success"
                verified, detail = (True, "bootstrap mode: planner verdict accepted") \
                    if self.mode == "bootstrap" and self.monitor.backend is None \
                    else self.monitor.verify_goal(task, self.observe(res))
                if claimed and not verified:
                    # false success: the most damaging error, so it costs a turn
                    self.gm.note_failure("false_success")
                    res.monitor_verdict = (
                        f"REJECTED your success claim. {detail} "
                        "Go and look at the place target before claiming success again.")
                    res.outcome = Outcome.FAILURE
                    last = res
                    injected = res.monitor_verdict
                    self.planner.record(call, res, ["false_success"])
                    failure_notes.append("planner claimed success that the monitor rejected")
                    self._step += 1
                    if self.on_step:
                        self.on_step(call, res)
                    continue
                ep.succeeded = claimed and verified
                ep.termination_reason = (f"planner called done({call.args.get('status')}); "
                                         f"{detail}")
                if self.on_step:
                    self.on_step(call, res)
                break

            # ---- execute ---------------------------------------------
            res = self.execute(call)
            if call.action in (PrimitiveName.NAVIGATE, PrimitiveName.PICK_UP,
                               PrimitiveName.PUT_DOWN):
                # Contact and base motion invalidate every grounded point. Keeping
                # them would let a stale coordinate masquerade as live evidence --
                # which is how a successful grasp gets misreported as an empty one.
                self._detections.clear()
            after = self.observe(res) # Second observation 

            # ---- monitor ---------------------------------------------
            verdicts = self.monitor.check(call, res, before, after)
            advice = self.monitor.advice_for(verdicts)
            stall = self.monitor.note(call, res)
            if stall:
                advice.append(stall)
            if advice:
                res.postcondition_met = False
                res.monitor_verdict = " ".join(advice)
                injected = "MONITOR: " + res.monitor_verdict
                failure_notes.extend(v.detector for v in verdicts if v.detector)
                if res.outcome is Outcome.SUCCESS:
                    # the primitive reported success but reality disagrees
                    res.outcome = Outcome.FAILURE
            else:
                res.postcondition_met = True

            ep.calls.append(call)
            ep.results.append(res)
            note = self.planner.record(
                call, res, [v.detector for v in verdicts if v.detector])
            if note:
                injected = (injected + "\n" + note).strip()
            last = res
            self._step += 1
            if self.on_step:
                self.on_step(call, res)
        else:
            ep.termination_reason = f"step budget of {self.max_steps} exhausted"

        self._consolidate(ep, failure_notes)
        self._write_log(ep)
        return ep

    # -------------------------------------------------------- memory write
    def _consolidate(self, ep: EpisodeRecord, failure_notes: list[str]) -> None:
        """Turn the episode into memory. This is the learning step.

        Successes become a parameterised trace; recurring failure detectors get
        their `seen` counts bumped so the prompt surfaces the ones this robot
        in this house actually hits, rather than a static list.
        """
        for name in set(failure_notes):
            self.gm.note_failure(name)
        if not ep.succeeded:
            return
        trace = TaskMemory.from_episode(ep)
        if trace.steps:
            self.tm.add(trace)
        if self.mode == "bootstrap":
            n_skills = sum(1 for c in ep.calls
                           if c.action in (PrimitiveName.NAVIGATE, PrimitiveName.PICK_UP,
                                           PrimitiveName.PUT_DOWN))
            if n_skills:
                self.gm.add_rule(
                    "global",
                    f"this task was completed in {len(ep.calls)} steps using "
                    f"{n_skills} skill calls; a run much longer than that is "
                    "flailing, not working")

    def _write_log(self, ep: EpisodeRecord) -> None:
        if not self.log_dir:
            return
        (self.log_dir / f"episode_{ep.episode_id}.json").write_text(ep.model_dump_json(indent=2))


# --------------------------------------------------------------------------
def _as_point(v: Any) -> Point3D | None:
    if isinstance(v, dict) and "x" in v and "y" in v:
        return Point3D(x=float(v["x"]), y=float(v["y"]), z=float(v.get("z", 0.0)))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return Point3D(x=float(v[0]), y=float(v[1]), z=float(v[2]) if len(v) > 2 else 0.0)
    return None


def _as_pose(v: Any) -> Pose2D | None:
    if isinstance(v, dict) and "x" in v and "y" in v:
        return Pose2D(x=float(v["x"]), y=float(v["y"]), theta=float(v.get("theta", 0.0)))
    return None
