"""Global Memory: success rules and failure models.

Task-Specific Memory answers "what sequence worked for this task". Global
Memory answers "what do this robot, these skills and this room get wrong, in
general" -- and it transfers across tasks, which the traces do not.

Two halves:

  success rules   prompting and staging strategies observed to work
  failure models  named failure modes, how to detect them, what to do instead

**Empty grasp** and **false success** are seeded because they are the two that
will actually bite you on a Stretch. An empty grasp looks identical to a real
one in the joint state if you only check that the gripper closed. A false
success ends the episode with the glass still on the sofa and the log saying
"done".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..schema import FailureModel, SuccessRule


SEED_FAILURE_MODELS: list[FailureModel] = [
    FailureModel(
        name="empty_grasp",
        scope="pick_up",
        signature=("the gripper reports closed, but effort is near zero, or a fresh "
                   "observe still finds the object at its original location"),
        detector="empty_grasp",
        recovery=("Do not proceed as if you are holding it. Open the gripper "
                  "(set_gripper open), back off with navigate relative {\"dx\": -0.2}, "
                  "observe the object again to get a fresh 3D point, and retry pick_up "
                  "naming the graspable feature -- 'the side wall of the glass', not "
                  "'the glass'."),
    ),
    FailureModel(
        name="false_success",
        scope="done",
        signature=("done(success) is emitted without an immediately preceding observe "
                   "whose detections show the goal state"),
        detector="false_success",
        recovery=("Never call done(success) from belief. look_at the place surface, "
                  "observe for the object there, and only then report success, quoting "
                  "the detection in your rationale."),
    ),
    FailureModel(
        name="skill_timeout",
        scope="skill",
        signature=("a skill ran out its timeout without reaching its own stopping "
                   "point, leaving the scene half-changed"),
        detector="skill_timeout",
        recovery=("Do not reissue the same call with a longer timeout -- that repeats "
                  "whatever was not working. Re-stage first: navigate relative to back "
                  "off, look_at and observe to re-establish where things actually are, "
                  "then a fresh call."),
    ),
    FailureModel(
        name="stop_predicate_false",
        scope="global",
        signature=("the skill returned without an error, but the stop predicate you "
                   "declared for the call is false when checked against the scene"),
        detector="stop_predicate_false",
        recovery=("Trust the check, not the skill's return. The step did not do what "
                  "you expected. Observe to find out the actual state before choosing "
                  "the next call, and do not build on the assumption it worked."),
    ),
    FailureModel(
        name="stop_predicate_unverified",
        scope="global",
        signature="a stop predicate referred to an object that no live detection covers",
        detector="stop_predicate_unverified",
        recovery=("Call observe for that object before or after the step, so the claim "
                  "can be checked. An unverifiable claim is worth as little as a "
                  "false one."),
    ),
    FailureModel(
        name="stopped_by_supervisor",
        scope="skill",
        signature=("the supervisor stopped a skill part-way through, because "
                   "something went wrong while it was running"),
        detector="stopped_by_supervisor",
        recovery=("The skill was interrupted at an arbitrary point, so the scene is "
                  "half-changed and your model of it is stale. Do not retry the same "
                  "call. Observe first to find out where things actually are, then "
                  "decide."),
    ),
    FailureModel(
        name="skill_stalled",
        scope="skill",
        signature=("a skill ran with no change in the hardware state -- pushing "
                   "against something, or stuck"),
        detector="skill_stalled",
        recovery=("Something is in the way, or the target is not where you think it "
                  "is. Back off with navigate relative {\"dx\": -0.2}, observe, and "
                  "re-ground before trying again."),
    ),
    FailureModel(
        name="out_of_reach",
        scope="pick_up",
        signature=("repeated pick or place attempts fail and the target is more than "
                   "about 0.8 m from the base, or far off the robot's side"),
        detector="out_of_reach",
        recovery=("This is a staging problem, not a manipulation problem. The skills "
                  "cannot drive the base across a room. Use navigate with a waypoint or "
                  "an explicit pose closer to the object, or a small relative nudge to "
                  "square up, then observe again before retrying."),
    ),
    FailureModel(
        name="arrived_wrong_place",
        scope="navigate",
        signature=("navigate reported finishing, but the base is nowhere near the "
                   "goal it was given"),
        detector="arrived_wrong_place",
        recovery=("Do not carry on as if you are where you meant to be -- everything "
                  "you plan from here will be wrong for a reason that is hard to see. "
                  "This is usually localisation drift or a blocked path the planner "
                  "gave up on. look_at and observe for something you recognise to work "
                  "out where you actually are, then navigate again. If the second "
                  "attempt also lands short, the route is blocked: try a different "
                  "waypoint rather than the same one."),
    ),
    FailureModel(
        name="navigate_with_arm_extended",
        scope="navigate",
        signature="navigate to a waypoint or pose was called while the arm was not stowed",
        detector="not_stowed",
        recovery=("Always stow() before navigating to a waypoint or pose. If you are "
                  "carrying something, stow(carry=true). Small `relative` nudges are "
                  "exempt -- those are allowed with the arm out."),
    ),
    FailureModel(
        name="object_not_in_frame",
        scope="observe",
        signature="observe returned no detections for a query you expected to match",
        detector="empty_observation",
        recovery=("The object is probably outside the camera's field of view, not "
                  "absent. Change the viewpoint before changing your belief: look_at a "
                  "different region or tilt, or navigate to a nearby waypoint, and "
                  "observe again. Only conclude absence after two viewpoints."),
    ),
]

SEED_SUCCESS_RULES: list[SuccessRule] = [
    SuccessRule(scope="global", rule=(
        "Observe immediately before and immediately after every skill call that "
        "touches an object. Before, to get a fresh 3D point; after, to learn what "
        "actually happened rather than what you intended.")),
    SuccessRule(scope="global", rule=(
        "Declare a stop predicate on every skill call that has a checkable outcome. "
        "It is verified for free against the scene, and it is the difference between "
        "noticing a failure now and noticing it three steps later.")),
    SuccessRule(scope="pick_up", rule=(
        "Name the graspable feature, not the object: 'the side wall of the glass, "
        "above the base' picks up better than 'the glass'.")),
    SuccessRule(scope="navigate", rule=(
        "Prefer named waypoints over raw poses. The waypoints were placed at poses "
        "from which the relevant surface is actually reachable; a geometrically valid "
        "pose often is not.")),
    SuccessRule(scope="skill", rule=(
        "A skill is a local specialist: it sees the current camera frame and nothing "
        "else. Your job is to stage it so that the last thirty centimetres is all "
        "that is left to do.")),
]


class GlobalMemory:
    def __init__(self, dir_path: str | Path, seed: bool = True):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rules_path = self.dir / "success_rules.jsonl"
        self.failures_path = self.dir / "failure_models.jsonl"
        self.rules: list[SuccessRule] = []
        self.failures: list[FailureModel] = []
        self.load()
        if seed and not self.rules and not self.failures:
            for r in SEED_SUCCESS_RULES:
                self.rules.append(r)
            for f in SEED_FAILURE_MODELS:
                self.failures.append(f)
            self.save()

    def load(self) -> None:
        self.rules, self.failures = [], []
        if self.rules_path.exists():
            for line in self.rules_path.read_text().splitlines():
                if line.strip():
                    self.rules.append(SuccessRule.model_validate_json(line))
        if self.failures_path.exists():
            for line in self.failures_path.read_text().splitlines():
                if line.strip():
                    self.failures.append(FailureModel.model_validate_json(line))

    def save(self) -> None:
        self.rules_path.write_text("".join(r.model_dump_json() + "\n" for r in self.rules))
        self.failures_path.write_text("".join(f.model_dump_json() + "\n" for f in self.failures))

    # ------------------------------------------------------------ update
    def add_rule(self, scope: str, rule: str) -> SuccessRule:
        for r in self.rules:
            if r.scope == scope and r.rule.strip().lower() == rule.strip().lower():
                r.evidence += 1
                self.save()
                return r
        r = SuccessRule(scope=scope, rule=rule)
        self.rules.append(r)
        self.save()
        return r

    def add_failure(self, fm: FailureModel) -> FailureModel:
        for f in self.failures:
            if f.name == fm.name and f.scope == fm.scope:
                f.seen += 1
                self.save()
                return f
        self.failures.append(fm)
        self.save()
        return fm

    def note_failure(self, name: str) -> None:
        for f in self.failures:
            if f.name == name:
                f.seen += 1
                self.save()
                return

    # ------------------------------------------------------------ lookup
    def for_scope(self, *scopes: str) -> tuple[list[SuccessRule], list[FailureModel]]:
        want = {"global", *scopes}
        return ([r for r in self.rules if r.scope in want],
                [f for f in self.failures if f.scope in want])

    def by_detector(self, detector: str) -> FailureModel | None:
        for f in self.failures:
            if f.detector == detector:
                return f
        return None

    def render(self, scopes: Iterable[str] = ()) -> str:
        want = {"global", *scopes} if scopes else None
        rules = [r for r in self.rules if want is None or r.scope in want]
        fails = [f for f in self.failures if want is None or f.scope in want]
        out = ["WHAT WORKS (learned from successful runs)"]
        out += [f"  - [{r.scope}] {r.rule}" for r in rules] or ["  (nothing recorded yet)"]
        out += ["", "KNOWN FAILURE MODES (learned from failed runs)"]
        if not fails:
            out.append("  (nothing recorded yet)")
        for f in fails:
            out.append(f"  - {f.name} [{f.scope}], seen {f.seen}x")
            out.append(f"      looks like: {f.signature}")
            out.append(f"      do instead: {f.recovery}")
        return "\n".join(out)
