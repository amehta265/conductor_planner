"""Tests that cover the paths you cannot conveniently reach on hardware."""
from __future__ import annotations

import json
import shutil
import time
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conductor_planner import (GlobalMemory, Grounder, Conductor, Monitor, Planner,
                             TaskMemory)
from conductor_planner.backends import ScriptedBackend, extract_json
from conductor_planner.memory.task_memory import QUERY_KEY
from conductor_planner.primitives import REGISTRY, check_guard, validate_call
from conductor_planner.robot.mock import MockRobot
from conductor_planner.schema import (Outcome, PrimitiveCall, PrimitiveName,
                                    StepResult, StopCause)
from conductor_planner.skills import Skill, available
from conductor_planner.testing import OracleGrounder, OraclePlanner

TASK = "Go to the living room, pick up a glass, and place it on the kitchen table"


@pytest.fixture
def mem(tmp_path):
    return TaskMemory(tmp_path / "t.jsonl"), GlobalMemory(tmp_path / "g")


def build(robot, mem, **kw):
    tm, gm = mem
    kw.setdefault("supervision", {"tick_hz": 50.0})   # mock skills run compressed
    return Conductor(robot, Planner(OraclePlanner(robot), TASK, tm, gm),
                   Grounder(OracleGrounder(robot)), Monitor(OracleGrounder(robot), gm),
                   tm, gm, **kw)


# ------------------------------------------------------------------ parsing
def test_json_extraction_survives_model_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nHope that helps') == {"a": 1}
    assert extract_json('{"a": {"b": "}"}} and then some prose') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no object here at all")


def test_repair_loop_recovers_from_a_bad_first_response():
    b = ScriptedBackend(["I think we should stow first.", '{"action": "stow", "args": {}}'])
    assert b.chat_json([])["action"] == "stow"


# -------------------------------------------------------------- validation
def test_every_primitive_renders_and_validates():
    assert len(REGISTRY) == 8
    for name, spec in REGISTRY.items():
        assert spec.postcondition
        problems = validate_call(name, {a.name: _dummy(spec, a) for a in spec.args})
        assert problems == [], f"{name.value}: {problems}"


def _dummy(spec, a):
    """A minimally valid value for an argument, so every spec can be exercised."""
    if a.enum:
        return a.enum[0]
    if a.name == "stop":
        kind = spec.stop_kinds[0] if spec.stop_kinds else "timeout"
        needs_arg = kind in ("object_in_gripper", "object_at_target",
                             "object_in_view", "vlm_predicate")
        return {"kind": kind, **({"argument": "the glass"} if needs_arg else {})}
    if a.name == "relative":
        return {"dx": -0.1}
    return {"string": "x", "integer": 4, "number": 1.0, "boolean": True}.get(
        a.type, {"x": 0.0, "y": 0.0, "z": 0.0})


def test_oversized_relative_move_is_rejected():
    p = validate_call(PrimitiveName.NAVIGATE, {"relative": {"dx": 3.0}})
    assert p and "too large" in p[0]


def test_stop_predicate_is_scoped_to_the_primitive():
    p = validate_call(PrimitiveName.PICK_UP,
                      {"object": "glass", "stop": {"kind": "base_at_target"}})
    assert p and "does not apply to pick_up" in p[0]


def test_stop_predicate_requiring_an_argument_is_rejected_without_one():
    p = validate_call(PrimitiveName.PICK_UP,
                      {"object": "glass", "stop": {"kind": "object_in_gripper"}})
    assert p and "needs 'argument'" in p[0]


def test_malformed_stop_predicate_is_rejected_readably():
    p = validate_call(PrimitiveName.PICK_UP,
                      {"object": "glass", "stop": {"kind": "teleport"}})
    assert p and "not a valid stop predicate" in p[0]


def test_unknown_argument_is_rejected_with_a_readable_message():
    p = validate_call(PrimitiveName.OBSERVE, {"query": "a glass", "colour": "blue"})
    assert any("'colour' is not an argument" in x for x in p)


# ------------------------------------------------------------------- guards
def _drive(robot, **params):
    """Run a skill to completion, the way the executor would."""
    s = robot.skill(params.pop("name"), **params)
    s.start()
    return s.join(timeout=10.0)


def test_navigating_to_a_waypoint_with_the_arm_out_is_refused():
    r = MockRobot()
    _drive(r, name="navigate", target="coffee_table")
    _drive(r, name="pick_up", object="the drinking glass")   # un-stows the arm
    rep = _drive(r, name="navigate", target="kitchen")
    assert rep.stopped_by is StopCause.ERROR and "stow" in (rep.error or "")


def test_a_small_relative_nudge_is_allowed_with_the_arm_out():
    """Re-staging is exactly the move you need while the arm is extended."""
    r = MockRobot()
    _drive(r, name="navigate", target="coffee_table")
    _drive(r, name="pick_up", object="the drinking glass")
    assert not r.get_state().is_stowed
    assert check_guard(PrimitiveName.NAVIGATE, r.get_state(),
                       {"relative": {"dx": -0.2}}) is None
    assert check_guard(PrimitiveName.NAVIGATE, r.get_state(),
                       {"target": "kitchen"}) is not None
    rep = _drive(r, name="navigate", relative={"dx": -0.2})
    assert rep.stopped_by is StopCause.COMPLETED


def test_stow_refuses_to_drop_what_it_is_carrying():
    r = MockRobot()
    _drive(r, name="navigate", target="coffee_table")
    _drive(r, name="pick_up", object="the drinking glass")
    assert r.stow().outcome is Outcome.FAILURE
    assert r.stow(carry=True).outcome is Outcome.SUCCESS


# ------------------------------------------------------------- end to end
def test_task_completes_in_the_mock_world(mem):
    r = MockRobot(fail_picks=0)
    ep = build(r, mem, max_steps=24).run(TASK)
    assert ep.succeeded, ep.termination_reason
    assert len(ep.calls) <= 14


def test_empty_grasp_is_detected_and_recovered_from(mem):
    r = MockRobot(fail_picks=1)
    ep = build(r, mem, max_steps=30).run(TASK)
    assert ep.succeeded, ep.termination_reason
    picks = [x for x in ep.results if x.action is PrimitiveName.PICK_UP]
    assert len(picks) >= 2, "the first pick should have been caught and retried"
    assert any("empty_grasp" in (x.monitor_verdict or "") for x in picks)


def test_a_false_stop_predicate_is_caught_for_free(mem):
    """The declared stop predicate is checked against reality, not trusted."""
    r = MockRobot(fail_picks=1)
    ep = build(r, mem, max_steps=24).run(TASK)
    picks = [x for x in ep.results if x.action is PrimitiveName.PICK_UP]
    assert picks[0].stop_predicate_met is False
    assert picks[-1].stop_predicate_met is True


def test_a_successful_episode_writes_a_parameterised_trace(mem):
    tm, gm = mem
    r = MockRobot(fail_picks=0)
    ep = build(r, mem, max_steps=24).run(TASK)
    assert ep.succeeded
    assert len(tm.traces) == 1
    trace = tm.traces[0]
    # no frozen coordinates survive abstraction
    for step in trace.steps:
        for v in step.args.values():
            if isinstance(v, dict) and "x" in v:
                pytest.fail(f"a concrete coordinate leaked into memory: {step.action} {v}")
    assert TaskMemory.pending_queries(trace), "the trace should carry at least one query"


def test_repeat_runs_merge_rather_than_multiply(mem):
    tm, _ = mem
    for _ in range(3):
        build(MockRobot(fail_picks=0), mem, max_steps=24).run(TASK)
    assert len(tm.traces) == 1
    assert tm.traces[0].n_success == 3


def test_retrieved_trace_grounds_against_a_moved_object(mem):
    """The point of abstraction: move the glass and the trace still applies."""
    tm, gm = mem
    build(MockRobot(fail_picks=0), mem, max_steps=24).run(TASK)
    trace = tm.traces[0]

    r2 = MockRobot(fail_picks=0)
    glass = next(o for o in r2.world.objects if "glass" in o.name)
    glass.x, glass.y = 2.90, 0.85            # somewhere else on the table
    _drive(r2, name="navigate", target="coffee_table")
    r2.look_at(target="drinking glass")
    r2.look_at(tilt_rad=-0.6)
    rgb, depth, cam, T = r2.capture()
    dets = Grounder(OracleGrounder(r2)).ground(
        "the drinking glass on the coffee table", rgb, depth, cam, T)
    grounded, missing = TaskMemory.ground(trace, dets)
    assert not missing, f"unbound queries: {missing}"
    tgt = next(s.args["target"] for s in grounded.steps
               if s.action is PrimitiveName.PICK_UP)
    assert abs(tgt["x"] - 2.90) < 0.35 and abs(tgt["y"] - 0.85) < 0.35


# ---------------------------------------------------------------- monitor
def test_false_success_claim_is_rejected(mem):
    """A planner that claims success early must not be believed."""
    tm, gm = mem
    r = MockRobot()
    liar = ScriptedBackend(lambda msgs: json.dumps(
        {"action": "done", "args": {"status": "success", "rationale": "I am sure"},
         "rationale": "done"}))
    h = Conductor(r, Planner(liar, TASK, tm, gm), Grounder(OracleGrounder(r)),
                Monitor(OracleGrounder(r), gm), tm, gm, max_steps=3)
    ep = h.run(TASK)
    assert not ep.succeeded
    assert any("REJECTED your success claim" in (x.monitor_verdict or "") for x in ep.results)
    assert len(tm.traces) == 0, "a false success must not enter memory"


def test_stall_detection_fires_on_repetition(mem):
    tm, gm = mem
    r = MockRobot()
    stuck = ScriptedBackend(lambda msgs: json.dumps(
        {"action": "look_at", "args": {"tilt_rad": -0.4}, "rationale": "again"}))
    h = Conductor(r, Planner(stuck, TASK, tm, gm), Grounder(OracleGrounder(r)),
                Monitor(OracleGrounder(r), gm), tm, gm, max_steps=6)
    ep = h.run(TASK)
    assert any("identical" in (x.monitor_verdict or "") for x in ep.results)


def test_out_of_reach_grasp_is_diagnosed_as_staging(mem):
    tm, gm = mem
    r = MockRobot()
    _drive(r, name="navigate", target="living_room")   # far from the coffee table
    reacher = ScriptedBackend(lambda msgs: json.dumps(
        {"action": "pick_up",
         "args": {"object": "the glass", "target": {"x": 2.8, "y": 1.05, "z": 0.74}},
         "rationale": "pick it up from here"}))
    h = Conductor(r, Planner(reacher, TASK, tm, gm), Grounder(OracleGrounder(r)),
                Monitor(OracleGrounder(r), gm), tm, gm, max_steps=2)
    ep = h.run(TASK)
    assert any("out_of_reach" in (x.monitor_verdict or "") or
               "staging problem" in (x.monitor_verdict or "") for x in ep.results)


# --------------------------------------------------------------- grounding
def test_transparent_object_does_not_produce_a_point_below_the_floor():
    """The depth ray passing through glass must not yield a usable coordinate."""
    r = MockRobot()
    _drive(r, name="navigate", target="living_room")   # glass visible but distant
    r.look_at(tilt_rad=-0.2)
    rgb, depth, cam, T = r.capture()
    dets = Grounder(OracleGrounder(r)).ground("the drinking glass", rgb, depth, cam, T)
    for d in dets:
        if d.point_3d is not None:
            assert -0.05 <= d.point_3d.z <= 2.2



# ----------------------------------------------------------------- renames
def test_an_old_primitive_name_is_mapped_not_rejected(mem):
    """A stale trace or a model prior can produce a name that no longer exists."""
    tm, gm = mem
    r = MockRobot()
    old_namer = ScriptedBackend(lambda msgs: json.dumps(
        {"action": "vla_grab", "args": {"object": "the glass"},
         "rationale": "using the name I remember"}))
    p = Planner(old_namer, TASK, tm, gm)
    call = p.propose(_obs(r), 10)
    assert call.action is PrimitiveName.PICK_UP
    assert "now 'pick_up'" in call.rationale


def _obs(robot):
    from conductor_planner.schema import Observation
    rgb, depth, cam, T = robot.capture()
    return Observation(step=0, rgb_path=rgb, state=robot.get_state())


# ------------------------------------------------------------ memory cadence
def test_scoped_memory_changes_between_turns(mem):
    """The per-turn block must actually re-scope, or the split buys nothing."""
    tm, gm = mem
    r = MockRobot()
    p = Planner(ScriptedBackend([]), TASK, tm, gm)
    o = _obs(r)

    p._last_action = PrimitiveName.PICK_UP
    after_pick = p.memory_block(o)
    p._last_action = PrimitiveName.NAVIGATE
    after_nav = p.memory_block(o)

    assert after_pick != after_nav
    assert "graspable feature" in after_pick
    assert "waypoints" in after_nav


def test_system_prompt_is_stable_across_turns(mem):
    """Volatile text in the system prompt would void the prompt cache."""
    tm, gm = mem
    r = MockRobot()
    p = Planner(ScriptedBackend([]), TASK, tm, gm)
    first = p.system
    p._last_action = PrimitiveName.PICK_UP
    p.memory_block(_obs(r))
    p._consecutive_failures = 5
    assert p.system == first


def test_trace_is_re_retrieved_after_repeated_failure(mem):
    tm, gm = mem
    r = MockRobot(fail_picks=0)
    build(r, mem, max_steps=24).run(TASK)          # populate memory

    p = Planner(ScriptedBackend([]), TASK, tm, gm, retrieve_after_failures=2)
    retrievals = p._retrievals
    bad = PrimitiveCall(action=PrimitiveName.OBSERVE, args={"query": "x"},
                        subgoal="find it")
    res = StepResult(call_id=bad.call_id, action=bad.action, outcome=Outcome.FAILURE)
    p._subgoal = "find it"
    p.record(bad, res)
    p.record(bad, res)
    assert p._retrievals > retrievals


def test_trace_cursor_advances_with_successful_steps(mem):
    tm, gm = mem
    build(MockRobot(fail_picks=0), mem, max_steps=24).run(TASK)
    p = Planner(ScriptedBackend([]), TASK, tm, gm)
    assert p.trace is not None and p.trace_cursor == 0
    first = p.trace.steps[0]
    call = PrimitiveCall(action=first.action, args=dict(first.args))
    p.record(call, StepResult(call_id=call.call_id, action=call.action,
                              outcome=Outcome.SUCCESS))
    assert p.trace_cursor == 1



# --------------------------------------------------------------- skills
def test_a_skill_is_observable_while_it_runs():
    r = MockRobot()
    _drive(r, name="navigate", target="living_room")
    s = Skill("nav", robot=r, target="kitchen_table")
    s.start()
    time.sleep(0.1)
    st = s.poll()
    assert st.running and "to go" in st.phase
    s.stop("done looking")
    assert not s.is_running()


def test_a_skill_can_be_stopped_mid_motion():
    """The whole point of local skills: interruptible between control ticks."""
    r = MockRobot()
    _drive(r, name="navigate", target="living_room")
    start_x = r.world.base.x
    s = Skill("nav", robot=r, target="kitchen_table")
    s.start()
    time.sleep(0.1)
    rep = s.stop("supervisor saw a person in the doorway", StopCause.MONITOR)
    assert rep.stopped_by is StopCause.MONITOR
    assert not s.is_running()
    # it moved, but nowhere near the goal -- it really was cut short
    assert r.world.base.x != start_x
    assert abs(r.world.base.x - 5.6) > 1.0


def test_skill_aliases_resolve():
    r = MockRobot()
    assert Skill("nav", robot=r, target="kitchen").name == "navigate"
    assert Skill("grab", robot=r, object="x").name == "pick_up"
    assert Skill("place", robot=r, surface="x").name == "put_down"
    assert "navigate" in available()


def test_an_unknown_skill_names_the_registered_ones():
    r = MockRobot()
    try:
        Skill("teleport", robot=r)
    except Exception as e:
        assert "registered skills are" in str(e)
    else:
        pytest.fail("expected an error for an unknown skill")


# ---------------------------------------------------------- supervision
def _status(**kw):
    from conductor_planner.schema import SkillStatus
    base = {"running": True, "elapsed_s": 1.0, "ticks": 20}
    base.update(kw)
    return SkillStatus(**base)


def test_a_gripper_still_closing_is_not_called_an_empty_grasp(mem):
    """A gripper ramps shut, and on the way down it looks exactly like a miss.

    Judging that on one sample aborts good picks mid-close, and the planner then
    backs off and retries something that was working. Found by running the suite
    under CPU load, which shifted the sampling into that window.
    """
    tm, gm = mem
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.PICK_UP, args={"object": "the glass"})
    sup = mon.supervisor(call, None)
    # closing ramp: aperture falling, effort not yet developed
    for ap in (0.60, 0.40, 0.22, 0.10, 0.02):
        d = sup.tick(_status(state={"gripper_aperture": ap, "gripper_effort": 0.0}))
        assert not d.stop, f"aborted mid-close at aperture {ap}"
    # settled on the object: effort appears, still no abort
    for _ in range(4):
        d = sup.tick(_status(state={"gripper_aperture": 0.0, "gripper_effort": 0.55}))
        assert not d.stop


def test_the_free_tier_stops_a_pick_that_closed_on_air(mem):
    """Deterministic: drive the supervisor directly, no threads in the assertion.

    Thread scheduling belongs in the integration test, not here -- a supervisor
    that is correct should be provably correct without a stopwatch.
    """
    tm, gm = mem
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.PICK_UP, args={"object": "the glass"})
    sup = mon.supervisor(call, None)
    for _ in range(5):
        d = sup.tick(_status(state={"gripper_aperture": 0.0, "gripper_effort": 0.55}))
        assert not d.stop, "a loaded gripper must not be stopped"
    # settled shut with no load, held for the confirmation window
    sup = mon.supervisor(call, None)
    stops = [sup.tick(_status(state={"gripper_aperture": 0.0, "gripper_effort": 0.03}))
             for _ in range(6)]
    fired = [d for d in stops if d.stop]
    assert fired and fired[0].detector == "empty_grasp" and fired[0].cost == "free"
    assert sup.vlm_checks == 0


def test_a_satisfied_stop_predicate_ends_the_skill_early(mem):
    """Deterministic: the predicate releases the skill the moment it holds."""
    tm, gm = mem
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.PICK_UP, args={
        "object": "the glass",
        "stop": {"kind": "object_in_gripper", "argument": "glass"}})
    sup = mon.supervisor(call, call.stop_predicate())
    d = sup.tick(_status(state={"gripper_aperture": 0.8, "gripper_effort": 0.0}))
    assert not d.stop
    d = sup.tick(_status(state={"gripper_aperture": 0.0, "gripper_effort": 0.55}))
    assert d.stop and d.cause is StopCause.PREDICATE and d.cost == "free"


def test_an_empty_grasp_is_caught_one_way_or_the_other(mem):
    """End to end: mid-flight if the supervisor sampled it, post-step otherwise.

    The supervisor SAMPLES, so a phase shorter than its tick period can slip
    past it -- on a loaded machine, or with a very fast skill. That is not a
    hole: the post-step detector is the safety net, and this test asserts the
    net holds by checking the outcome rather than the route.
    """
    r = MockRobot(fail_picks=1)
    ep = build(r, mem, max_steps=30).run(TASK)
    picks = [x for x in ep.results if x.action is PrimitiveName.PICK_UP]
    bad = picks[0]
    assert bad.outcome is not Outcome.SUCCESS, bad.message
    caught_live = bad.stopped_by is StopCause.MONITOR
    caught_after = "empty_grasp" in (bad.monitor_verdict or "")
    assert caught_live or caught_after, bad.message


def test_supervision_costs_no_model_calls_by_default(mem):
    """The expensive tier is opt-in. If this regresses, so does the bill."""
    r = MockRobot(fail_picks=1)
    ep = build(r, mem, max_steps=30).run(TASK)
    assert sum(x.vlm_checks for x in ep.results) == 0
    assert any(x.supervisor_ticks > 0 for x in ep.results), "nothing was supervised"


def test_the_vlm_tier_arms_itself_for_a_vlm_predicate(mem):
    from conductor_planner.monitor import Supervisor
    tm, gm = mem
    r = MockRobot()
    mon = Monitor(OracleGrounder(r), gm)
    structural = PrimitiveCall(action=PrimitiveName.PICK_UP, args={
        "object": "glass", "stop": {"kind": "object_in_gripper", "argument": "glass"}})
    visual = PrimitiveCall(action=PrimitiveName.PICK_UP, args={
        "object": "glass",
        "stop": {"kind": "vlm_predicate", "argument": "is the glass in the gripper?"}})
    assert not mon.supervisor(structural, structural.stop_predicate()).enable_vlm
    assert mon.supervisor(visual, visual.stop_predicate()).enable_vlm


def test_a_stalled_skill_is_detected_without_a_model_call(mem):
    """A skill whose state stops changing is stuck, and that is free to see."""
    from conductor_planner.monitor import Supervisor
    from conductor_planner.schema import SkillStatus
    tm, gm = mem
    r = MockRobot()
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.PUT_DOWN, args={"surface": "table"})
    sup = mon.supervisor(call, None, stall_after_s=1.0)
    frozen = {"gripper_aperture": 0.5, "gripper_effort": 0.2}
    d = sup.tick(SkillStatus(running=True, elapsed_s=1.5, ticks=30, state=frozen))
    assert not d.stop
    d = sup.tick(SkillStatus(running=True, elapsed_s=3.0, ticks=90, state=frozen))
    assert d.stop and d.detector == "skill_stalled" and d.cost == "free"
    assert sup.vlm_checks == 0


def test_a_starved_skill_is_not_mistaken_for_a_stalled_one(mem):
    """Wall-clock time alone is not evidence of a stall.

    On a loaded machine a skill thread can go seconds without being scheduled.
    Killing it then would be wrong, and the resulting failure would look like a
    robot fault rather than a scheduling one.
    """
    from conductor_planner.schema import SkillStatus
    tm, gm = mem
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.PUT_DOWN, args={"surface": "table"})
    sup = mon.supervisor(call, None, stall_after_s=1.0)
    frozen = {"gripper_aperture": 0.5, "gripper_effort": 0.2}
    sup.tick(SkillStatus(running=True, elapsed_s=1.5, ticks=4, state=frozen))
    # lots of wall time, almost no ticks: starved, not stalled
    d = sup.tick(SkillStatus(running=True, elapsed_s=20.0, ticks=5, state=frozen))
    assert not d.stop



# ------------------------------------------------------- arriving wrongly
def test_navigating_to_the_wrong_place_is_detected(mem):
    """Nav2 says 'goal reached', localisation drifted, robot is in the wrong room.

    Nothing downstream catches this directly: the next observe simply fails to
    find the object, and the planner concludes the object is missing rather
    than that it is standing somewhere else.
    """
    from conductor_planner.schema import Observation
    tm, gm = mem
    r = MockRobot()
    _drive(r, name="navigate", target="living_room")
    rep = _drive(r, name="navigate", target="kitchen_table")
    assert "goal_x" in rep.final_state, "the navigate skill must report its goal"

    r.world.base.x, r.world.base.y = 2.0, 0.5          # drift after "arrival"
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.NAVIGATE, args={"target": "kitchen_table"})
    res = StepResult(call_id=call.call_id, action=call.action,
                     outcome=Outcome.SUCCESS, skill_state=rep.final_state)
    obs = Observation(step=1, state=r.get_state())
    fired = mon.check(call, res, obs, obs)
    assert any(v.detector == "arrived_wrong_place" for v in fired)
    advice = " ".join(mon.advice_for(fired))
    assert "where you actually are" in advice          # the recovery, not just the fact


def test_arriving_correctly_does_not_fire(mem):
    from conductor_planner.schema import Observation
    tm, gm = mem
    r = MockRobot()
    rep = _drive(r, name="navigate", target="coffee_table")
    mon = Monitor(None, gm)
    call = PrimitiveCall(action=PrimitiveName.NAVIGATE, args={"target": "coffee_table"})
    res = StepResult(call_id=call.call_id, action=call.action,
                     outcome=Outcome.SUCCESS, skill_state=rep.final_state)
    obs = Observation(step=1, state=r.get_state())
    assert not any(v.detector == "arrived_wrong_place"
                   for v in mon.check(call, res, obs, obs))


def test_base_at_target_is_checkable_for_a_waypoint(mem):
    """Previously this bailed with 'goal was a waypoint name; no pose to compare'."""
    from conductor_planner.monitor import evaluate_stop_predicate
    from conductor_planner.schema import Observation, StopPredicate
    tm, gm = mem
    r = MockRobot()
    rep = _drive(r, name="navigate", target="coffee_table")
    call = PrimitiveCall(action=PrimitiveName.NAVIGATE, args={"target": "coffee_table"})
    res = StepResult(call_id=call.call_id, action=call.action,
                     outcome=Outcome.SUCCESS, skill_state=rep.final_state)
    obs = Observation(step=1, state=r.get_state())
    assert evaluate_stop_predicate(StopPredicate(kind="base_at_target"),
                                   call, res, obs).ok
    r.world.base.x += 3.0
    obs2 = Observation(step=2, state=r.get_state())
    v = evaluate_stop_predicate(StopPredicate(kind="base_at_target"), call, res, obs2)
    assert not v.ok and "finished" in v.detail


def test_the_planner_sees_an_image_every_turn(mem):
    """The planner is a VLM, not an LLM: it gets the camera frame each turn."""
    tm, gm = mem
    r = MockRobot()
    captured = ScriptedBackend(lambda msgs: json.dumps(
        {"action": "look_at", "args": {"tilt_rad": -0.4}, "rationale": "x"}))
    p = Planner(captured, TASK, tm, gm)
    rgb, _, _, _ = r.capture()
    from conductor_planner.schema import Observation
    p.propose(Observation(step=0, rgb_path=rgb, state=r.get_state()), 10)
    sent = captured.calls[-1]
    assert any(m.images for m in sent), "no image reached the planner"
    assert sent[-1].images == [rgb], "the planner got a stale frame"
