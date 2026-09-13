# Bringing the planner up on the Stretch 4

Order matters. Each stage ends with a check you can actually run, and nothing
later works if an earlier check fails. Do not skip to stage 5.

Total time if nothing goes wrong: about a day. Budget two.

Assumes one RTX 5070 Ti (16 GB) and an Anthropic API key. No GPU
work is required to get the planner running.

---

## Stage 0 — where things run

```
┌─ one process, on the robot or a workstation ──────────┐
│  conductor_planner                                    │
│    planner  ──► Anthropic API ────────────────────────┼──► HTTPS
│    supervisor                                         │
│    skills: navigate / pick_up / put_down  (in-process)│
│      └──► ROS 2 ──► stretch_driver, Nav2, cameras     │
│  keys/anthropic.key                                   │
│  no model weights, no GPU needed                      │
└───────────────────────────────────────────────────────┘
```

Skills run **in this process**, not behind a service. That is not only about
latency: it is what lets the supervisor stop a skill mid-motion, which is where
most of the value is. It also means the process needs ROS 2 access, so run it
on the robot, or on a workstation on the same ROS domain while developing.

The planner holds no model weights and spends most of its time waiting on
HTTPS, so it is light either way.

The 5070 Ti is not in this diagram on purpose. See the next section.

---

## Models and the GPU

**The hard constraint: one RTX 5070 Ti, 16 GB VRAM.** That rules out most of
what the robotics-VLM literature assumes, and it is the reason all three model
roles are hosted API calls by default.

What would actually fit, if you ever want a local model:

| model | bf16 weights | verdict on 16 GB |
|---|---|---|
| RoboBrain2.5-4B | ~9 GB | fits, with headroom |
| Qwen3-VL-4B | ~9 GB | fits |
| RoboBrain2.5-8B-NV | ~17 GB | **does not fit** — needs AWQ/FP8 (~9 GB) |
| MiMo-Embodied-7B | ~16 GB | does not fit alongside anything else |

Vision models spike well above their weight footprint on the image tower, so
budget 2–3 GB of headroom over those numbers and cap resolution. And the 5070 Ti
is Blackwell, compute capability 12.0: older PyTorch wheels have no `sm_120`
kernels and fail at load with *"no kernel image is available for execution on
the device"*. You need a build against CUDA 12.8 or newer. Budget an afternoon
for that the first time.

### The three roles

| role | default | why |
|---|---|---|
| planner | `claude-sonnet-4-5` | long-horizon composition and strict JSON. The role that most rewards a strong model; do not economise here first |
| grounder | `claude-sonnet-4-5` | phrase → pixel, called once per `observe`. This is where per-call cost accumulates fastest |
| monitor | `claude-haiku-4-5` | yes/no questions about a scene. Cheap questions deserve a cheap model |

**If the API bill becomes the constraint, move the grounder first and leave the
planner hosted.** The grounder is called far more often, its job (point at the
named thing) is the one small embodied models were specifically post-trained
for, and a 4B model fits your card:

```bash
vllm serve BAAI/RoboBrain2.5-4B --port 8001 --max-model-len 8192
```

then uncomment the `openai_compat` grounder block in
`configs/stretch_glass_task.yaml`. Moving the *planner* to a 4B model is a much
worse trade: long-horizon planning is exactly where small models fall over, and
you will spend the savings debugging plans instead.

### Keys

```
keys/anthropic.key      <- one line, gitignored
keys/openai.key
```

or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment. Full resolution
order in `keys/README.md`. Check what is configured:

```bash
python3 -c "from conductor_planner.backends.keys import status; print(status())"
```

### Watching the bill

The system prompt is ~14k characters and identical on every turn of an episode,
so the Anthropic backend marks it cacheable and you pay for it once:

```python
conductor.planner.backend.cost_report()
# anthropic:claude-sonnet-4-5: 412030 input tokens (94% served from cache), 8210 output
```

If that cached percentage is low, something is mutating the system prompt
between turns. That is a bug, not a tuning parameter — volatile memory belongs
in the per-turn block, which is exactly what `Planner.memory_block` is for.

---

## Stage 1 — the robot is healthy

Before any of this, on the robot:

```bash
python3 ~/stretch_tools/stretch_system_check_fixed.py
stretch_robot_home
stretch_robot_stow
```

Three things the lab has already hit, all of which look like something else:

**`stretch_driver` dies at import.** Symptom: the web teleop UI loads, cameras
stream, nothing moves, and `ros2 node list` has no `/stretch_driver`. Cause:
NumPy 2 in `~/.local` against apt's `python3-transforms3d` 0.4.1. Fix:

```bash
python3 -m pip install --user --no-deps "transforms3d==0.4.2"
```

Do **not** downgrade NumPy. Half the factory stack needs NumPy 2.

**`stretch_robot_stow` fails after ~6 s.** The factory `stow['arm']` is `0.0`,
which is below the arm's hard stop, so the inner wait can never be satisfied.
Add to `~/stretch_user/<robot>/stretch_user_params.yaml`:

```yaml
eoa_wrist_dw4_tool_sg4:
  stow:
    arm: 0.02
```

then **restart `stretch_body_server`** — params load at server startup only, and
a stale server looks exactly like a failed fix. `StretchRobot.check_stow_param()`
refuses to start if this override is missing, so you find out at launch rather
than mid-episode.

**Head camera model mismatch.** If the robot logs `OAK-FFC-3P not found ...
falling back to OAK-D-S2`, the camera TF geometry is wrong. Every 3D point this
package produces inherits that error — a systematic few-centimetre offset in
every grasp target, which is maddening to debug from the planner's end. Fix the
camera descriptor before you tune anything else.

**Check:** `stretch_robot_stow` prints `Stowing routine ended.` and
`ros2 node list` shows `/stretch_driver`.

---

## Stage 2 — map, waypoints, navigation

You already have a map. What you need now are **standing poses**, which is a
stricter thing than free cells.

A waypoint is a pose from which the relevant surface is inside the arm's
workspace. On a Stretch this is not where you would guess: the arm telescopes
out of the **left side** of the base, so standing squarely facing a table puts
the surface out of reach. You want to be alongside, left side toward the table.

Record them by driving there with teleop and reading the pose:

```bash
ros2 run tf2_ros tf2_echo map base_link
```

Then, before you write one down, prove it: with the base parked there, extend
the arm and touch the surface. If you cannot reach it by hand, the planner will
not reach it either, and every failure downstream will look like a grasping
problem when it is a waypoint problem.

Put them in `configs/semantic_map.yaml`. Five is plenty to start:
`living_room`, `coffee_table`, `kitchen`, `kitchen_table`, `charger`.

**Check:** for each waypoint,

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 2.2, y: 0.9}}}}"
```

arrives within ~0.25 m, and the arm can physically touch the target surface from
there.

---

## Stage 3 — the skills

Give the skills team `docs/SKILL_INTERFACE.md`. They own `navigate`, `pick_up`
and `put_down`, as classes in `conductor_planner/skills/stretch.py`.

Until those are written, the mock skills registered by
`conductor_planner/skills/mock.py` stand in — they take real wall-clock time and
are genuinely stoppable, so the supervision loop is exercised properly without
a robot.

```bash
python3 -c "from conductor_planner.skills import available; print(available())"
# ['navigate', 'pick_up', 'put_down']
```

The single most important line in their implementation is in `read_state()`:

```python
"gripper_effort": st.gripper_effort,      # MEASURED, never nominal
```

The supervisor's free tier reads that number ten times a second to tell a real
grasp from a gripper closed on air. A nominal value there disables empty-grasp
detection silently, and nothing downstream will catch it in time.

The second most important: `step()` must not block. Navigation **polls** the
Nav2 goal handle; it never waits on the result. A skill that blocks is a skill
that cannot be stopped, and then all of this is a blocking function call with
extra machinery.

**Check:** start a skill by hand and stop it half-way.

```python
from conductor_planner.skills import Skill
s = Skill("nav", target="kitchen_table", robot=robot)
s.start(); time.sleep(1.0)
print(s.poll())            # running, with a phase
print(s.stop("checking"))  # stopped_by=monitor, and the base has halted
```

If the base is still rolling after `stop()` returns, `on_stop` is not doing its
job and nothing else in the system can save you from that.

## Stage 4 — the models

Covered in "Models and the GPU" above. Nothing to install; put a key in
`keys/anthropic.key` and confirm the prompt assembles:

```bash
python3 -m conductor_planner.cli configs/stretch_glass_task.yaml --dry-run
```

That prints the full system prompt and the retrieved trace. Read it once —
everything the planner knows is in there, and it is generated from
`primitives.py`, so if a primitive looks wrong the fix is in the code, not in a
prompt string.

## Stage 5 — the planner, in increasing danger

**5a. Offline, no robot, no key.**

```bash
python3 scripts/run_mock.py --fresh
python3 tests/run_without_pytest.py     # or: python -m pytest tests/
```

The task should complete in ~11 steps with one injected empty grasp recovered
from — stopped mid-lift by the supervisor, not diagnosed afterwards — and 35
tests should pass. If this fails, stop — the problem is in the
package, not on the robot.

**5b. Real models, mock robot.** Set `robot: {kind: mock}` in the config and
leave the Anthropic model blocks alone. This is the run that tells you whether
the model can actually drive the loop, and it costs a few cents instead of an
hour of robot time. Expect to spend most of your prompt-tuning time here.

**5c. Real robot, sensing only.**

```bash
python3 -m conductor_planner.robot.stretch_ros2 --selftest
```

Prints state, saves a frame, checks the `map ← camera` TF, summarises the scan,
lists the registered skills. Every line must be sane before you let anything move.

**5d. Real robot, analytic primitives only.** Finish the three
`NotImplementedError` methods in `stretch_ros2.py` (`look_at`, `set_gripper`,
`stow`) and the runstop subscription in `get_state`, then drive the robot from a
script through the capability API — look, observe, open and close the gripper,
and `navigate` with `use_skill_for_navigation: false` so Nav2 drives directly.
No skills involved yet.

**5e. The full loop, hand on the runstop.** Start with

```bash
python3 -m conductor_planner.cli configs/stretch_glass_task.yaml --max-steps 12
```

and a step budget too small to finish. You want to watch the first few decisions
and stop, not discover the planner's opinions at step 28.

**5f. Bootstrap the memory.**

```bash
python3 -m conductor_planner.cli configs/stretch_glass_task.yaml --bootstrap
```

Bootstrap mode gives a generous budget and treats failure as cheap, because the
point is to discover a working composition. The first success writes a
parameterised trace into `memory_store/stretch/task_traces.jsonl`. Run it three
or four times; the traces merge and pick up a reliability number. From then on,
deploy runs retrieve that trace and ground it against the live scene.

---

## Wiring the unimplemented methods

They are marked `TODO(you)` in `conductor_planner/robot/stretch_ros2.py`. All are
`FollowJointTrajectory` goals or vendor calls:

| method | joints / call | notes |
|---|---|---|
| `look_at` | `joint_head_pan`, `joint_head_tilt` | the point→angle geometry is already done; you send the trajectory. Tilt is negative-is-down |
| `set_gripper` | `joint_gripper_finger_left` | roughly `+0.17` open, `-0.35` closed on a stock PG4 |
| `stow` | `stretch_body` stow routine, or lift+arm directly | `carry=True` wants a different pose: arm in, lift held at carrying height |
| `get_state` runstop | subscribe `/is_runstopped` | currently hardcoded `False`; the guards depend on it |

Everything else on the robot class — capture, TF, the scan summary, state — is
implemented. The three skills are separate: they live in
`conductor_planner/skills/stretch.py` and belong to the skills team.

Note the exemption in `navigate`: a small `relative` nudge is allowed with the
arm extended, because that is precisely the move the planner needs after a
failed pick. The same exemption lives in `primitives._guard_navigate`. **The two
must agree.** If they drift apart you get a guard that passes and a robot that
refuses, which reads as a planner bug and is not one.

## Supervision, and what it costs

While a skill runs, the supervisor watches it in three tiers. Tuning these is
tuning your API bill, so the defaults are deliberately conservative.

| tier | default rate | cost per check | what it catches |
|---|---|---|---|
| free | 10 Hz | nothing | empty grasp, stalls, timeouts, structural stop predicates, runstop |
| grounding | 1 Hz | one grounder call | "is the object still in view / at the target" |
| model | 0.2 Hz | one VLM call | anything visual — **off by default** |

The model tier arms itself only when the planner declares a `vlm_predicate` on
the call. You can force it on for every skill with `supervision.enable_vlm:
true` in the config, but do the arithmetic first: a 20-second pick at 0.2 Hz is
four vision calls, on every pick, forever. A test asserts the default costs
zero model calls, and if that test starts failing, so does your budget.

Almost everything worth catching mid-skill is in the free tier, because
almost everything worth catching shows up in the gripper and the base pose
before it shows up in a picture. Reach for the model tier when you have a
failure mode you genuinely cannot see in the state — an object knocked over
out of the workspace, say — and then write a free detector for it as soon as
you understand it.

To see what actually ran:

```python
res.supervisor_ticks    # how many times the skill was checked
res.vlm_checks          # how many of those cost a model call
```

---

## Tuning, in the order that pays

1. **Waypoints.** More failures trace back to a badly placed standing pose than
   to anything else. Fix these first and several "grasping problems" disappear.
2. **`REACH_LIMIT_M` and the gripper mapping** in `monitor.py` and
   `stretch_ros2.py`. Measure your robot; the defaults are approximations.
3. **`EMPTY_GRASP_EFFORT`** and **`CONFIRM_SAMPLES`** in `monitor.py`. Pick up a glass, print the effort;
   close on air, print the effort; put the threshold between them. This single
   number decides whether the most important recovery path fires, now during
   the skill rather than after it. Do this with the skills team present — they
   have to report the same number honestly from `read_state()`.

   `CONFIRM_SAMPLES` (default 3) is how many settled samples the supervisor
   wants before it will call a grasp empty. It exists because a gripper ramps
   shut, and on the way down it looks exactly like a miss — judging that on one
   sample aborts good picks mid-close. Raise it if your gripper's aperture
   reporting is noisy; lower it only if you have measured that it is not.
4. **Skill phrasing.** Match what the policies were trained on. Record the
   phrasings that work as success rules — `GlobalMemory.add_rule("pick_up",
   "...")` — and they appear in every future prompt, scoped to `pick_up` calls.
5. **The planner prompt**, last, and only after an A/B against a frontier model
   tells you the prompt is actually the problem.

---

## When it misbehaves

| symptom | look here first |
|---|---|
| planner emits prose instead of JSON | the Anthropic backend forces a tool call, so this should be rare; if it happens, check the repair loop output in `backends/base.py` |
| 401 from the model | `python3 -c "from conductor_planner.backends.keys import status; print(status())"` |
| input token count is not dropping across turns | something volatile got into the system prompt; see `cost_report()` above |
| planner keeps declaring stop predicates that come back false | a skill is reporting optimistically — check `read_state()["gripper_effort"]` against a real measurement |
| a skill will not stop | its `step()` is blocking; `join()` reports this explicitly. Look for a `wait_for_result()` |
| supervisor never fires mid-skill | the skill finishes too fast to observe, or `read_state()` returns nothing useful |
| good picks aborted mid-close | `CONFIRM_SAMPLES`, and check the gripper reports intermediate apertures rather than snapping 1.0 → 0.0 |
| API bill climbing during skills | `res.vlm_checks`; check `supervision.enable_vlm` is not forced on |
| every pick fails just out of reach | waypoint placement, then `REACH_LIMIT_M` |
| "no detections" for something clearly visible | head tilt sign, then camera TF (stage 1), then the pointing output format in `grounding.parse_points` |
| 3D points land below the floor | depth fell through a transparent object; the grounder already refuses these — read the `notes` field |
| planner repeats one call forever | stall detection fires at 3; if it is not firing, `Monitor.stall_limit` |
| episode ends "successful" with the task undone | the monitor's goal check is passing wrongly — test `Monitor.verify_goal` on a saved frame |
| glass ends up somewhere unexpected | check the skill's `stop_reason`; a `timeout` on a put_down means it was still moving when the clock ran out |

Every episode is written to `logs/stretch/episode_*.json` with every call, every
result and every monitor verdict. Read those before re-running anything.
