# Skill contract

**Hand this to whoever is building `navigate`, `pick_up` and `put_down`.**

A skill is a Python object in this process — not a service, not a blocking
function. It gets **started**, **watched while it runs**, and **stopped**.

```python
s_nav = Skill("nav", target="kitchen_table", robot=robot)
s_nav.start()
...                                   # the supervisor watches observations
report = s_nav.stop("a person walked into the doorway")
```

Reference implementations you can read and run today:
`conductor_planner/skills/mock.py`. Skeletons to fill in:
`conductor_planner/skills/stretch.py`.

---

## Why an object and not a function call

Because a blocking call can only be judged **after** it has finished failing.

If `pick_up()` blocks for forty seconds, the first moment anyone can tell that
the gripper closed on air is second forty — after the arm has already lifted
nothing, swung to the kitchen, and opened over the table. A skill that can be
stopped from outside is judged every tick, and that empty grasp is caught in
the half-second between the gripper closing and the arm starting to lift, while
undoing it still costs nothing.

That is the entire reason for this shape. Everything below follows from it.

---

## Writing a skill

```python
from conductor_planner.skills import ThreadedSkill, register
from conductor_planner.schema import StopCause

@register
class StretchPickUp(ThreadedSkill):
    name = "pick_up"                 # what the planner emits
    rate_hz = 20.0                   # how often step() is called
    default_timeout_s = 45.0

    def on_start(self):
        """Set up. Runs once on the worker thread, before the first step()."""

    def step(self) -> bool:
        """One control tick. Return True to continue, False when finished."""

    def on_stop(self, cause: StopCause):
        """Always called, however it ended. Leave the hardware safe."""

    def read_state(self) -> dict:
        """Cheap state for the supervisor. Polled at control rate."""

    def guard(self) -> str | None:
        """Hardware abort, checked every tick. Return a reason, or None."""
```

Construction arguments arrive in `self.params`; `self.robot` is the robot
handle. `@register` makes the name available to `Skill(...)`.

---

## The three rules

Ordered by how much damage breaking them does.

### 1. `step()` must be short and non-blocking

It is called at `rate_hz`, and **the gap between calls is the only moment the
skill can be stopped.** A `step()` that blocks for three seconds is a skill
that ignores the stop signal for three seconds, while the supervisor has
already worked out the grasp is empty.

Concretely, for navigation: send the Nav2 goal in `on_start`, then **poll** the
goal handle in `step()`. Never `wait_for_result()`. A skill that blocks on an
action result is a blocking function with extra steps.

### 2. `read_state()` must be cheap and honest

It is polled at control rate, so read values you already have — never trigger a
sensor read or a computation there.

And report `gripper_effort` **as measured**. The supervisor's free tier is
built on that number:

```python
if aperture <= 0.25 and effort < 0.10:
    # closed on nothing — stop now
```

A nominal or optimistic value there silently disables empty-grasp detection,
and nothing else in the system will catch it in time. If the gripper closed on
air, report the real 0.02.

Report the aperture **as it moves**, too. The supervisor debounces on it: a
gripper ramping shut passes through "aperture below threshold, effort still
zero", which is indistinguishable from an empty grasp in a single sample, so
the check waits until the aperture has stopped changing before judging. If you
snap the reported aperture straight from 1.0 to 0.0 with no intermediate
values, that debounce has nothing to work with.

### 3. Never report success

`SkillReport` has no success field, deliberately. Report **mechanism**: how you
stopped, how long you ran, where the hardware ended up. Whether the task
succeeded is decided by the monitor against a fresh observation.

A skill that grades its own homework produces the false-success failure mode:
the episode ends, the log says "done", and the glass is still on the sofa.

---

## The lifecycle in detail

| method | called by | must |
|---|---|---|
| `start()` | executor | return immediately |
| `step()` | the worker loop, at `rate_hz` | be short; return False when finished |
| `guard()` | the worker loop, before each `step()` | be cheap; return a reason to abort |
| `read_state()` | the supervisor, at `tick_hz` | be cheap and honest |
| `poll()` | the supervisor | (provided) return the published status |
| `stop(reason, cause)` | executor | (provided) set the flag, join, report |
| `on_stop(cause)` | the worker, always | leave the hardware safe |

`on_stop` runs on every path — natural completion, a stop request, a timeout,
an exception. Assume you may be half-way through a motion.

**What `on_stop` should not do:** open the gripper. If the supervisor stopped
you for an empty grasp it is already empty; if it stopped you for anything else
you may be holding the object, and dropping it is worse than stopping with it.
The planner has `set_gripper` for that decision.

---

## What the supervisor does while you run

Described fully in `monitor.Supervisor`. The short version, because it shapes
what you should report:

| tier | rate | cost | reads |
|---|---|---|---|
| free | 10 Hz | nothing | `read_state()`, elapsed time, structural stop predicates, stall detection |
| grounding | ~1 Hz | one grounder call | only when the stop predicate needs an object's position |
| model | ~0.2 Hz | a VLM call | **off unless the planner asked for it** |

So: the more useful your `read_state()`, the less anything expensive has to
run. Gripper aperture and effort, base pose, lift and arm extension cover
almost everything worth catching mid-skill.

`stop()` may arrive at any tick, with a `cause`:

- `PREDICATE` — the declared stop condition was satisfied. This is a *good*
  outcome; you did your job and were released early.
- `MONITOR` — the supervisor saw something wrong.
- `TIMEOUT`, `GUARD`, `PLANNER`, `ERROR` — as named.

---

## Errors

Raise `ValueError` from `on_start` or `step` with a message **written for a
language model to read** — it goes straight into the planner's next prompt.

```python
raise ValueError(
    f"IK unreachable: the target is {d:.2f} m from the base and the arm "
    f"extends to {reach:.2f} m. Move the base closer.")
```

That lets the planner fix its staging. `RuntimeError("solver failed")` does not.

---

## Checklist before you say it's ready

- [ ] `step()` returns in well under one tick period, always
- [ ] navigation **polls** the Nav2 goal; it never waits on a result
- [ ] `stop()` takes effect within one tick, verified with a stopwatch
- [ ] `read_state()` triggers no sensor reads and returns in microseconds
- [ ] a pick that closes on air reports the real near-zero `gripper_effort`
- [ ] aperture is reported continuously as the gripper moves, not snapped
- [ ] `on_stop` leaves the arm safe when called mid-motion, and does not open
      the gripper
- [ ] `navigate` with `relative` works **with the arm extended** and does not
      stow first
- [ ] `pick_up` runs reach → close → **lift**, so an empty grasp is visible for
      a moment before the skill ends
- [ ] `pick_up` and `put_down` never drive the base more than a few
      centimetres; crossing a room is `navigate`'s job
- [ ] nothing you return contains a success field
