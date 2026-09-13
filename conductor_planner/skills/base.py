"""Skills: local, startable, stoppable, supervised.

    s_nav = Skill("nav", "kitchen_table", robot=robot)
    s_nav.start()
    ...                       # the supervisor watches observations
    report = s_nav.stop("the doorway is blocked")

A skill is an in-process object, not a service. Three consequences, and the
third is the one that matters:

1. No serialisation, no sockets, no per-call latency floor.
2. A skill can hold references to whatever it needs -- the robot handle, a
   policy, a controller -- instead of passing state through JSON.
3. **It can be stopped from outside while it is running.** That is what makes
   real supervision possible. A blocking call can only be judged after it
   finishes; a running skill can be judged every tick and cut short the moment
   something is wrong.

## Implementing a skill

Subclass `ThreadedSkill` and write three methods:

    class MySkill(ThreadedSkill):
        name = "my_skill"

        def on_start(self):      # set up; called once, on the worker thread
            ...
        def step(self) -> bool:  # one control tick; return False when finished
            ...
        def on_stop(self, cause):  # always called; leave the hardware safe
            ...

`step()` is called in a loop at `rate_hz`. Keep it short and non-blocking --
the loop is also what makes the skill stoppable, so a `step()` that blocks for
five seconds is a skill that cannot be stopped for five seconds.

`poll()` must stay cheap. It is called at control rate by the supervisor and
must only read state the skill already has. Never take a sensor reading there.

## What a skill must never do

Report success. `SkillReport` has no success field on purpose. Report what
happened -- how you stopped, where the hardware ended up -- and let the monitor
decide what it means against a fresh observation. Most importantly: if the
gripper closed on nothing, report the real, near-zero effort. That number is
what the empty-grasp detector reads, and it is the difference between
recovering now and carrying an imaginary glass to the kitchen.
"""
from __future__ import annotations

import abc
import threading
import time
import traceback
from typing import Any, ClassVar

from ..schema import SkillReport, SkillStatus, StopCause

# name -> class, filled by `register`
REGISTRY: dict[str, type["BaseSkill"]] = {}

# Short forms the planner or a person may use.
ALIASES = {
    "nav": "navigate",
    "navigate_to": "navigate",
    "goto": "navigate",
    "pick": "pick_up",
    "grab": "pick_up",
    "place": "put_down",
    "put": "put_down",
}


def canonical(name: str) -> str:
    n = name.strip().lower()
    return ALIASES.get(n, n)


def register(cls: type["BaseSkill"]) -> type["BaseSkill"]:
    REGISTRY[cls.name] = cls
    return cls


class SkillError(RuntimeError):
    pass


# --------------------------------------------------------------------------
class BaseSkill(abc.ABC):
    """The lifecycle contract. `ThreadedSkill` is the implementation you want."""

    name: ClassVar[str] = "base"

    @abc.abstractmethod
    def start(self) -> None:
        """Begin executing. Returns immediately -- this must not block."""

    @abc.abstractmethod
    def stop(self, reason: str = "", cause: StopCause = StopCause.MONITOR) -> SkillReport:
        """Ask the skill to stop, wait for it to wind down, return what happened."""

    @abc.abstractmethod
    def poll(self) -> SkillStatus:
        """A cheap, non-blocking snapshot. Called at control rate."""

    @abc.abstractmethod
    def join(self, timeout: float | None = None) -> SkillReport:
        """Wait for natural completion and return the report."""

    def is_running(self) -> bool:
        return self.poll().running


# --------------------------------------------------------------------------
class ThreadedSkill(BaseSkill):
    """Runs `step()` on a worker thread until it finishes or is stopped.

    The thread exists so that `stop()` can interrupt the skill between control
    ticks. Everything the supervisor needs is published into `_status` under a
    lock, so `poll()` never touches the skill's own working state.
    """

    rate_hz: ClassVar[float] = 20.0
    default_timeout_s: ClassVar[float] = 60.0

    def __init__(self, *, robot: Any = None, timeout_s: float | None = None,
                 **params: Any):
        self.robot = robot
        self.params = params
        self.timeout_s = timeout_s if timeout_s is not None else self.default_timeout_s
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._status = SkillStatus(running=False)
        self._report: SkillReport | None = None
        self._cause = StopCause.COMPLETED
        self._reason = ""
        self._t0 = 0.0

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            raise SkillError(f"{self.name} has already been started; "
                             "construct a new skill for a new attempt")
        self._t0 = time.time()
        self._stop_evt.clear()
        with self._lock:
            self._status = SkillStatus(running=True, phase="starting")
        self._thread = threading.Thread(target=self._run, name=f"skill:{self.name}",
                                        daemon=True)
        self._thread.start()

    def stop(self, reason: str = "", cause: StopCause = StopCause.MONITOR) -> SkillReport:
        self._reason = reason or self._reason
        self._cause = cause
        self._stop_evt.set()
        return self.join(timeout=5.0)

    def join(self, timeout: float | None = None) -> SkillReport:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # The worker is wedged. Say so plainly rather than hanging the
                # episode: a skill that ignores its stop flag is a bug in the
                # skill, and the planner needs to hear about it now.
                return SkillReport(
                    skill=self.name, stopped_by=StopCause.ERROR,
                    reason=(f"{self.name} did not stop within {timeout}s of being "
                            "asked. Its step() is probably blocking. The hardware "
                            "may still be moving."),
                    elapsed_s=time.time() - self._t0,
                    final_state=self.read_state(), error="stop timed out")
        return self._report or SkillReport(
            skill=self.name, stopped_by=self._cause, reason=self._reason,
            elapsed_s=time.time() - self._t0, final_state=self.read_state())

    def poll(self) -> SkillStatus:
        with self._lock:
            return self._status.model_copy(deep=True)

    # ------------------------------------------------------------- worker
    def _run(self) -> None:
        cause, reason, err = StopCause.COMPLETED, "", None
        ticks = 0
        period = 1.0 / max(self.rate_hz, 1e-6)
        try:
            self.on_start()
            self._publish("running", ticks)
            while not self._stop_evt.is_set():
                if time.time() - self._t0 > self.timeout_s:
                    cause = StopCause.TIMEOUT
                    reason = (f"{self.name} ran its full {self.timeout_s:.0f}s without "
                              "reaching its own stopping point")
                    break
                guard = self.guard()
                if guard:
                    cause, reason = StopCause.GUARD, guard
                    break
                more = self.step()
                ticks += 1
                self._publish(self.phase(), ticks)
                if not more:
                    cause = StopCause.COMPLETED
                    reason = f"{self.name} finished"
                    break
                # Waiting on the event rather than sleeping means stop() takes
                # effect within one tick instead of one full period.
                self._stop_evt.wait(period)
            else:
                cause, reason = self._cause, self._reason
        except Exception as e:                       # never kill the episode
            cause = StopCause.ERROR
            err = f"{type(e).__name__}: {e}"
            reason = f"{self.name} raised: {err}"
            traceback.print_exc()
        finally:
            try:
                self.on_stop(cause)
            except Exception as e:
                err = (err or "") + f" | on_stop raised: {e}"
            state = {}
            try:
                state = self.read_state()
            except Exception:
                pass
            self._report = SkillReport(
                skill=self.name, stopped_by=cause, reason=reason,
                elapsed_s=time.time() - self._t0, ticks=ticks,
                final_state=state, error=err)
            with self._lock:
                self._status = SkillStatus(
                    running=False, phase="stopped",
                    elapsed_s=time.time() - self._t0, ticks=ticks,
                    state=state, error=err)

    def _publish(self, phase: str, ticks: int) -> None:
        with self._lock:
            self._status = SkillStatus(
                running=True, phase=phase, elapsed_s=time.time() - self._t0,
                ticks=ticks, progress=self.progress(), state=self.read_state())

    # ------------------------------------------------- implementer hooks
    def on_start(self) -> None:
        """Set up. Runs on the worker thread, before the first step()."""

    @abc.abstractmethod
    def step(self) -> bool:
        """One control tick. Return True to keep going, False when finished."""

    def on_stop(self, cause: StopCause) -> None:
        """Always called, however the skill ended. Leave the hardware safe."""

    def guard(self) -> str | None:
        """Checked every tick before step(). Return a reason to abort.

        This is the skill's own safety check -- runstop, joint limits, a
        force threshold. Distinct from the supervisor, which watches from
        outside and knows about the task; this one knows about the hardware.
        """
        return None

    def read_state(self) -> dict[str, Any]:
        """Hardware state for the supervisor. Must be cheap and non-blocking.

        Include `gripper_aperture` and `gripper_effort` wherever they exist --
        the supervisor's free tier is built on them.
        """
        return {}

    def phase(self) -> str:
        return "running"

    def progress(self) -> float | None:
        return None


# --------------------------------------------------------------------------
class Skill:
    """Facade. `Skill("nav", "kitchen_table", robot=r)` builds the right skill.

    Delegates the whole lifecycle to the registered implementation, so calling
    code never imports a concrete skill class and the set of skills can change
    without touching the executor.
    """

    def __init__(self, name: str, *args: Any, robot: Any = None, **kwargs: Any):
        key = canonical(name)
        cls = REGISTRY.get(key)
        if cls is None:
            raise SkillError(
                f"unknown skill {name!r} (resolved to {key!r}); "
                f"registered skills are {sorted(REGISTRY)}")
        self.name = key
        self._impl = cls(*args, robot=robot, **kwargs)

    # -- lifecycle, straight through ------------------------------------
    def start(self) -> None:
        self._impl.start()

    def stop(self, reason: str = "",
             cause: StopCause = StopCause.MONITOR) -> SkillReport:
        return self._impl.stop(reason, cause)

    def join(self, timeout: float | None = None) -> SkillReport:
        return self._impl.join(timeout)

    def poll(self) -> SkillStatus:
        return self._impl.poll()

    def is_running(self) -> bool:
        return self._impl.is_running()

    @property
    def impl(self) -> BaseSkill:
        return self._impl

    def __repr__(self) -> str:
        st = self._impl.poll()
        return (f"<Skill {self.name} "
                f"{'running' if st.running else 'stopped'} "
                f"{st.elapsed_s:.1f}s phase={st.phase!r}>")


def available() -> list[str]:
    return sorted(REGISTRY)
