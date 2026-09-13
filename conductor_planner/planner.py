"""The agentic planner.

One responsibility: given an observation and the retrieved memory, emit one
`PrimitiveCall`. It does not execute anything, does not talk to the robot, and
does not decide whether the task succeeded. That separation is what lets you
swap the planner model, replay an episode offline against a different model, or
unit-test planning with no robot in the loop.

Two things here are worth reading before changing them.

**Context management.** A long-horizon household task can run 30+ steps. Append
every observation and every image to a growing transcript and you will blow the
context window and pay for it twice. So the transcript keeps a rolling window of
recent turns, carries older history as a compact digest, and attaches only the
most recent image.

**Memory cadence.** Memory is split by how often it changes. The *catalogue* --
every success rule, every failure model, and the retrieved trace -- goes into
the system prompt once per episode, where it is byte-identical across turns and
so is served from the prompt cache. The *scoping* is rebuilt every turn: the
handful of rules and failure modes that bear on what just happened, plus the
planner's position in the trace. That gives per-step relevance without
invalidating the cache, which on a 30-step episode is most of the input bill.
The trace itself is re-retrieved when the subgoal changes or after repeated
failure, because those are the two moments when a different trace might apply.
"""
from __future__ import annotations

import json
from typing import Any

from .backends import ChatMessage, VLMBackend
from .memory import GlobalMemory, TaskMemory
from .primitives import REGISTRY, apply_defaults, check_guard, validate_call
from .prompts import (build_system_prompt, call_json_schema, render_memory_block,
                      render_observation, render_rejection, render_trace_progress)
from .schema import (RENAMED, Observation, PrimitiveCall, PrimitiveName,
                     StepResult, Trace)


class Planner:
    def __init__(
        self,
        backend: VLMBackend,
        task: str,
        task_memory: TaskMemory,
        global_memory: GlobalMemory,
        semantic_map: dict[str, Any] | None = None,
        history_window: int = 6,
        attach_images: int = 1,
        extra_notes: str = "",
        max_tokens: int = 900,
        temperature: float = 0.0,
        scoped_memory: bool = True,
        retrieve_on_subgoal_change: bool = True,
        retrieve_after_failures: int = 3,
    ):
        self.backend = backend
        self.task = task
        self.tm = task_memory
        self.gm = global_memory
        self.history_window = history_window
        self.attach_images = attach_images
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.scoped_memory = scoped_memory
        self.retrieve_on_subgoal_change = retrieve_on_subgoal_change
        self.retrieve_after_failures = retrieve_after_failures

        self.trace: Trace | None = None
        self.trace_cursor = 0
        self._retrievals = 0
        self._subgoal = ""
        self._consecutive_failures = 0
        self._recent_detectors: list[str] = []
        self._last_action: PrimitiveName | None = None

        self._retrieve(task)
        self.system = build_system_prompt(
            task, global_memory.render(), semantic_map, extra_notes
        ) + "\n### MEMORY: A TRACE FROM A SIMILAR PAST TASK\n\n" + self._trace_text
        self.turns: list[ChatMessage] = []
        self.digest: list[str] = []

    # ------------------------------------------------------------ retrieval
    def _retrieve(self, query: str) -> None:
        hits = self.tm.retrieve(query, k=1)
        self.trace = hits[0] if hits else None
        self.trace_cursor = 0
        self._retrievals += 1
        self._trace_text = self.tm.render(query, k=1)

    def maybe_retrieve(self, call: PrimitiveCall) -> str | None:
        """Re-retrieve when the situation has changed enough to warrant it.

        Not every turn: retrieval is a lexical scan, but swapping the trace
        mid-episode resets the planner's sense of where it is, so it should
        happen at the two moments when a different trace plausibly applies --
        the planner moved on to a new subgoal, or the current approach has
        failed repeatedly.
        """
        reason = None
        if (self.retrieve_on_subgoal_change and call.subgoal
                and call.subgoal != self._subgoal):
            reason = f"subgoal changed to {call.subgoal!r}"
            self._subgoal = call.subgoal
        elif (self.retrieve_after_failures
              and self._consecutive_failures >= self.retrieve_after_failures):
            reason = f"{self._consecutive_failures} failures in a row"
            self._consecutive_failures = 0
        if reason is None:
            return None
        before = self.trace.trace_id if self.trace else None
        self._retrieve(f"{self.task} {call.subgoal}".strip())
        after = self.trace.trace_id if self.trace else None
        if after != before:
            return (f"Memory re-retrieved ({reason}): now following trace {after}."
                    if after else
                    f"Memory re-retrieved ({reason}): no trace matches; composing fresh.")
        return None

    # ------------------------------------------------------- per-turn memory
    def memory_block(self, obs: Observation) -> str:
        """The dynamic half of memory. Rebuilt every turn; cheap by design."""
        if not self.scoped_memory:
            return ""
        scopes: set[str] = {"global"}
        if self._last_action is not None:
            scopes.add(self._last_action.value)
            spec = REGISTRY.get(self._last_action)
            if spec:
                scopes.add(spec.group)
        if self.trace and self.trace_cursor < len(self.trace.steps):
            nxt = self.trace.steps[self.trace_cursor].action
            scopes.add(nxt.value)
            spec = REGISTRY.get(nxt)
            if spec:
                scopes.add(spec.group)
        if not obs.detections:
            scopes.add("observe")

        rules, failures = self.gm.for_scope(*scopes)
        # the catalogue is already in the cached system prompt; only narrow here
        rules = [r for r in rules if r.scope != "global"][:4]
        failures = [f for f in failures if f.scope != "global"][:3]
        progress = render_trace_progress(
            self.trace, self.trace_cursor,
            TaskMemory.pending_queries(self.trace) if self.trace else [],
        ) if self.trace else ""
        return render_memory_block(rules, failures, progress, self._recent_detectors)

    # ------------------------------------------------------------------ plan
    def propose(self, obs: Observation, budget_left: int,
                injected: str = "") -> PrimitiveCall:
        """Emit the next call, already validated and defaulted."""
        blocks = [b for b in (injected, self.memory_block(obs)) if b]
        blocks.append(render_observation(obs, budget_left))
        user_text = "\n\n".join(blocks)

        images = [obs.rgb_path] if (obs.rgb_path and self.attach_images) else None
        self.turns.append(ChatMessage("user", user_text, images=images))

        obj = self.backend.chat_json(
            self._assemble(), json_schema=call_json_schema(),
            max_tokens=self.max_tokens, temperature=self.temperature, retries=2,
        )
        self.turns.append(ChatMessage("assistant", _compact(obj)))
        return self._to_call(obj)

    def reject(self, problems: list[str]) -> str:
        return render_rejection(problems)

    # ------------------------------------------------------------------
    # Fixed lag smoothing
    def _assemble(self) -> list[ChatMessage]:
        msgs = [ChatMessage("system", self.system)]
        if self.digest:
            msgs.append(ChatMessage(
                "user",
                "### EARLIER IN THIS EPISODE (condensed)\n" + "\n".join(self.digest[-25:]),
            ))
        window = list(self.turns[-(self.history_window * 2):])
        seen = 0
        for i in range(len(window) - 1, -1, -1):
            if window[i].images:
                seen += 1
                if seen > self.attach_images:
                    window[i] = ChatMessage(
                        window[i].role,
                        window[i].text + "\n(image omitted to save context)",
                        images=None)
        msgs.extend(window)
        return msgs

    def record(self, call: PrimitiveCall, result: StepResult,
               detectors: list[str] | None = None) -> str | None:
        """Fold a completed step into the digest, the trace cursor and scoping.

        Returns a note to inject next turn if memory was re-retrieved.
        """
        self.digest.append(
            f"  step {len(self.digest) + 1}: {call.action.value}"
            f"({_short_args(call.args)}) -> {result.outcome.value}"
            f"{': ' + result.message[:110] if result.message else ''}"
        )
        self._last_action = call.action
        self._recent_detectors = list(detectors or [])
        if result.ok:
            self._consecutive_failures = 0
            if (self.trace and self.trace_cursor < len(self.trace.steps)
                    and self.trace.steps[self.trace_cursor].action is call.action):
                self.trace_cursor += 1
        else:
            self._consecutive_failures += 1

        if len(self.turns) > self.history_window * 2:
            self.turns = self.turns[-(self.history_window * 2):]
        return self.maybe_retrieve(call)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_call(obj: dict[str, Any]) -> PrimitiveCall:
        raw = str(obj.get("action", "")).strip().lower()
        note = ""
        if raw in RENAMED:
            # A stale memory trace or the model's own priors can produce a name
            # that used to exist. Mapping it and saying so beats a hard failure,
            # which would burn a turn teaching the model something we already know.
            note = f"(you said '{raw}'; that primitive is now '{RENAMED[raw]}') "
            raw = RENAMED[raw]
        try:
            action = PrimitiveName(raw)
        except ValueError:
            raise ValueError(
                f"'{raw}' is not a primitive. Valid actions: "
                f"{[p.value for p in PrimitiveName]}") from None
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("'args' must be a JSON object")
        return PrimitiveCall(
            action=action, args=args,
            rationale=(note + str(obj.get("rationale", "")))[:1000],
            expect=str(obj.get("expect", ""))[:500],
            subgoal=str(obj.get("subgoal", ""))[:200],
        )


def validate(call: PrimitiveCall, obs: Observation) -> list[str]:
    """Full pre-execution validation: schema, then embodiment guard."""
    problems = validate_call(call.action, call.args)
    if problems:
        return problems
    guard = check_guard(call.action, obs.state, call.args)
    return [guard] if guard else []


def finalize(call: PrimitiveCall) -> PrimitiveCall:
    return call.model_copy(update={"args": apply_defaults(call.action, call.args)})


def _compact(obj: dict[str, Any]) -> str:
    return json.dumps({k: obj.get(k) for k in ("action", "args", "subgoal") if k in obj})


def _short_args(args: dict[str, Any]) -> str:
    bits = []
    for k, v in args.items():
        if isinstance(v, dict) and "x" in v:
            bits.append(f"{k}=({v['x']:.2f},{v['y']:.2f})")
        elif isinstance(v, str):
            bits.append(f"{k}={v[:34]!r}")
        else:
            bits.append(f"{k}={v}")
    return ", ".join(bits)[:180]
