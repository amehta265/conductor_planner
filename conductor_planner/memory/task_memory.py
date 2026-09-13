"""Task-Specific Memory.

Stores successful primitive compositions as JSONL traces. The trick that makes
a trace reusable rather than a one-off recording is *abstraction*: before
storing, every concrete coordinate is replaced by the perception query that
produced it.

    stored:     {"action": "pick_up", "args": {"target": {"$query": "the drinking glass"}}}
    at deploy:  {"action": "pick_up", "args": {"target": {"x": 2.81, "y": 1.07, "z": 0.74}}}

So the trace says *what to do and in what order*, and the live RGB-D says
*where*. Move the glass to the other end of the sofa and the same trace still
applies. That is the whole idea.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..schema import (Detection, EpisodeRecord, Observation, Outcome,
                      PrimitiveCall, PrimitiveName, Trace, TraceStep)

QUERY_KEY = "$query"

# Argument paths that hold concrete geometry and should be abstracted away.
_GEOMETRIC_ARGS = {"target", "point", "pose", "place_target"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


class TaskMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.traces: list[Trace] = []
        self.load()

    # ---------------------------------------------------------------- io
    def load(self) -> None:
        self.traces = []
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.traces.append(Trace.model_validate_json(line))

    def save(self) -> None:
        self.path.write_text(
            "".join(t.model_dump_json() + "\n" for t in self.traces)
        )

    # --------------------------------------------------------- abstraction
    @staticmethod
    def abstract_call(call: PrimitiveCall, observations: dict[str, str],
                      last_query: str = "") -> TraceStep:
        """Replace concrete geometry with the query that produced it.

        `observations` maps a serialised point back to the phrase that the
        preceding `observe` used. Anything we cannot attribute to a query is
        dropped rather than frozen -- a stale coordinate in memory is worse
        than no coordinate, because the planner will trust it.
        """
        args: dict[str, Any] = {}
        for k, v in call.args.items():
            if k in _GEOMETRIC_ARGS and isinstance(v, dict) and {"x", "y"} <= set(v):
                key = f"{round(v.get('x', 0.0), 2)},{round(v.get('y', 0.0), 2)}"
                q = observations.get(key)
                args[k] = {QUERY_KEY: q or last_query or call.subgoal or "the target object"}
            elif k in _GEOMETRIC_ARGS and isinstance(v, str) and k != "place_target":
                # a waypoint name is symbolic already -- keep it
                args[k] = v
            else:
                args[k] = v
        return TraceStep(action=call.action, args=args,
                         subgoal=call.subgoal, note=call.rationale[:200])

    @classmethod
    def from_episode(cls, ep: EpisodeRecord) -> Trace:
        """Turn a successful episode into a reusable trace.

        Rejected and failed calls are stripped: memory records the path that
        worked, not the flailing. The flailing goes to Global Memory as a
        failure model, which is a different kind of lesson.
        """
        obs_index: dict[str, str] = {}
        for res in ep.results:
            for d in res.detections:
                if d.point_3d:
                    obs_index[f"{round(d.point_3d.x, 2)},{round(d.point_3d.y, 2)}"] = d.query
        steps: list[TraceStep] = []
        by_id = {r.call_id: r for r in ep.results}
        last_query = ""
        for call in ep.calls:
            res = by_id.get(call.call_id)
            if res is None or res.outcome is not Outcome.SUCCESS:
                continue
            if call.action is PrimitiveName.OBSERVE:
                # remember the phrase, so a coordinate we cannot attribute to a
                # specific detection still abstracts to the query that most
                # plausibly produced it rather than to a subgoal label
                last_query = str(call.args.get("query", "")) or last_query
            if call.action is PrimitiveName.DONE:
                continue
            steps.append(cls.abstract_call(call, obs_index, last_query))
        return Trace(task=ep.task, steps=steps)

    # --------------------------------------------------------- retrieval
    def retrieve(self, task: str, k: int = 1, min_reliability: float = 0.0) -> list[Trace]:
        """Lexical overlap retrieval, deliberately.

        An embedding index is one `pip install` away and you should add it when
        the library passes a few hundred traces. Below that it buys nothing but
        a dependency and a model download on the robot.
        """
        want = _tokens(task)
        scored: list[tuple[float, Trace]] = []
        for t in self.traces:
            if t.reliability < min_reliability:
                continue
            have = _tokens(t.task)
            if not have:
                continue
            j = len(want & have) / len(want | have)
            scored.append((j + 0.05 * t.reliability, t))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [t for score, t in scored[:k] if score > 0.15]

    def add(self, trace: Trace, merge_threshold: float = 0.9) -> Trace:
        """Add a trace, merging into a near-identical existing one.

        Without merging, a hundred runs of the same task give you a hundred
        traces and retrieval becomes a coin flip. With it, you get one trace
        with a reliability number attached, which is the thing you actually
        want to condition on.
        """
        for existing in self.traces:
            if existing.task == trace.task and _same_shape(existing, trace, merge_threshold):
                existing.n_success += 1
                existing.n_attempts += 1
                self.save()
                return existing
        self.traces.append(trace)
        self.save()
        return trace

    def record_attempt(self, trace_id: str, success: bool) -> None:
        for t in self.traces:
            if t.trace_id == trace_id:
                t.n_attempts += 1
                if success:
                    t.n_success += 1
                self.save()
                return

    # --------------------------------------------------------- grounding
    @staticmethod
    def pending_queries(trace: Trace) -> list[str]:
        out: list[str] = []
        for step in trace.steps:
            for v in step.args.values():
                if isinstance(v, dict) and QUERY_KEY in v and v[QUERY_KEY]:
                    out.append(v[QUERY_KEY])
        return list(dict.fromkeys(out))

    @staticmethod
    def ground(trace: Trace, detections: Iterable[Detection]) -> tuple[Trace, list[str]]:
        """Bind $query placeholders to live detections.

        Returns the grounded trace and the list of queries that could not be
        bound. Unbound queries are not an error: they tell the planner exactly
        which `observe` calls it still has to make before the trace is usable.
        """
        index: dict[str, Detection] = {}
        for d in detections:
            if d.point_3d is not None:
                index.setdefault(d.query.lower().strip(), d)
        grounded = trace.model_copy(deep=True)
        missing: list[str] = []
        for step in grounded.steps:
            for key, v in list(step.args.items()):
                if isinstance(v, dict) and QUERY_KEY in v:
                    q = (v[QUERY_KEY] or "").lower().strip()
                    hit = index.get(q) or _fuzzy(index, q)
                    if hit and hit.point_3d:
                        step.args[key] = hit.point_3d.model_dump(exclude={"frame"})
                    else:
                        missing.append(v[QUERY_KEY])
        return grounded, list(dict.fromkeys(missing))

    # ------------------------------------------------------------ prompt
    def render(self, task: str, k: int = 1) -> str:
        hits = self.retrieve(task, k=k)
        if not hits:
            return ("No stored trace matches this task. You are composing from "
                    "scratch -- be conservative: observe before you act, and keep "
                    "skill calls tightly scoped.")
        out = []
        for t in hits:
            out.append(
                f"Trace {t.trace_id} for task: {t.task!r}\n"
                f"  succeeded {t.n_success}/{t.n_attempts} times "
                f"(reliability {t.reliability:.2f})\n"
                f"  This is a SUGGESTION, not a script. The scene may differ. "
                f"Any {{\"{QUERY_KEY}\": ...}} placeholder must be resolved with a "
                f"live `observe` before you use it.\n"
            )
            for i, s in enumerate(t.steps, 1):
                out.append(f"  {i:>2}. {s.action.value} {json.dumps(s.args)}"
                           + (f"   # {s.subgoal}" if s.subgoal else ""))
            out.append("")
        return "\n".join(out)


def _same_shape(a: Trace, b: Trace, thresh: float) -> bool:
    sa = [s.action for s in a.steps]
    sb = [s.action for s in b.steps]
    if not sa or not sb:
        return False
    same = sum(1 for i in range(n) if sa[i] == sb[i])
    return same / max(len(sa), len(sb)) >= thresh


def _fuzzy(index: dict[str, Detection], q: str) -> Detection | None:
    want = _tokens(q)
    best, best_score = None, 0.0
    for key, det in index.items():
        have = _tokens(key)
        if not have or not want:
            continue
        score = len(want & have) / len(want | have)
        if score > best_score:
            best, best_score = det, score
    return best if best_score >= 0.4 else None
