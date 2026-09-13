#!/usr/bin/env python3
"""Run the test suite when pytest is not installed.

`python -m pytest tests/` is the normal way. This exists so the suite is
runnable on the robot, where you will not want to add packages.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Fail(AssertionError):
    pass


def _make_pytest_shim() -> types.ModuleType:
    m = types.ModuleType("pytest")

    def fixture(fn=None, **_kw):
        def wrap(f):
            f.__is_fixture__ = True
            return f
        return wrap(fn) if fn else wrap

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            if t is None:
                raise _Fail(f"expected {self.exc.__name__} but nothing was raised")
            return issubclass(t, self.exc)

    m.fixture = fixture
    m.raises = lambda exc: _Raises(exc)
    m.fail = lambda msg="": (_ for _ in ()).throw(_Fail(msg))
    return m


def main() -> int:
    sys.modules["pytest"] = _make_pytest_shim()
    import test_conductor as T  # noqa: E402

    fixtures = {n: f for n, f in vars(T).items() if getattr(f, "__is_fixture__", False)}
    tests = [(n, f) for n, f in sorted(vars(T).items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        tmp = Path(tempfile.mkdtemp(prefix="ht_"))
        kwargs = {}
        for argname in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            if argname in fixtures:
                kwargs[argname] = fixtures[argname](tmp)
            elif argname == "tmp_path":
                kwargs[argname] = tmp
        try:
            fn(**kwargs)
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
