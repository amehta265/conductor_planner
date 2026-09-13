"""VLM planner for long-horizon household tasks.

A VLM composes a fixed library of eight primitives; three of them -- navigate,
pick_up, put_down -- are local, startable, stoppable skills that a supervisor
watches while they run;
memory carries what worked and what failed; a monitor decides success from
observation rather than from belief.

No GPU required: the planner, grounder and monitor are hosted API calls.

    from conductor_planner import build_from_config
    conductor = build_from_config("configs/stretch_glass_task.yaml")
    episode = conductor.run("Go to the living room, pick up a glass, and place it on the kitchen table")
"""
from .executor import Conductor
from .grounding import Grounder
from .memory import GlobalMemory, TaskMemory
from .monitor import Monitor
from .planner import Planner
from .config import build_from_config, load_config
from .schema import (EpisodeRecord, Observation, Outcome, PrimitiveCall,
                     PrimitiveName, StepResult, StopPredicate, Trace)

__version__ = "0.1.0"
__all__ = [
    "Conductor", "Planner", "Grounder", "Monitor", "TaskMemory", "GlobalMemory",
    "build_from_config", "load_config",
    "PrimitiveCall", "PrimitiveName", "StepResult", "StopPredicate",
    "Observation", "EpisodeRecord", "Outcome", "Trace",
]
