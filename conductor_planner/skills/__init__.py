"""Local, supervised skills. No network, no servers.

    from conductor_planner.skills import Skill
    s = Skill("nav", "kitchen_table", robot=robot)
    s.start()
    ...
    report = s.stop("arrived")

Importing this package registers the built-in skills.
"""
from .base import (ALIASES, REGISTRY, BaseSkill, Skill, SkillError,
                   ThreadedSkill, available, canonical, register)

# registration side effects. Mock skills always load; the Stretch ones are
# imported on demand so that importing this package never requires ROS.
from . import mock as _mock          # noqa: E402,F401


def load_stretch() -> None:
    """Register the Stretch skills, replacing the mock ones of the same name."""
    from . import stretch as _stretch  # noqa: F401

__all__ = [
    "Skill", "BaseSkill", "ThreadedSkill", "SkillError",
    "REGISTRY", "ALIASES", "register", "canonical", "available", "load_stretch",
]
