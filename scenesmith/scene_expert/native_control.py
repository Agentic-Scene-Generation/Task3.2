"""Opt-in wrappers for reproducible openings and fresh target retry sessions.

Install these in each native worker before building its stage agents. Importing
this module changes no SceneSmith classes, random state, budgets, or tools.
"""

from __future__ import annotations

import random
import threading

from collections.abc import Callable
from functools import wraps
from typing import Any

_OPENING_RANDOM_LOCK = threading.RLock()


def install_opening_randomness(mixin: type, random_module: Any, *, seed: int) -> None:
    """Give each opening-tool instance an independent, explicitly seeded RNG.

    The original methods retain their uniform sampling, validation, and geometry
    rules. Their module's RNG binding is restored even when a method raises.
    Both methods share the instance's stream; other process RNG consumers do not.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("opening seed must be an integer")
    for name in ("_add_door_impl", "_add_window_impl"):
        method = getattr(mixin, name)
        previous_seed = getattr(method, "_sceneexpert_opening_seed", None)
        if previous_seed is not None and previous_seed != seed:
            raise ValueError("opening RNG wrapper already uses a different seed")
    for name in ("_add_door_impl", "_add_window_impl"):
        original = getattr(mixin, name)
        if getattr(original, "_sceneexpert_opening_seed", None) is None:
            setattr(mixin, name, _opening_method(original, random_module, seed))


def _opening_method(
    original: Callable[..., Any], random_module: Any, seed: int
) -> Callable[..., Any]:
    @wraps(original)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        with _OPENING_RANDOM_LOCK:
            rng = self.__dict__.get("_sceneexpert_opening_rng")
            if rng is None:
                rng = random.Random(seed)
                self.__dict__["_sceneexpert_opening_rng"] = rng
            previous = random_module.random
            random_module.random = rng
            try:
                return original(self, *args, **kwargs)
            finally:
                random_module.random = previous

    call._sceneexpert_opening_seed = seed
    return call


def install_target_retry_sessions(agent_class: type) -> None:
    """Isolate planner, designer, and critic history on same-target retries.

    The first setup in a stage attempt keeps the native names. Repeated setups
    get distinct database prefixes without clearing old messages or changing the
    stage attempt, retry count, planner behavior, or designer turn budget.
    """
    original = agent_class._setup_furniture_agents
    if getattr(original, "_sceneexpert_target_retry_sessions", False):
        return
    agent_class._setup_furniture_agents = _target_setup(original)


def _target_setup(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def setup(self: Any, furniture_id: Any, furniture_description: str) -> Any:
        attempts = self.__dict__.setdefault("_sceneexpert_target_setup_attempts", {})
        key = (self._stage_execution_attempt, str(furniture_id))
        number = attempts.get(key, 0) + 1
        attempts[key] = number
        if number == 1:
            return original(self, furniture_id, furniture_description)

        factory = self._create_sessions
        had_override = "_create_sessions" in self.__dict__
        previous_override = self.__dict__.get("_create_sessions")

        def fresh(
            session_prefix: str = "", *, stage_execution_attempt: int | None = None
        ) -> Any:
            return factory(
                session_prefix=f"{session_prefix}target_retry_{number:02d}_",
                stage_execution_attempt=stage_execution_attempt,
            )

        self._create_sessions = fresh
        try:
            return original(self, furniture_id, furniture_description)
        finally:
            if had_override:
                self._create_sessions = previous_override
            else:
                del self.__dict__["_create_sessions"]

    setup._sceneexpert_target_retry_sessions = True
    return setup
