"""Evaluate a submitted policy on NesyLink mathematical-logic tasks.

Expected policy interfaces:

- a module-level `act(obs, info) -> int`
- a module-level `policy` object with `.act(obs, info)` or `__call__(obs, info)`
- a `Policy` class with `.act(obs, info)`
- a `make_policy()` function returning any of the above

Example:

    python utils/evaluate_policy.py --policy submissions/student_policy.py
    python utils/evaluate_policy.py --policy submissions.student_policy:make_policy --tasks mathematical_logic/task_3
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env
from nesylink.tasks import list_tasks


DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))

TASK_MILESTONES: dict[str, tuple[str, ...]] = {
    "mathematical_logic/task_3": (
        "monster_killed",
        "key_collected",
    ),
    "mathematical_logic/task_4": (
        "switch_activated",
        "key_collected",
        "door_opened",
        "item_collected",
        "monster_killed",
    ),
}

TASK5_EVENTS = (
    "chest_opened",
    "key_collected",
    "gold_collected",
    "item_collected",
    "agent_healed",
    "button_pressed",
    "room_changed",
    "door_opened",
    "trap_triggered",
    "monster_killed",
    "exit_reached",
    "environment_completed",
    "world_completed",
)


@dataclass
class EpisodeResult:
    task_id: str
    seed: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    success: bool
    terminal_reason: str | None
    event_counts: dict[str, int]
    milestones: dict[str, bool]


def split_policy_spec(spec: str) -> tuple[str, str | None]:
    if ":" not in spec:
        return spec, None
    target, attr = spec.rsplit(":", 1)
    return target, attr or None


def load_module(target: str):
    path = Path(target)
    if path.suffix == ".py" or path.exists():
        module_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not module_path.exists():
            raise FileNotFoundError(f"policy file not found: {module_path}")
        module_name = f"_nesylink_policy_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load policy module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def load_policy(policy_spec: str):
    target, attr = split_policy_spec(policy_spec)
    module = load_module(target)

    candidate_names = (attr,) if attr else ("make_policy", "Policy", "policy", "act")
    for name in candidate_names:
        if name is None or not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if name == "make_policy":
            return candidate()
        if name == "Policy" and isinstance(candidate, type):
            return candidate()
        return candidate

    expected = ", ".join(candidate_names)
    raise AttributeError(f"policy module must expose one of: {expected}")


def reset_policy(policy: Any, *, seed: int, task_id: str) -> None:
    reset = getattr(policy, "reset", None)
    if reset is None:
        return
    try:
        reset(seed=seed, task_id=task_id)
    except TypeError:
        try:
            reset(seed=seed)
        except TypeError:
            reset()


def call_policy(policy: Any, obs: np.ndarray, info: dict[str, Any]) -> int:
    actor: Callable[..., Any]
    if hasattr(policy, "act"):
        actor = policy.act
    elif callable(policy):
        actor = policy
    else:
        raise TypeError("policy must be callable or expose an act(obs, info) method")

    try:
        action = actor(obs, info)
    except TypeError:
        action = actor(obs)

    if isinstance(action, dict):
        action = action.get("action")
    if isinstance(action, (tuple, list)) and action:
        action = action[0]
    return int(np.asarray(action).item())


def event_names(info: dict[str, Any]) -> list[str]:
    names = [
        str(record.get("name"))
        for record in info.get("events", {}).get("records", [])
        if isinstance(record, dict) and record.get("name") is not None
    ]
    game = info.get("game", {})
    if game.get("world_completed", False) or info.get("terminal_reason") == "world_completed":
        names.append("world_completed")
    if game.get("dead", False) or info.get("terminal_reason") == "agent_dead":
        names.append("agent_dead")
    return names


def is_success(info: dict[str, Any], terminated: bool) -> bool:
    return bool(
        info.get("game", {}).get("world_completed", False)
        or info.get("terminal_reason") == "world_completed"
        or (terminated and info.get("reward", {}).get("terminated_reason") == "world_completed")
    )


def milestone_names(task_id: str) -> tuple[str, ...]:
    if task_id == "mathematical_logic/task_5":
        return TASK5_EVENTS
    return TASK_MILESTONES.get(task_id, ())


def run_episode(
    *,
    policy: Any,
    task_id: str,
    seed: int,
    max_steps: int | None,
    render_mode: str | None,
) -> EpisodeResult:
    env_kwargs: dict[str, Any] = {
        "observation_mode": "pixels",
        "render_mode": render_mode,
    }
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    env = make_env(task_id=task_id, **env_kwargs)
    reset_policy(policy, seed=seed, task_id=task_id)

    obs, info = env.reset(seed=seed)
    event_counter: Counter[str] = Counter()
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    try:
        while not (terminated or truncated):
            action = call_policy(policy, obs, info)
            if not env.action_space.contains(action):
                raise ValueError(f"policy returned invalid action {action!r} for {task_id}")
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += float(reward)
            event_counter.update(event_names(info))
    finally:
        env.close()

    milestones = {
        name: event_counter.get(name, 0) > 0
        for name in milestone_names(task_id)
    }
    return EpisodeResult(
        task_id=task_id,
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
        success=is_success(info, terminated),
        terminal_reason=info.get("terminal_reason"),
        event_counts=dict(sorted(event_counter.items())),
        milestones=milestones,
    )


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    by_task: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)

    summary: dict[str, Any] = {}
    for task_id, task_results in sorted(by_task.items()):
        episodes = len(task_results)
        event_totals: Counter[str] = Counter()
        milestone_successes: Counter[str] = Counter()
        for result in task_results:
            event_totals.update(result.event_counts)
            for name, achieved in result.milestones.items():
                if achieved:
                    milestone_successes[name] += 1
        summary[task_id] = {
            "episodes": episodes,
            "success_rate": sum(result.success for result in task_results) / episodes,
            "avg_steps": sum(result.steps for result in task_results) / episodes,
            "avg_reward": sum(result.total_reward for result in task_results) / episodes,
            "milestone_rates": {
                name: milestone_successes[name] / episodes
                for name in milestone_names(task_id)
            },
            "event_totals": dict(sorted(event_totals.items())),
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    for task_id, stats in summary.items():
        print(f"\n{task_id}")
        print(f"  episodes:     {stats['episodes']}")
        print(f"  success_rate: {stats['success_rate']:.3f}")
        print(f"  avg_steps:    {stats['avg_steps']:.1f}")
        print(f"  avg_reward:   {stats['avg_reward']:.3f}")
        if stats["milestone_rates"]:
            print("  milestones:")
            for name, rate in stats["milestone_rates"].items():
                print(f"    {name}: {rate:.3f}")
        if task_id == "mathematical_logic/task_5":
            print("  game_event_totals:")
            for name in TASK5_EVENTS:
                print(f"    {name}: {stats['event_totals'].get(name, 0)}")


def parse_args() -> argparse.Namespace:
    task_ids = [task.task_id for task in list_tasks()]
    parser = argparse.ArgumentParser(description="Evaluate a NesyLink policy submission.")
    parser.add_argument("--policy", required=True, help="Policy module or file, optionally with :attribute")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=task_ids,
        help="Task IDs to evaluate.",
    )
    parser.add_argument("--num-envs", type=int, default=10, help="Number of episodes/env instances per task.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed. Episode seed is seed + episode index.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override task max_steps during evaluation.")
    parser.add_argument("--render-mode", default=None, choices=["rgb_array"], help="Optional render mode.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for detailed JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    policy = load_policy(args.policy)
    results: list[EpisodeResult] = []
    for task_id in args.tasks:
        for episode_index in range(args.num_envs):
            seed = args.seed + episode_index
            result = run_episode(
                policy=policy,
                task_id=task_id,
                seed=seed,
                max_steps=args.max_steps,
                render_mode=args.render_mode,
            )
            results.append(result)
            print(
                f"{task_id} seed={seed} success={result.success} "
                f"steps={result.steps} reward={result.total_reward:.3f}"
            )

    summary = summarize(results)
    print_summary(summary)

    if args.json_out is not None:
        payload = {
            "summary": summary,
            "episodes": [asdict(result) for result in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    # sys.argv = [
    #     "eval.py",  # 脚本名（任意）
    #     "--policy", "agent.py",
    #     "--num-envs", "1",
    #     "--tasks", "mathematical_logic/task_1",
    #     "--seed", "0",
    #     "--json-out",
    #     "None"
    # ]
    main()
