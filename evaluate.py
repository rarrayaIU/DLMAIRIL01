#Roberto Arraya, 2026
"""
Evaluation sweep for trained DQN Pong agent.

Loads a frozen checkpoint and plays under configurable environment
conditions. Does not train, does not modify weights and does
write to checkpoint.

Purpose: test agent robustness, verify whether learned policy is closed-loop that
tracks the ball, or an open-loop action sequence that exploits ALE's
determinism (Machado et al., 2018).

How to run each condition:
-----
Watch one game under training conditions:
    python evaluate.py --watch

Evaluate 30 episodes at the training defaults:
    python evaluate.py --episodes 30

Sticky actions (control-channel noise):
    python evaluate.py --episodes 30 --sticky 0.25

Random no-op starts (initial-state shift):
    python evaluate.py --episodes 30 --noop-max 30

Harder opponent:
    python evaluate.py --episodes 30 --difficulty 1

Run the full sweep and write results/eval_conditions.csv:
    python evaluate.py --sweep --episodes 30

Sanity check that mode/difficulty are actually applied:
    python evaluate.py --check-kwargs
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import ale_py  # noqa: F401  (registers ALE envs)
import gymnasium as gym
import numpy as np
import torch

# Import the network definition ONLY. Training constants are re-declared
# below so this file never depends on, or alters, the training configuration.
from dqn_pong_v2 import QNetwork

ENV_ID = "ALE/Pong-v5"
DEFAULT_CHECKPOINT = Path("dqn_pong_checkpoint.pt") #loads checkpoint from training v2
CSV_PATH = Path("results/eval_conditions.csv")

ACTION_NAMES = ["NOOP", "FIRE", "RIGHT(up)", "LEFT(down)", "RIGHTFIRE", "LEFTFIRE"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_eval_env(
    sticky: float = 0.0,
    noop_max: int = 0,
    mode: int = 0,
    difficulty: int = 0,
    render: bool = False,
) -> gym.Env:
    """Build an evaluation environment.

    Every parameter is passed explicitly, including the values that match
    training, so each condition is self-documenting. ALE exposes setters
    but no getters for mode/difficulty, so defaults cannot be queried.

    Preprocessing must match training exactly: frame_skip 4, grayscale,
    84x84, uint8, stack of 4. Only the arguments under test may vary.
    """
    env = gym.make(
        ENV_ID,
        frameskip=1,                          # wrapper owns skipping
        repeat_action_probability=sticky,
        mode=mode,
        difficulty=difficulty,
        render_mode="human" if render else None,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        screen_size=84,
        noop_max=noop_max,                    # 0 = deterministic start
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

def load_agent(checkpoint: Path, n_actions: int, device: torch.device) -> QNetwork:
    ckpt = torch.load(checkpoint, map_location=device)
    net = QNetwork(n_actions).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"Loaded checkpoint from agent step {ckpt['frame']:,}")
    return net

def run_episodes(
    net: QNetwork,
    env: gym.Env,
    device: torch.device,
    n_episodes: int,
    seed: int = 100,
    verbose: bool = True,
) -> dict:
    """Play n_episodes greedily. Returns summary statistics.

    Fully greedy: epsilon = 0, no exploration, no gradients, no buffer.
    """
    returns, lengths = [], []
    action_counts = np.zeros(env.action_space.n, dtype=int)

    t0 = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)     # distinct seed per episode
        total, steps, done = 0.0, 0, False
        while not done:
            with torch.no_grad():
                q = net(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0))
            action = int(q.argmax().item())
            action_counts[action] += 1
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            steps += 1
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
        if verbose:
            print(f"  episode {ep + 1:>3}/{n_episodes}: {total:>6.1f}  ({steps:,} steps)")

    top_share = action_counts.max() / action_counts.sum()
    return {
        "episodes": n_episodes,
        "mean_return": statistics.mean(returns),
        "std_return": statistics.stdev(returns) if n_episodes > 1 else 0.0,
        "min_return": min(returns),
        "max_return": max(returns),
        "mean_length": statistics.mean(lengths),
        "top_action_share": top_share,
        "seconds": time.time() - t0,
        "returns": returns,
    }

def print_summary(label: str, stats: dict) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  mean return      : {stats['mean_return']:+.2f} "
          f"+/- {stats['std_return']:.2f}")
    print(f"  range            : {stats['min_return']:+.0f} to {stats['max_return']:+.0f}")
    print(f"  mean ep length   : {stats['mean_length']:,.0f} agent steps")
    print(f"  top action share : {stats['top_action_share']:.1%}")
    print(f"  elapsed          : {stats['seconds']:.0f} s")

# ------------------------------- Conditions --------------------------------
# Each entry: (label, kwargs for make_eval_env)
SWEEP_CONDITIONS = [
    ("deterministic no-op 0 sticky 0", dict(sticky=0.0, noop_max=0, mode=0, difficulty=0)),
    ("sticky actions 0.25",            dict(sticky=0.25, noop_max=0, mode=0, difficulty=0)),
    ("no-op starts 1-30",              dict(sticky=0.0, noop_max=30, mode=0, difficulty=0)),
    ("difficulty 1",                   dict(sticky=0.0, noop_max=0, mode=0, difficulty=1)),
    ("difficulty 2",                   dict(sticky=0.0, noop_max=0, mode=0, difficulty=2)),
    ("difficulty 3",                   dict(sticky=0.0, noop_max=0, mode=0, difficulty=3)),
    ("mode 1",                         dict(sticky=0.0, noop_max=0, mode=1, difficulty=0)),
]

def check_kwargs() -> None:
    """Verify that mode and difficulty are actually applied rather than
    silently ignored. Compares random-policy behavior across settings."""
    print("Checking whether mode/difficulty change environment behavior.")
    print("(Random policy, 3 episodes each. Identical numbers across rows")
    print(" would suggest the arguments are being ignored.)\n")
    for label, kwargs in [
        ("mode 0, difficulty 0", dict(mode=0, difficulty=0)),
        ("mode 0, difficulty 1", dict(mode=0, difficulty=1)),
        ("mode 0, difficulty 3", dict(mode=0, difficulty=3)),
        ("mode 1, difficulty 0", dict(mode=1, difficulty=0)),
    ]:
        env = make_eval_env(**kwargs)
        rets, lens = [], []
        for ep in range(3):
            obs, _ = env.reset(seed=ep)
            total, steps, done = 0.0, 0, False
            while not done:
                obs, r, term, trunc, _ = env.step(env.action_space.sample())
                total += r
                steps += 1
                done = term or trunc
            rets.append(total)
            lens.append(steps)
        env.close()
        print(f"  {label:<22} returns {rets}  mean length {statistics.mean(lens):,.0f}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--sticky", type=float, default=0.0)
    p.add_argument("--noop-max", type=int, default=0)
    p.add_argument("--mode", type=int, default=0)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--watch", action="store_true", help="render one episode")
    p.add_argument("--sweep", action="store_true", help="run all conditions")
    p.add_argument("--check-kwargs", action="store_true")
    args = p.parse_args()

    if args.check_kwargs:
        check_kwargs()
        return

    device = get_device()
    print(f"Device: {device}")

    if args.watch:
        env = make_eval_env(
            sticky=args.sticky, noop_max=args.noop_max,
            mode=args.mode, difficulty=args.difficulty, render=True,
        )
        net = load_agent(args.checkpoint, env.action_space.n, device)
        stats = run_episodes(net, env, device, n_episodes=1, seed=args.seed)
        print_summary("watch", stats)
        env.close()
        return

    if args.sweep:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for label, kwargs in SWEEP_CONDITIONS:
            env = make_eval_env(**kwargs)
            net = load_agent(args.checkpoint, env.action_space.n, device)
            stats = run_episodes(
                net, env, device, args.episodes, seed=args.seed, verbose=False
            )
            print_summary(label, stats)
            rows.append({
                "condition": label,
                **{k: v for k, v in kwargs.items()},
                "episodes": stats["episodes"],
                "mean_return": round(stats["mean_return"], 2),
                "std_return": round(stats["std_return"], 2),
                "min_return": stats["min_return"],
                "max_return": stats["max_return"],
                "mean_length": round(stats["mean_length"], 0),
                "top_action_share": round(stats["top_action_share"], 4),
            })
            env.close()

        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {CSV_PATH}")
        return

    # Single condition
    env = make_eval_env(
        sticky=args.sticky, noop_max=args.noop_max,
        mode=args.mode, difficulty=args.difficulty,
    )
    net = load_agent(args.checkpoint, env.action_space.n, device)
    stats = run_episodes(net, env, device, args.episodes, seed=args.seed)
    label = (f"sticky={args.sticky}, noop_max={args.noop_max}, "
             f"mode={args.mode}, difficulty={args.difficulty}")
    print_summary(label, stats)
    env.close()

if __name__ == "__main__":
    main()
# END CODE, RA