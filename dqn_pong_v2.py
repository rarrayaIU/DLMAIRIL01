# Roberto Arraya, 2026
"""
DQN for Atari Pong (Gymnasium / ALE), optimized for Apple Silicon (MPS).

Based on Mnih et al. (2015), "Human-level control through deep
reinforcement learning", Nature 518.

VERSION 2 -- revised after the v1 run failed to learn (flat -21.0 at 2M
frames, Q-values collapsed to a degenerate policy). Four changes, all
validated against a Stable-Baselines3 DQN control run that reached +11.9
on the same machine:

  1. TARGET_UPDATE_FREQ  10_000 -> 1_000   (v1 chased a 10x-too-stale target)
  2. TRAIN_START         20_000 -> 50_000  (learn from a fuller buffer)
  3. repeat_action_probability -> 0.0      (ALE/Pong-v5 defaults to 0.25
                                            sticky actions; Mnih et al. had none)
  4. TOTAL_FRAMES        2M -> 3M          (control was still climbing at 1M)

Also adds CSV logging of episode rewards for the paper's figures, and
decouples checkpointing from target syncing.

Install:
    pip install torch gymnasium[atari] ale-py numpy

Run:
    python dqn_pong_v2.py                 # train from scratch
    python dqn_pong_v2.py --resume        # resume from checkpoint
    python dqn_pong_v2.py --watch         # watch trained agent play

Expected: reward leaves -21 around 300-500k frames, crosses 0 around
1-1.5M, plateaus near +18..+21. Budget ~3 hours per million frames.
"""

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path

import ale_py  # noqa: F401  (registers ALE envs with gymnasium)
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Hyperparameters -----------------------------
# Values follow the Nature DQN paper, with a smaller replay buffer to fit
# comfortably in 24 GB unified memory (1M uint8 frames ~ 7 GB; 200k is plenty
# for Pong and leaves headroom).
ENV_ID = "ALE/Pong-v5"
STICKY_ACTIONS = 0.0          # v5 defaults to 0.25; Mnih et al. (2015) used none
TOTAL_FRAMES = 3_000_000
BUFFER_SIZE = 200_000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
TRAIN_START = 50_000          # frames of pure exploration before learning
TRAIN_FREQ = 4                # gradient step every N frames
TARGET_UPDATE_FREQ = 1_000    # copy online net -> target net every N frames
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAMES = 300_000    # linear decay horizon (10% of training)
CHECKPOINT_EVERY = 50_000     # save to disk every N frames (independent of sync)
CHECKPOINT_PATH = Path("dqn_pong_checkpoint.pt")
CSV_PATH = Path("results/dqn_log.csv")
LOG_EVERY_EPISODES = 5


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ------------------------------- Environment -------------------------------
def make_env(render: bool = False) -> gym.Env:
    """Standard Atari preprocessing: grayscale, 84x84, frame-skip 4,
    stack of 4 frames -> observation shape (4, 84, 84), dtype uint8."""
    env = gym.make(
        ENV_ID,
        frameskip=1,  # let the wrapper handle frame skipping
        repeat_action_probability=STICKY_ACTIONS,
        render_mode="human" if render else None,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,   # keep uint8; we normalize inside the network
        screen_size=84,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

# --------------------------------- Network ---------------------------------
class QNetwork(nn.Module):
    """The classic Nature-DQN convolutional architecture."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0  # normalize uint8 frames on the fly
        return self.head(self.conv(x))


# ------------------------------ Replay buffer ------------------------------
class ReplayBuffer:
    """Ring buffer storing uint8 frames to keep memory usage low."""

    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, len(self), size=batch_size)
        to = lambda arr, dtype: torch.as_tensor(arr[idxs], dtype=dtype, device=device)
        return (
            to(self.obs, torch.uint8),
            to(self.actions, torch.int64),
            to(self.rewards, torch.float32),
            to(self.next_obs, torch.uint8),
            to(self.dones, torch.float32),
        )

# -------------------------------- Training ---------------------------------
def epsilon_by_frame(frame: int) -> float:
    frac = min(1.0, frame / EPS_DECAY_FRAMES)
    return EPS_START + frac * (EPS_END - EPS_START)

def train(resume: bool = False):
    device = get_device()
    print(f"Training on device: {device}")

    env = make_env()
    n_actions = env.action_space.n

    online = QNetwork(n_actions).to(device)
    target = QNetwork(n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE, obs_shape=(4, 84, 84))

    start_frame = 0
    episode_rewards: deque = deque(maxlen=100)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_log = not CSV_PATH.exists() or not resume
    csv_file = open(CSV_PATH, "a" if resume else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if new_log:
        csv_writer.writerow(
            ["frame", "episode", "reward", "mean100", "epsilon", "seconds"]
        )
        csv_file.flush()

    if resume and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        online.load_state_dict(ckpt["model"])
        target.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_frame = ckpt["frame"]
        print(f"Resumed from frame {start_frame:,}")

    obs, _ = env.reset(seed=0)
    obs = np.asarray(obs)
    episode_reward = 0.0
    episode_count = 0
    t0 = time.time()

    for frame in range(start_frame, TOTAL_FRAMES):
        # --- act ---
        eps = epsilon_by_frame(frame)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = online(torch.as_tensor(obs, device=device).unsqueeze(0))
                action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs)
        done = terminated or truncated
        buffer.push(obs, action, np.sign(reward), next_obs, done)  # reward clipping
        obs = next_obs
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1
            mean100 = float(np.mean(episode_rewards))
            elapsed = time.time() - t0

            # every episode goes to CSV 
            csv_writer.writerow(
                [frame, episode_count, episode_reward,
                 f"{mean100:.2f}", f"{eps:.4f}", f"{elapsed:.1f}"]
            )
            csv_file.flush()

            if episode_count % LOG_EVERY_EPISODES == 0:
                fps = (frame - start_frame + 1) / elapsed
                print(
                    f"frame {frame:>9,} | episode {episode_count:>5} | "
                    f"reward {episode_reward:>6.1f} | mean100 {mean100:>6.1f} | "
                    f"eps {eps:.3f} | fps {fps:,.0f}"
                )
            obs, _ = env.reset()
            obs = np.asarray(obs)
            episode_reward = 0.0

        # --- learn ---
        if frame >= TRAIN_START and frame % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones = buffer.sample(
                BATCH_SIZE, device
            )
            q_values = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target(next_states).max(dim=1).values
                targets = rewards + GAMMA * (1.0 - dones) * next_q
            loss = F.smooth_l1_loss(q_values, targets)  # Huber loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()

        # --- sync target network ---
        if frame % TARGET_UPDATE_FREQ == 0 and frame > 0:
            target.load_state_dict(online.state_dict())

        # --- checkpoint to disk (independent rhythm) ---
        if frame % CHECKPOINT_EVERY == 0 and frame > 0:
            torch.save(
                {
                    "model": online.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "frame": frame,
                },
                CHECKPOINT_PATH,
            )

    csv_file.close()
    env.close()
    print("Training complete.")

# -------------------------------- Evaluation --------------------------------
# This code was originally used to evaluate the trained agent, 
# but now the evaluation.py script is more flexible and supports a wider range of conditions. 
# The watch() function below allows watching the trained agent play in real-time.
def watch():
    device = get_device()
    env = make_env(render=True)
    net = QNetwork(env.action_space.n).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    obs, _ = env.reset()
    total = 0.0
    while True:
        with torch.no_grad():
            q = net(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0))
        obs, reward, terminated, truncated, _ = env.step(int(q.argmax().item()))
        total += reward
        if terminated or truncated:
            print(f"Episode reward: {total}")
            total = 0.0
            obs, _ = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        watch()
    else:
        train(resume=args.resume)
# END CODE, RA