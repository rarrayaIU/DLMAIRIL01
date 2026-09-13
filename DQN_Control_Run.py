#Roberto Arraya, 2026
"""
DQN control run using Stable-Baselines3.

Used as diagnostic only to verify platform functionality. 

Note the environment uses PongNoFrameskip-v4, which has no sticky actions, matching the
Mnih et al. (2015) setup. ALE/Pong-v5 defaults to repeat_action_probability=0.25.

Run:
    python dqn_sb3_control.py
    tensorboard --logdir ./dqn_sb3_logs
"""

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)  # registers the v4 IDs with gymnasium

env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=100_000,
    learning_starts=50_000,
    target_update_interval=1_000,   # vs 10,000 in dqn_pong.py -- prime suspect
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,       # decay over 10% of total steps
    exploration_final_eps=0.01,
    learning_rate=1e-4,
    batch_size=32,
    tensorboard_log="./dqn_sb3_logs",
)

model.learn(total_timesteps=1_000_000, log_interval=10)
model.save("dqn_sb3_control")
## END CODE, RA