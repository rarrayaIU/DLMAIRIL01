# DLMAIRIL01
Reinforcement Learning Course IU
# DQN on Atari Pong — DLMAIRIL01

From-scratch Deep Q-Network trained to play Atari Pong on consumer hardware
(Apple MacBook Air M4, 24 GB unified memory), with a Stable-Baselines3
reference run for comparison.

Companion code for the research essay *Implementing a Deep Q-Network for
Atari Pong on Consumer Hardware: Configuration Diagnosis and Baseline
Comparison* (IU International University of Applied Sciences, DLMAIRIL01).

## Contents

| File | Purpose |
|---|---|
| `dqn_pong_v2.py` | From-scratch DQN: network, replay buffer, training loop |
| `DQN_Control_Run.py` | Stable-Baselines3 reference run (RL Baselines3 Zoo settings) |
| `evaluate.py` | Evaluation of a frozen checkpoint under modified conditions |
| `dqn_pong_checkpoint.pt` | Trained weights, mean100 = +19.4 at 3M agent steps |
| `requirements.txt` | Dependencies |

The initial configuration (v1), which failed to learn, is not included here.
It differed from `dqn_pong_v2.py` in four hyperparameters and the training
length; these are listed in Table 1 of the essay.

## Setup

```bash
python3 -m venv venv          # macOS/Linux
#py -m venv venv              # Windows (or use python)

source venv/bin/activate      # macOS/Linux   
#venv\Scripts\activate        # Windows:
pip install -r requirements.txt
```

On Windows or Linux with an NVIDIA GPU, install the CUDA build of PyTorch
from https://pytorch.org rather than the default CPU wheel. Device selection
in the scripts is automatic: MPS, then CUDA, then CPU.

## Watching the trained agent

No training required — this loads the committed checkpoint:

```bash
python evaluate.py --watch
```

## Evaluation sweep

Reproduces Table 2 of the essay: seven conditions varying starting state,
sticky actions, difficulty and game mode, 15 episodes each.

```bash
python evaluate.py --sweep --episodes 15
```

Results are written to `results/eval_conditions.csv`. Individual conditions
can also be run directly:

```bash
python evaluate.py --episodes 15 --difficulty 2
python evaluate.py --episodes 15 --sticky 0.25
python evaluate.py --episodes 15 --mode 1
```

## Training

```bash
python dqn_pong_v2.py
```

Roughly 6.7 hours for 3M agent steps on an M4 MacBook Air. Progress is
logged to `results/dqn_log.csv`.

The replay buffer allocates two uint8 arrays of shape (200000, 4, 84, 84),
about 10.5 GiB in total. On machines with 16 GB of RAM or less, reduce
`BUFFER_SIZE` in `dqn_pong_v2.py`. Evaluation allocates no buffer and runs
in well under 1 GB.

## Reference run

```bash
python DQN_Control_Run.py
tensorboard --logdir ./dqn_sb3_logs
```

Configured to the RL Baselines3 Zoo tuned Atari hyperparameters, with two
deviations: learning starts at 50,000 rather than 100,000 steps, and
training capped at 1M rather than 10M steps.

## Environment versions

Results reported in the essay were produced with ale-py 0.12.0,
gymnasium 1.3.0, numpy 2.5.1, torch 2.13.0 and stable-baselines3 2.9.0.

## References

Mnih, V., et al. (2015). Human-level control through deep reinforcement
learning. *Nature*, 518, 529–533.

Raffin, A. (2020). *RL Baselines3 Zoo* [Computer software].
https://github.com/DLR-RM/rl-baselines3-zoo

## License

MIT

