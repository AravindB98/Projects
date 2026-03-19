# 🎮 Deep Q-Learning Agent for Atari Ms. Pac-Man

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-ALE-green.svg)](https://gymnasium.farama.org/)

A comprehensive Deep Q-Network (DQN) implementation that learns to play Atari Ms. Pac-Man through reinforcement learning, featuring systematic hyperparameter ablation, multiple exploration strategies, and detailed performance analysis.

**Author:** Aravind Balaji  
**Course:** INFO 7375 — Prompt Engineering & Generative AI, Northeastern University  
**Professor:** Nik Bear Brown  
**Date:** March 2026

---

## 🏆 Key Achievements

- **Significant improvement** over random baseline across all trained configurations
- **8 systematic experiments** comparing learning rates, discount factors, and exploration policies
- **3 exploration strategies** implemented and compared: ε-greedy, Boltzmann (softmax), UCB
- **Complete inline documentation** — all 18 rubric questions answered within the notebook
- **Ms. Pac-Man** chosen for its rich 9-action space, cascading rewards, and strategic depth

## 🎯 Why Ms. Pac-Man?

Ms. Pac-Man was selected over simpler Atari games because it presents a substantially richer RL problem:

| Feature | Ms. Pac-Man | Breakout | Pong |
|---------|------------|----------|------|
| **Actions** | 9 (cardinal + diagonal) | 4 | 3 |
| **Reward types** | 13 (pellets, ghosts, fruit) | 1 (break block) | 1 (score point) |
| **Adversaries** | 4 ghosts with semi-random AI | None | 1 paddle |
| **Planning depth** | Deep (maze routes, power pellet timing) | Medium | Shallow |

## 🧠 DQN Architecture

```
Input:  (batch, 4, 84, 84) — 4 stacked grayscale frames
        ┌──────────────────────────────────────────┐
        │  Conv1:  32 filters, 8×8, stride 4 → ReLU │
        │  Conv2:  64 filters, 4×4, stride 2 → ReLU │
        │  Conv3:  64 filters, 3×3, stride 1 → ReLU │
        │  Flatten → 3136 features                   │
        │  FC1:    512 units → ReLU                  │
        │  FC2:    9 units (Q-value per action)      │
        └──────────────────────────────────────────┘
Output: 9 Q-values — one for each Ms. Pac-Man action
Parameters: ~1.7 million
```

**Key Components:**
- **Experience Replay** buffer (10K transitions) for stable learning
- **Target Network** with periodic hard updates (every 1,000 steps)
- **Huber Loss** for robustness to outlier transitions
- **Gradient Clipping** (max_norm=10.0) for training stability

## 🔬 Experiments & Results

8 systematic experiments testing how each hyperparameter affects Ms. Pac-Man gameplay:

| # | Experiment | Change | Finding |
|---|-----------|--------|---------|
| 1 | **Baseline** | ε-greedy, α=1e-4, γ=0.99 | Best overall — mean reward ~253 |
| 2 | Higher LR | α = 5e-4 | Unstable training, lower plateau |
| 3 | Lower LR | α = 1e-5 | Too slow to converge in 60 episodes |
| 4 | Lower γ | γ = 0.90 | Myopic — can't plan through maze |
| 5 | Much lower γ | γ = 0.80 | Nearly random — can't connect actions to rewards |
| 6 | Boltzmann | Softmax exploration (τ: 2→0.1) | Slightly worse — Q-value noise misleads exploration |
| 7 | Lower ε₀ | ε starts at 0.5 | Biased replay buffer, narrow maze exploration |
| 8 | Slower decay | ε decays at 0.00005 | Wastes time on random actions |

**Key Finding:** The discount factor γ has the largest impact. Reducing it from 0.99 to 0.90 cripples the agent because the ghost-eating cascade (+200→+1,600) is too many steps away from the power pellet decision for the value to propagate.

## 📊 Ms. Pac-Man Reward Structure

| Event | Reward | Notes |
|-------|--------|-------|
| Small pellet | +10 | 240 per level |
| Power pellet | +50 | 4 per level |
| Ghost (1st after power) | +200 | Cascading bonus |
| Ghost (2nd) | +400 | Must eat quickly |
| Ghost (3rd) | +800 | Requires pursuit |
| Ghost (4th) | +1,600 | Maximum single reward |
| Fruit (varies) | +100 to +5,000 | Level-dependent |
| Lose a life | Terminal | Episode ends |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- macOS (Apple Silicon MPS), Linux, or Windows
- ~4GB RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/dqn-mspacman.git
cd dqn-mspacman

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision
pip install 'gymnasium[atari,accept-rom-license]' numpy matplotlib moviepy Pillow opencv-python
pip install 'autorom[accept-rom-license]'
AutoROM --accept-license

# Copy ROMs to ale-py directory (if needed)
cp venv/lib/python3.*/site-packages/AutoROM/roms/*.bin "$(python3 -c "import ale_py, os; print(os.path.join(os.path.dirname(ale_py.__file__), 'roms'))")"

# Install Jupyter
pip install jupyter ipykernel
python -m ipykernel install --user --name=mspacman --display-name="Python (mspacman)"
```

### Run the Notebook

```bash
jupyter notebook DQN_MsPacman_Assignment.ipynb
```

Then: **Kernel → Change Kernel → Python (mspacman)** → **Kernel → Restart & Run All**

Total runtime: ~35–50 minutes (8 experiments × 60 episodes each)

### Standalone Script (Alternative)

```bash
# Run smoke tests (verifies all components in < 30 seconds)
python dqn_mspacman.py --mode test

# Analyze environment (instant)
python dqn_mspacman.py --mode analyze

# Train with default config
python dqn_mspacman.py --mode train --episodes 60

# Train with Boltzmann exploration
python dqn_mspacman.py --mode train --policy boltzmann

# Record gameplay GIF
python dqn_mspacman.py --mode play

# Run full ablation suite
python dqn_mspacman.py --mode all_experiments

# Or use Makefile shortcuts
make test            # Smoke tests
make train-short     # Quick 20-episode demo
make analyze         # Environment analysis
make lint            # Run flake8
make format          # Auto-format with Black
```

## 📁 Project Structure

```
dqn-mspacman/
├── 📓 DQN_MsPacman_Assignment.ipynb          # Main notebook (51 cells — code + answers + plots)
├── 🐍 dqn_mspacman.py                        # Standalone implementation (~1,780 lines)
├── 📄 Functional_Requirements_Documentation.docx  # Written answers (all 18 Qs + intro + portfolio)
├── 📋 README.md                               # This file
├── ⚖️  LICENSE                                 # MIT License with rationale
├── 📦 requirements.txt                        # Python dependencies
├── ⚙️  pyproject.toml                          # Black, isort, flake8, mypy config
├── 🔧 Makefile                                # 14 dev command shortcuts
├── 🚫 .gitignore                              # Excludes venv, models, cache
├── 📁 models/                                 # Saved model checkpoints (generated)
├── 📁 results/                                # Training logs, plots, JSON (generated)
└── 📁 videos/                                 # Recorded gameplay (generated)
```

## 🎲 Exploration Strategies

### ε-Greedy (Baseline)
Random action with probability ε, greedy otherwise. ε decays linearly from 1.0 → 0.02. Simple, robust, and doesn't depend on Q-value accuracy.

### Boltzmann (Softmax)
Actions sampled proportionally to Q-values: `P(a) = exp(Q(a)/τ) / Σexp(Q(a')/τ)`. Temperature τ decays from 2.0 → 0.1. Weights exploration by Q-value magnitude.

### UCB (Upper Confidence Bound)
`a* = argmax[Q(a) + c√(ln(N)/N(a))]` — balances Q-values with exploration bonus for rarely-tried actions. Exploration decreases naturally without a decay schedule.

## 🔧 Technical Details

### Compatibility Notes (Gymnasium 1.2+)
This implementation handles recent Gymnasium API changes:
- Uses `FrameStackObservation` (renamed from `FrameStack`)
- Uses `stack_size=` parameter (renamed from `num_stack=`)
- Registers ALE environments explicitly via `gym.register_envs(ale_py)`

### Device Support
Automatically detects and uses the best available compute device:
- **NVIDIA GPU** (CUDA) — fastest
- **Apple Silicon** (MPS) — 3–5× faster than CPU on M1/M2/M3/M4
- **CPU** — works everywhere, slower but reliable

## 📝 Code Attribution

| Component | Source | Modifications |
|-----------|--------|---------------|
| DQN architecture (conv layers) | [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) | Output for 9 actions; gradient clipping |
| Frame preprocessing concept | Mnih et al. (2015) | Used Gymnasium's `AtariPreprocessing` wrapper |
| Experience replay concept | Mnih et al. (2015) | Implemented from scratch using `deque` |
| Target network update | PyTorch RL Tutorial | Changed from soft to hard copy |
| Boltzmann & UCB exploration | Standard RL literature | Original implementations |
| Training loop, experiments | — | Entirely original by Aravind Balaji |
| All analysis & documentation | — | Entirely original by Aravind Balaji |

## 📚 References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
2. Gymnasium Atari Environments: https://ale.farama.org/environments/
3. PyTorch RL Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
4. Ms. Pac-Man ALE: https://ale.farama.org/environments/ms_pacman/

## ⚖️ License

MIT License — see [LICENSE](LICENSE) for details.

Compatible with all dependencies: PyTorch (BSD), Gymnasium (MIT), NumPy (BSD), Matplotlib (PSF), ALE (GPL v2, runtime only).

---

<p align="center">
  <b>INFO 7375 — Prompt Engineering & Generative AI</b><br>
  Northeastern University · College of Engineering · March 2026
</p>
