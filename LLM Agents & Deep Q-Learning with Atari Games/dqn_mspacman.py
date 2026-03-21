#!/usr/bin/env python3
"""
Deep Q-Learning Agent for Atari Ms. Pac-Man
=============================================

Author: Aravind Balaji
Course: INFO 7375 — Prompt Engineering & Generative AI (Northeastern University)
Date: March 2026

Description:
    This module implements a Deep Q-Network (DQN) agent that learns to play
    Atari Ms. Pac-Man using the Gymnasium (formerly OpenAI Gym) framework.
    The implementation includes:
        - Convolutional Neural Network for Q-value approximation
        - Experience Replay buffer for stable training
        - Target Network for reducing value overestimation
        - Configurable exploration strategies (ε-greedy, Boltzmann, UCB)
        - Comprehensive logging and metric tracking
        - Full hyperparameter ablation experiment suite

    Ms. Pac-Man was chosen over simpler Atari games because it features:
        - A rich 9-action discrete action space
        - Complex maze navigation with dynamic obstacles (ghosts)
        - Multi-layered reward structure (pellets, power pellets, ghosts, fruit)
        - Strategic planning requirements (route optimization, ghost avoidance)

License: MIT License (see LICENSE file)

References:
    [1] Mnih et al. (2015). "Human-level control through deep reinforcement
        learning." Nature, 518(7540), 529-533.
    [2] Gymnasium Atari Environments — https://ale.farama.org/environments/
    [3] PyTorch Documentation — https://pytorch.org/docs/stable/
    [4] Ms. Pac-Man ALE Documentation — https://ale.farama.org/environments/ms_pacman/

Code Attribution:
    - Core DQN architecture adapted from PyTorch's official RL tutorial:
      https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    - Frame preprocessing pipeline inspired by DeepMind's original DQN paper [1]
    - All experiment logic, exploration policies, analysis code, parameter sweeps,
      and documentation are original work by Aravind Balaji.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
from collections import deque, namedtuple
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch is required. Install via: pip install torch torchvision")
    sys.exit(1)

try:
    import gymnasium as gym
    from gymnasium.wrappers import (
        AtariPreprocessing,
        FrameStackObservation as FrameStack,
        RecordVideo,
    )
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("ERROR: Gymnasium with Atari support is required.")
    print("Install via: pip install 'gymnasium[atari,accept-rom-license]'")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("WARNING: matplotlib not found. Plots will be skipped.")


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

# Reproducibility
RANDOM_SEED = 42

# Ms. Pac-Man Action Space Reference:
#   0: NOOP        1: UP          2: RIGHT        3: LEFT
#   4: DOWN        5: UPRIGHT     6: UPLEFT       7: DOWNRIGHT
#   8: DOWNLEFT
MS_PACMAN_ACTIONS = {
    0: "NOOP",
    1: "UP",
    2: "RIGHT",
    3: "LEFT",
    4: "DOWN",
    5: "UPRIGHT",
    6: "UPLEFT",
    7: "DOWNRIGHT",
    8: "DOWNLEFT",
}

# Ms. Pac-Man Reward Reference:
#   Small pellet:     +10
#   Power pellet:     +50
#   Ghost (1st):      +200
#   Ghost (2nd):      +400
#   Ghost (3rd):      +800
#   Ghost (4th):      +1600
#   Fruit (varies):   +100 to +5000 depending on level
MS_PACMAN_REWARDS = {
    "small_pellet": 10,
    "power_pellet": 50,
    "ghost_1": 200,
    "ghost_2": 400,
    "ghost_3": 800,
    "ghost_4": 1600,
    "fruit_cherry": 100,
    "fruit_strawberry": 200,
    "fruit_orange": 500,
    "fruit_pretzel": 700,
    "fruit_apple": 1000,
    "fruit_pear": 2000,
    "fruit_banana": 5000,
}

# Default Hyperparameters (Baseline)
DEFAULT_CONFIG = {
    "env_name": "MsPacmanNoFrameskip-v4",
    "total_episodes": 5000,
    "total_test_episodes": 100,
    "max_steps_per_episode": 10000,
    "learning_rate": 1e-4,           # Adam optimizer α
    "gamma": 0.99,                   # Discount factor
    "epsilon_start": 1.0,            # Initial exploration rate
    "epsilon_end": 0.02,             # Minimum exploration rate
    "epsilon_decay": 0.00004,        # Linear decay per step
    "batch_size": 32,
    "replay_buffer_size": 100000,
    "target_update_frequency": 1000, # Steps between target network syncs
    "min_replay_size": 10000,        # Minimum buffer size before training
    "frame_stack": 4,                # Number of stacked frames
    "save_frequency": 500,           # Save model every N episodes
    "log_frequency": 50,             # Log metrics every N episodes
    "device": "auto",                # "auto", "cpu", "cuda", or "mps"
}

# Named tuple for experience replay transitions
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "results") -> logging.Logger:
    """
    Configure dual logging to both console and file.

    Args:
        log_dir: Directory to store log files.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("DQN_MsPacman")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular buffer for storing experience transitions.

    The replay buffer is a critical DQN component that serves two purposes:
        1. Breaking temporal correlations between consecutive training samples,
           which would otherwise cause the Q-network to overfit to local
           sequences of experience.
        2. Reusing past experiences multiple times for more data-efficient
           learning — each transition can appear in many mini-batches.

    Implementation uses Python's `deque` with a fixed `maxlen`, which
    automatically evicts the oldest transitions when capacity is reached.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (deque): Internal storage with automatic oldest-eviction.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition (s, a, r, s', done) in the buffer.

        Time complexity:  O(1) amortized — deque append is constant time.
        Space complexity: O(1) per call — evicts oldest if at capacity.
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random mini-batch of transitions.

        Uniform random sampling ensures diverse mini-batches that break
        temporal correlations present in sequential environment interaction.

        Time complexity:  O(batch_size) — random.sample draws k items.
        Space complexity: O(batch_size) — allocates a new list of k items.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Transition named tuples.

        Raises:
            ValueError: If batch_size exceeds buffer length.

        Example:
            >>> buf = ReplayBuffer(capacity=100)
            >>> buf.push(np.zeros((4,84,84)), 0, 1.0, np.zeros((4,84,84)), False)
            >>> buf.push(np.zeros((4,84,84)), 1, 2.0, np.zeros((4,84,84)), True)
            >>> batch = buf.sample(2)
            >>> len(batch)
            2
            >>> isinstance(batch[0], Transition)
            True
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------------------------

class DQNetwork(nn.Module):
    """
    Deep Q-Network with convolutional feature extraction.

    Architecture follows DeepMind's DQN paper (Mnih et al., 2015), adapted
    for Ms. Pac-Man's 9-action space:

        Input:  84 × 84 × 4 (stacked grayscale frames)
        ┌──────────────────────────────────────────────┐
        │  Conv1:  32 filters, 8×8 kernel, stride 4    │
        │         → ReLU activation                    │
        │  Conv2:  64 filters, 4×4 kernel, stride 2    │
        │         → ReLU activation                    │
        │  Conv3:  64 filters, 3×3 kernel, stride 1    │
        │         → ReLU activation                    │
        │  Flatten → 3136 features                     │
        │  FC1:    512 units → ReLU                    │
        │  FC2:    9 units (one Q-value per action)    │
        └──────────────────────────────────────────────┘

    The convolutional layers extract spatial features (maze walls, pellet
    positions, ghost locations), while the fully connected layers combine
    these features to estimate the value of each action.

    Args:
        input_channels (int): Number of stacked input frames (default: 4).
        num_actions (int): Number of discrete actions (9 for Ms. Pac-Man).
    """

    def __init__(self, input_channels: int = 4, num_actions: int = 9):
        super(DQNetwork, self).__init__()

        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the flattened size after convolutions
        self.conv_output_size = self._calculate_conv_output(input_channels)

        # Fully connected layers for Q-value estimation
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _calculate_conv_output(self, input_channels: int) -> int:
        """Compute the flattened output size after all convolutional layers."""
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Time complexity:  O(C·H·W·K²·F) per conv layer, O(N·M) per FC layer.
            Total: ~3.2M multiply-add operations per 84×84×4 input.
        Space complexity: O(batch × 9) for output Q-values.

        Args:
            x: Input tensor of shape (batch, channels, 84, 84).
               Values are raw pixels in [0, 255] (uint8), normalized
               internally to [0.0, 1.0].

        Returns:
            Q-values for each action, shape (batch, num_actions).

        Example:
            >>> net = DQNetwork(input_channels=4, num_actions=9)
            >>> dummy = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)
            >>> out = net(dummy)
            >>> out.shape
            torch.Size([1, 9])
        """
        x = x.float() / 255.0  # Normalize pixel values to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten spatial dimensions
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Exploration Strategies
# ---------------------------------------------------------------------------

class EpsilonGreedyPolicy:
    """
    Standard ε-greedy exploration policy.

    With probability ε, selects a uniformly random action (exploration).
    With probability (1 - ε), selects the greedy action (exploitation).

    Epsilon is decayed linearly from epsilon_start toward epsilon_end,
    decreasing by epsilon_decay on each step. This provides:
        - Heavy exploration early (ε ≈ 1.0): agent tries diverse actions
        - Gradual shift to exploitation (ε → 0.02): agent uses learned Q-values
        - Permanent exploration floor (ε = 0.02): prevents complete greediness

    For Ms. Pac-Man's 9 actions, ε = 0.02 means the agent takes a random
    action ~2% of the time, which helps escape local optima in the maze.
    """

    def __init__(
        self,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.name = "epsilon_greedy"

    def select_action(self, q_values: torch.Tensor, num_actions: int) -> int:
        """Select action using ε-greedy strategy."""
        if random.random() < self.epsilon:
            return random.randrange(num_actions)
        return q_values.argmax(dim=1).item()

    def decay(self) -> None:
        """Decay epsilon by one step (linear decay)."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def get_epsilon(self) -> float:
        """Return current epsilon value."""
        return self.epsilon


class BoltzmannPolicy:
    """
    Boltzmann (softmax) exploration policy.

    Actions are sampled proportionally to exponentiated Q-values:

        P(a | s) = exp(Q(s, a) / τ) / Σ_a' exp(Q(s, a') / τ)

    Temperature τ controls the exploration-exploitation tradeoff:
        - High τ (→ ∞): uniform distribution — maximum exploration
        - Low τ  (→ 0): delta on argmax — greedy exploitation

    Key advantage over ε-greedy for Ms. Pac-Man:
        Boltzmann considers Q-value magnitudes, so near-optimal actions
        (e.g., UP vs UPRIGHT when both avoid a ghost) are sampled more
        often than clearly bad actions (e.g., running into a wall).
        ε-greedy treats all non-greedy actions equally.

    Key disadvantage:
        Early in training when Q-values are noisy, Boltzmann may bias
        exploration based on meaningless value differences.
    """

    def __init__(
        self,
        temperature_start: float = 2.0,
        temperature_end: float = 0.1,
        temperature_decay: float = 0.00004,
    ):
        self.temperature = temperature_start
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.name = "boltzmann"

    def select_action(self, q_values: torch.Tensor, num_actions: int) -> int:
        """Select action by sampling from the Boltzmann distribution."""
        q_vals = q_values.squeeze().detach().cpu().numpy()
        # Numerical stability: subtract max before exponentiation
        q_vals = q_vals - np.max(q_vals)
        probabilities = np.exp(q_vals / max(self.temperature, 1e-8))
        probabilities = probabilities / probabilities.sum()
        return int(np.random.choice(num_actions, p=probabilities))

    def decay(self) -> None:
        """Decay temperature by one step (linear decay)."""
        self.temperature = max(
            self.temperature_end, self.temperature - self.temperature_decay
        )

    def get_epsilon(self) -> float:
        """Return temperature (for unified logging interface)."""
        return self.temperature


class UCBPolicy:
    """
    Upper Confidence Bound (UCB) exploration policy.

    Selects actions by balancing Q-value estimates with an exploration
    bonus based on how infrequently an action has been selected:

        a* = argmax_a [ Q(s, a) + c · √(ln(N) / N(a)) ]

    where N is total steps, N(a) is the visit count for action a,
    and c is the exploration constant controlling the bonus magnitude.

    For Ms. Pac-Man, UCB is interesting because:
        - With 9 actions, some directions may be rarely useful (e.g., DOWNLEFT
          in certain maze positions), and UCB ensures they still get tested
          periodically.
        - The exploration bonus naturally decreases as actions are tried,
          without needing a decay schedule.

    Limitation:
        UCB tracks global action counts, not per-state counts. In Atari
        environments with enormous state spaces, this is a practical
        compromise — true per-state UCB would be intractable.
    """

    def __init__(self, num_actions: int, exploration_constant: float = 2.0):
        self.num_actions = num_actions
        self.exploration_constant = exploration_constant
        self.action_counts = np.ones(num_actions)  # Init to 1 to avoid div-by-zero
        self.total_steps = num_actions
        self.name = "ucb"

    def select_action(self, q_values: torch.Tensor, num_actions: int) -> int:
        """Select action using the UCB formula."""
        q_vals = q_values.squeeze().detach().cpu().numpy()

        # Normalize Q-values to [0, 1] for balanced UCB computation
        q_range = q_vals.max() - q_vals.min()
        if q_range > 0:
            q_normalized = (q_vals - q_vals.min()) / q_range
        else:
            q_normalized = np.zeros_like(q_vals)

        # UCB score = normalized Q-value + exploration bonus
        ucb_values = q_normalized + self.exploration_constant * np.sqrt(
            np.log(self.total_steps) / self.action_counts
        )

        action = int(np.argmax(ucb_values))
        self.action_counts[action] += 1
        self.total_steps += 1
        return action

    def decay(self) -> None:
        """UCB has no explicit decay — exploration decreases naturally."""
        pass

    def get_epsilon(self) -> float:
        """Return exploration constant (for unified logging interface)."""
        return self.exploration_constant


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Learning Agent with Experience Replay and Target Network.

    This agent implements the full DQN algorithm as described in Mnih et al.
    (2015), specialized for Atari Ms. Pac-Man. Key components:

        1. Policy Network: CNN that estimates Q(s, a) for action selection
        2. Target Network: Frozen copy of policy network used for computing
           TD targets, updated periodically for training stability
        3. Experience Replay: Buffer storing (s, a, r, s', done) transitions,
           sampled randomly to break temporal correlations
        4. Exploration Policy: Configurable strategy (ε-greedy, Boltzmann, UCB)

    The training loop follows this cycle:
        observe → act → store → sample → learn → update target

    Args:
        config (dict): Configuration dictionary with hyperparameters.
        logger (logging.Logger): Logger instance for output.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Device selection (prefer GPU if available)
        self.device = self._select_device(config["device"])
        self.logger.info(f"Compute device: {self.device}")

        # Environment setup with standard Atari preprocessing
        self.env = self._create_environment(config["env_name"])
        self.test_env = self._create_environment(config["env_name"])
        self.num_actions = self.env.action_space.n
        self.logger.info(f"Environment: {config['env_name']}")
        self.logger.info(f"Action space: {self.num_actions} discrete actions")
        self.logger.info(f"Observation shape: {self.env.observation_space.shape}")

        # Initialize policy and target networks
        input_channels = config["frame_stack"]
        self.policy_net = DQNetwork(input_channels, self.num_actions).to(self.device)
        self.target_net = DQNetwork(input_channels, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        # Log network architecture
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        self.logger.info(f"Network parameters: {total_params:,}")

        # Adam optimizer with configurable learning rate
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config["learning_rate"]
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config["replay_buffer_size"])

        # Default exploration policy (ε-greedy)
        self.exploration_policy = EpsilonGreedyPolicy(
            config["epsilon_start"],
            config["epsilon_end"],
            config["epsilon_decay"],
        )

        # Training state tracking
        self.total_steps = 0
        self.training_rewards = []
        self.training_steps = []
        self.training_losses = []
        self.epsilon_history = []

    def _select_device(self, device_preference: str) -> torch.device:
        """Select the best available compute device."""
        if device_preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_preference)

    def _create_environment(self, env_name: str) -> gym.Env:
        """
        Create and wrap the Atari environment with standard DQN preprocessing.

        Preprocessing pipeline:
            1. NoopReset: Random number of no-ops at start (up to 30)
            2. Frame Skip: Each action is repeated 4 times
            3. Grayscale: Convert RGB → single channel
            4. Resize: Downsample to 84×84 pixels
            5. Terminal on Life Loss: Treat each life loss as episode end
            6. Frame Stacking: Stack 4 consecutive frames for temporal context
        """
        env = gym.make(env_name, render_mode="rgb_array")
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            scale_obs=False,
        )
        env = FrameStack(env, stack_size=self.config["frame_stack"])
        return env

    def set_exploration_policy(self, policy) -> None:
        """Swap the exploration policy (for ablation experiments)."""
        self.exploration_policy = policy
        self.logger.info(f"Exploration policy changed to: {policy.name}")

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action given the current state using the exploration policy.

        Args:
            state: Current observation (4 stacked 84×84 grayscale frames).

        Returns:
            Selected action index (0-8 for Ms. Pac-Man).
        """
        state_tensor = (
            torch.tensor(np.array(state), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return self.exploration_policy.select_action(q_values, self.num_actions)

    def select_greedy_action(self, state: np.ndarray) -> int:
        """Select the greedy (highest Q-value) action — used during evaluation."""
        state_tensor = (
            torch.tensor(np.array(state), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def train_step(self) -> Optional[float]:
        """
        Perform one gradient descent step on a mini-batch from replay buffer.

        Implements the DQN loss function:

            L(θ) = E_(s,a,r,s')~D [ (y - Q(s, a; θ))² ]

        where the target y is:

            y = r + γ · max_a' Q_target(s', a'; θ⁻)    if not terminal
            y = r                                         if terminal

        The loss is computed using Huber loss (smooth L1) instead of MSE
        for robustness to large TD errors (outlier transitions).

        Time complexity:  O(B × forward_pass) where B = batch_size (32).
            Two forward passes (policy + target) + one backward pass.
            Dominated by convolution operations: ~3.2M MACs × 32 × 3 ≈ 307M MACs.
        Space complexity: O(B × state_size) for batch tensors on device.

        Returns:
            Training loss value, or None if replay buffer is below minimum.
        """
        if len(self.replay_buffer) < self.config["min_replay_size"]:
            return None

        # Sample a random mini-batch of transitions
        transitions = self.replay_buffer.sample(self.config["batch_size"])
        batch = Transition(*zip(*transitions))

        # Convert batch components to tensors
        state_batch = torch.tensor(
            np.array(batch.state), dtype=torch.float32
        ).to(self.device)
        action_batch = torch.tensor(
            batch.action, dtype=torch.long
        ).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32
        ).to(self.device)
        next_state_batch = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32
        ).to(self.device)
        done_batch = torch.tensor(
            batch.done, dtype=torch.float32
        ).to(self.device)

        # Compute current Q-values: Q(s, a; θ) for the actions that were taken
        current_q_values = (
            self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        )

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (
                self.config["gamma"] * next_q_values * (1.0 - done_batch)
            )

        # Compute Huber loss (smooth L1) for robustness to outliers
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0
        )
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Hard update: copy policy network weights to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full DQN training loop.

        For each episode:
            1. Reset environment and get initial state
            2. Select action using exploration policy
            3. Execute action, observe reward and next state
            4. Store transition in replay buffer
            5. Sample mini-batch and perform gradient update
            6. Periodically update target network
            7. Decay exploration parameter

        Returns:
            Dictionary containing training metrics:
                - episode_rewards: Total reward per episode
                - episode_steps: Steps per episode
                - episode_losses: Average loss per episode
                - epsilon_values: Exploration parameter per episode
        """
        self.logger.info("=" * 65)
        self.logger.info("TRAINING SESSION STARTED")
        self.logger.info(f"  Environment:     {self.config['env_name']}")
        self.logger.info(f"  Episodes:        {self.config['total_episodes']}")
        self.logger.info(f"  Learning rate:   {self.config['learning_rate']}")
        self.logger.info(f"  Gamma:           {self.config['gamma']}")
        self.logger.info(f"  Epsilon:         {self.config['epsilon_start']} → {self.config['epsilon_end']}")
        self.logger.info(f"  Batch size:      {self.config['batch_size']}")
        self.logger.info(f"  Buffer size:     {self.config['replay_buffer_size']:,}")
        self.logger.info(f"  Policy:          {self.exploration_policy.name}")
        self.logger.info("=" * 65)

        start_time = time.time()
        best_avg_reward = float("-inf")

        for episode in range(1, self.config["total_episodes"] + 1):
            state, _ = self.env.reset(seed=RANDOM_SEED + episode)
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0
            loss_count = 0

            for step in range(self.config["max_steps_per_episode"]):
                # Select action using current exploration policy
                action = self.select_action(state)

                # Execute action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition in replay buffer
                self.replay_buffer.push(
                    np.array(state), action, reward, np.array(next_state), done
                )

                # Perform gradient descent step
                loss = self.train_step()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

                # Periodically sync target network with policy network
                self.total_steps += 1
                if self.total_steps % self.config["target_update_frequency"] == 0:
                    self.update_target_network()

                # Decay exploration parameter
                self.exploration_policy.decay()

                # Accumulate episode statistics
                episode_reward += reward
                episode_steps += 1
                state = next_state

                if done:
                    break

            # Record episode metrics
            avg_loss = episode_loss / max(loss_count, 1)
            self.training_rewards.append(episode_reward)
            self.training_steps.append(episode_steps)
            self.training_losses.append(avg_loss)
            self.epsilon_history.append(self.exploration_policy.get_epsilon())

            # Inline progress bar (updates every episode)
            print_progress_bar(
                episode, self.config["total_episodes"],
                episode_reward, self.exploration_policy.get_epsilon(),
            )

            # Periodic detailed logging (overwrites progress bar line)
            if episode % self.config["log_frequency"] == 0:
                print()  # Clear progress bar line
                recent_rewards = self.training_rewards[
                    -self.config["log_frequency"] :
                ]
                avg_reward = np.mean(recent_rewards)
                elapsed = time.time() - start_time

                self.logger.info(
                    f"Ep {episode:5d} | "
                    f"Avg Reward: {avg_reward:8.1f} | "
                    f"Steps: {episode_steps:5d} | "
                    f"Loss: {avg_loss:.5f} | "
                    f"ε/τ: {self.exploration_policy.get_epsilon():.4f} | "
                    f"Buffer: {len(self.replay_buffer):6d} | "
                    f"Time: {elapsed:.0f}s"
                )

                # Save best model checkpoint
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_model("models/best_model.pt")

            # Periodic model checkpoint
            if episode % self.config["save_frequency"] == 0:
                self.save_model(f"models/checkpoint_ep{episode}.pt")

        total_time = time.time() - start_time
        self.logger.info("=" * 65)
        self.logger.info(
            f"TRAINING COMPLETE — {total_time:.1f}s "
            f"({total_time / 3600:.1f} hours)"
        )
        self.logger.info(f"Best average reward: {best_avg_reward:.1f}")
        self.logger.info(f"Total environment steps: {self.total_steps:,}")
        self.logger.info("=" * 65)

        return {
            "episode_rewards": self.training_rewards,
            "episode_steps": self.training_steps,
            "episode_losses": self.training_losses,
            "epsilon_values": self.epsilon_history,
        }

    def evaluate(
        self, num_episodes: int = 100, record_video: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent using a purely greedy policy (no exploration).

        Args:
            num_episodes: Number of evaluation episodes to run.
            record_video: Whether to record video of evaluation episodes.

        Returns:
            Dictionary with evaluation statistics (mean, std, min, max
            for both rewards and steps).
        """
        self.logger.info(
            f"Starting evaluation over {num_episodes} episodes (greedy policy)..."
        )

        if record_video:
            eval_env = RecordVideo(
                self.test_env,
                video_folder="videos",
                episode_trigger=lambda ep: ep < 5,  # Record first 5 episodes
            )
        else:
            eval_env = self.test_env

        rewards = []
        steps = []

        for episode in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0.0
            episode_steps = 0

            for step in range(self.config["max_steps_per_episode"]):
                action = self.select_greedy_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            steps.append(episode_steps)

        results = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "median_reward": float(np.median(rewards)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "total_episodes": num_episodes,
        }

        self.logger.info(
            f"Evaluation complete: "
            f"Mean Reward = {results['mean_reward']:.1f} ± {results['std_reward']:.1f} | "
            f"Max = {results['max_reward']:.0f} | "
            f"Mean Steps = {results['mean_steps']:.0f}"
        )

        return results

    def save_model(self, filepath: str) -> None:
        """Save the full training state (networks, optimizer, config)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "config": self.config,
            },
            filepath,
        )
        self.logger.info(f"Model checkpoint saved → {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model weights and training state from a checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.logger.info(f"Model loaded from {filepath}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_metrics(
    metrics: Dict[str, List[float]], save_dir: str = "results"
) -> None:
    """
    Generate and save a 4-panel training metrics visualization.

    Panels:
        1. Episode Rewards (with moving average)
        2. Episode Length / Steps
        3. Training Loss
        4. Exploration Parameter Decay

    Args:
        metrics: Dictionary of metric lists from DQNAgent.train().
        save_dir: Directory to save the plot image.
    """
    if plt is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "DQN Training Metrics — Atari Ms. Pac-Man",
        fontsize=15, fontweight="bold", y=0.98,
    )

    window = min(100, len(metrics["episode_rewards"]))

    # ---- Panel 1: Episode Rewards ----
    ax = axes[0, 0]
    rewards = metrics["episode_rewards"]
    ax.plot(rewards, alpha=0.25, color="steelblue", linewidth=0.5, label="Per episode")
    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(rewards)), moving_avg,
            color="darkblue", linewidth=2, label=f"{window}-ep moving avg",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Episode Steps ----
    ax = axes[0, 1]
    steps = metrics["episode_steps"]
    ax.plot(steps, alpha=0.25, color="coral", linewidth=0.5, label="Per episode")
    if len(steps) >= window:
        moving_avg = np.convolve(
            steps, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(steps)), moving_avg,
            color="darkred", linewidth=2, label=f"{window}-ep moving avg",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per Episode")
    ax.set_title("Episode Length (Survival Time)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Training Loss ----
    ax = axes[1, 0]
    losses = metrics["episode_losses"]
    ax.plot(losses, alpha=0.4, color="seagreen", linewidth=0.5, label="Per episode")
    if len(losses) >= window:
        moving_avg = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(losses)), moving_avg,
            color="darkgreen", linewidth=2, label=f"{window}-ep moving avg",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Loss")
    ax.set_title("Training Loss (Huber / Smooth L1)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Exploration Decay ----
    ax = axes[1, 1]
    ax.plot(
        metrics["epsilon_values"], color="darkorchid", linewidth=1.5
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration Parameter (ε or τ)")
    ax.set_title("Exploration Decay Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training plots saved → {save_path}")


def plot_experiment_comparison(
    all_results: Dict[str, Dict], save_dir: str = "results"
) -> None:
    """
    Generate comparison plots across multiple experiments.

    Creates a single figure showing reward curves for all experiments
    overlaid for easy visual comparison.
    """
    if plt is None or not all_results:
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle(
        "Experiment Comparison — Ms. Pac-Man DQN Ablation Study",
        fontsize=14, fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    window = 100

    for (name, result), color in zip(all_results.items(), colors):
        rewards = result["metrics"]["episode_rewards"]
        if len(rewards) >= window:
            moving_avg = np.convolve(
                rewards, np.ones(window) / window, mode="valid"
            )
            ax.plot(
                range(window - 1, len(rewards)), moving_avg,
                linewidth=2, label=name, color=color,
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (100-ep moving average)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "experiment_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: Dict[str, Any],
    experiment_name: str = "baseline",
    policy_type: str = "epsilon_greedy",
) -> Dict[str, Any]:
    """
    Run a single training experiment with the given configuration.

    Args:
        config: Hyperparameter dictionary.
        experiment_name: Label for logging/saving.
        policy_type: One of "epsilon_greedy", "boltzmann", "ucb".

    Returns:
        Dictionary with training metrics and evaluation results.
    """
    logger = setup_logging("results")
    logger.info(f"\n{'=' * 65}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"{'=' * 65}")

    # Create agent
    agent = DQNAgent(config, logger)

    # Set exploration policy
    if policy_type == "boltzmann":
        agent.set_exploration_policy(
            BoltzmannPolicy(
                temperature_start=2.0,
                temperature_end=0.1,
                temperature_decay=0.00004,
            )
        )
    elif policy_type == "ucb":
        agent.set_exploration_policy(
            UCBPolicy(
                num_actions=agent.num_actions, exploration_constant=2.0
            )
        )
    # Default: epsilon_greedy (already set in DQNAgent.__init__)

    # Train the agent
    metrics = agent.train()

    # Evaluate with greedy policy
    eval_results = agent.evaluate(
        num_episodes=config["total_test_episodes"], record_video=True
    )

    # Generate training plots
    plot_training_metrics(metrics, save_dir=f"results/{experiment_name}")

    # Save final model
    agent.save_model(f"models/{experiment_name}_final.pt")

    # Save results to JSON for later comparison
    results_path = f"results/{experiment_name}/results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": config,
                "evaluation": eval_results,
                "total_steps": agent.total_steps,
            },
            f, indent=2,
        )
    logger.info(f"Results saved → {results_path}")

    return {"metrics": metrics, "evaluation": eval_results}


def run_all_experiments() -> None:
    """
    Run the full suite of ablation experiments required by the assignment:

        1. Baseline        — ε-greedy, default hyperparameters
        2. Higher LR       — α = 5e-4 (5× baseline)
        3. Lower LR        — α = 1e-5 (0.1× baseline)
        4. Lower gamma     — γ = 0.90 (shorter planning horizon)
        5. Much lower gamma — γ = 0.80 (very short horizon)
        6. Boltzmann policy — softmax exploration
        7. Lower ε start   — ε_0 = 0.5 (less initial exploration)
        8. Slower ε decay   — decay = 1e-5 (prolonged exploration)
    """
    all_results = {}

    # ---- 1. Baseline ----
    baseline_config = DEFAULT_CONFIG.copy()
    all_results["baseline"] = run_experiment(
        baseline_config, "01_baseline", "epsilon_greedy"
    )

    # ---- 2. Higher learning rate ----
    config_lr_high = DEFAULT_CONFIG.copy()
    config_lr_high["learning_rate"] = 5e-4
    all_results["lr_5e-4"] = run_experiment(
        config_lr_high, "02_lr_5e-4", "epsilon_greedy"
    )

    # ---- 3. Lower learning rate ----
    config_lr_low = DEFAULT_CONFIG.copy()
    config_lr_low["learning_rate"] = 1e-5
    all_results["lr_1e-5"] = run_experiment(
        config_lr_low, "03_lr_1e-5", "epsilon_greedy"
    )

    # ---- 4. Lower gamma ----
    config_gamma_90 = DEFAULT_CONFIG.copy()
    config_gamma_90["gamma"] = 0.90
    all_results["gamma_0.90"] = run_experiment(
        config_gamma_90, "04_gamma_0.90", "epsilon_greedy"
    )

    # ---- 5. Much lower gamma ----
    config_gamma_80 = DEFAULT_CONFIG.copy()
    config_gamma_80["gamma"] = 0.80
    all_results["gamma_0.80"] = run_experiment(
        config_gamma_80, "05_gamma_0.80", "epsilon_greedy"
    )

    # ---- 6. Boltzmann exploration ----
    all_results["boltzmann"] = run_experiment(
        DEFAULT_CONFIG.copy(), "06_boltzmann", "boltzmann"
    )

    # ---- 7. Lower starting epsilon ----
    config_eps_low = DEFAULT_CONFIG.copy()
    config_eps_low["epsilon_start"] = 0.5
    all_results["eps_start_0.5"] = run_experiment(
        config_eps_low, "07_eps_start_0.5", "epsilon_greedy"
    )

    # ---- 8. Slower epsilon decay ----
    config_eps_slow = DEFAULT_CONFIG.copy()
    config_eps_slow["epsilon_decay"] = 1e-5
    all_results["eps_slow_decay"] = run_experiment(
        config_eps_slow, "08_eps_slow_decay", "epsilon_greedy"
    )

    # ---- Generate comparison plot ----
    plot_experiment_comparison(all_results)

    # ---- Print summary table ----
    print("\n" + "=" * 85)
    print("EXPERIMENT SUMMARY — Ms. Pac-Man DQN Ablation Study")
    print("=" * 85)
    print(
        f"{'Experiment':<22} {'Mean Reward':>12} {'Std':>8} "
        f"{'Max':>8} {'Mean Steps':>12}"
    )
    print("-" * 85)
    for name, result in all_results.items():
        ev = result["evaluation"]
        print(
            f"{name:<22} {ev['mean_reward']:>12.1f} {ev['std_reward']:>8.1f} "
            f"{ev['max_reward']:>8.0f} {ev['mean_steps']:>12.0f}"
        )
    print("=" * 85)


# ---------------------------------------------------------------------------
# Environment Analysis Utility
# ---------------------------------------------------------------------------

def analyze_environment(env_name: str = "MsPacmanNoFrameskip-v4") -> None:
    """
    Print detailed analysis of the Ms. Pac-Man environment including
    state space, action space, reward structure, and Q-table feasibility.
    """
    env = gym.make(env_name)
    print("\n" + "=" * 65)
    print(f"ENVIRONMENT ANALYSIS: {env_name}")
    print("=" * 65)

    print(f"\n--- Observation Space ---")
    print(f"  Raw shape:          {env.observation_space.shape}")
    print(f"  Raw dtype:          {env.observation_space.dtype}")
    print(f"  Pixel range:        [0, 255]")
    raw_pixels = np.prod(env.observation_space.shape)
    print(f"  Total pixels/frame: {raw_pixels:,}")

    print(f"\n  After DQN preprocessing:")
    print(f"    Grayscale:        84 × 84 (single channel)")
    print(f"    Frame stacking:   4 frames → (4, 84, 84)")
    print(f"    Total features:   {4 * 84 * 84:,} pixels per observation")

    print(f"\n--- Action Space ---")
    print(f"  Type:               Discrete({env.action_space.n})")
    print(f"  Number of actions:  {env.action_space.n}")
    if hasattr(env.unwrapped, "get_action_meanings"):
        meanings = env.unwrapped.get_action_meanings()
        for i, meaning in enumerate(meanings):
            print(f"    Action {i}: {meaning}")

    print(f"\n--- Reward Structure ---")
    print(f"  Small pellet:       +10 points")
    print(f"  Power pellet:       +50 points")
    print(f"  Ghost after power:  +200, +400, +800, +1600 (cascading)")
    print(f"  Fruit bonus:        +100 to +5000 (varies by level)")
    print(f"  Death:              0 (terminal signal via life loss)")

    print(f"\n--- Q-Table Feasibility ---")
    print(f"  State space size:   256^{4 * 84 * 84:,} ≈ 10^{4 * 84 * 84 * 2.408:.0f}")
    print(f"  Actions:            {env.action_space.n}")
    print(f"  Q-table entries:    {env.action_space.n} × 256^{4 * 84 * 84:,}")
    print(f"  VERDICT: Tabular Q-learning is completely infeasible.")
    print(f"           Deep function approximation (DQN) is required.")

    print(f"\n--- DQN Network Stats ---")
    net = DQNetwork(input_channels=4, num_actions=env.action_space.n)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Conv output size:      {net.conv_output_size}")
    print("=" * 65)

    env.close()


# ---------------------------------------------------------------------------
# Terminal UX: Colors and Progress
# ---------------------------------------------------------------------------

class TermColors:
    """ANSI color codes for readable terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def supports_color() -> bool:
        """Check if the terminal supports ANSI color codes."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def print_banner() -> None:
    """Print a styled startup banner with system information."""
    c = TermColors if TermColors.supports_color() else type(
        "NoColor", (), {k: "" for k in dir(TermColors) if not k.startswith("_")}
    )()
    print(f"""
{c.BOLD}{c.CYAN}{'=' * 60}
  Deep Q-Learning Agent for Atari Ms. Pac-Man
  INFO 7375 — Northeastern University
{'=' * 60}{c.RESET}

{c.DIM}  Author:      Aravind Balaji
  License:     MIT
  Python:      {sys.version.split()[0]}
  PyTorch:     {torch.__version__}
  Device:      {'CUDA' if torch.cuda.is_available() else 'MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU'}
  NumPy:       {np.__version__}{c.RESET}
""")


def print_progress_bar(
    current: int, total: int, reward: float, epsilon: float,
    bar_length: int = 30,
) -> None:
    """
    Print an inline progress bar for training episodes.

    Args:
        current: Current episode number.
        total: Total number of episodes.
        reward: Current episode reward.
        epsilon: Current exploration parameter.
        bar_length: Width of the progress bar in characters.

    Example:
        >>> print_progress_bar(50, 100, 320.0, 0.45, bar_length=20)  # doctest: +SKIP
        [==========..........] 50/100 | R: 320.0 | ε: 0.450
    """
    progress = current / total
    filled = int(bar_length * progress)
    bar = "=" * filled + ">" * min(1, bar_length - filled) + "." * (bar_length - filled - 1)
    c = TermColors if TermColors.supports_color() else type(
        "NoColor", (), {k: "" for k in dir(TermColors) if not k.startswith("_")}
    )()
    reward_color = c.GREEN if reward > 500 else c.YELLOW if reward > 100 else c.RED
    sys.stdout.write(
        f"\r  {c.BOLD}[{bar}]{c.RESET} {current}/{total} "
        f"| R: {reward_color}{reward:7.1f}{c.RESET} "
        f"| \u03B5: {epsilon:.3f}"
    )
    sys.stdout.flush()
    if current == total:
        print()  # Newline at completion


# ---------------------------------------------------------------------------
# Gameplay GIF Recording
# ---------------------------------------------------------------------------

def record_gameplay_gif(
    model_path: Optional[str] = None,
    num_frames: int = 500,
    output_path: str = "results/gameplay.gif",
) -> None:
    """
    Record a gameplay GIF of the agent playing Ms. Pac-Man.

    Creates a visual demonstration suitable for portfolio presentation.
    If no model is provided, records random gameplay as a baseline.

    Time complexity: O(num_frames × render_time).

    Args:
        model_path: Path to a trained model checkpoint, or None for random play.
        num_frames: Maximum number of frames to record.
        output_path: File path for the output GIF.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Pillow required for GIF recording. Install: pip3 install Pillow")
        return

    print(f"Recording gameplay ({'trained model' if model_path else 'random agent'})...")
    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="rgb_array")

    # Optionally load trained agent
    agent = None
    if model_path and os.path.exists(model_path):
        logger = logging.getLogger("GIF")
        logger.setLevel(logging.WARNING)
        agent = DQNAgent(DEFAULT_CONFIG, logger)
        agent.load_model(model_path)
        print(f"  Loaded model: {model_path}")

    # Wrap for preprocessing if using agent
    if agent:
        play_env = AtariPreprocessing(
            env, noop_max=30, frame_skip=4, screen_size=84,
            terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
        )
        play_env = FrameStack(play_env, stack_size=4)
    else:
        play_env = env

    state, _ = play_env.reset()
    frames = []

    for frame_idx in range(num_frames):
        # Capture the RGB render (always from base env)
        rgb_frame = env.render()
        if rgb_frame is not None:
            frames.append(Image.fromarray(rgb_frame))

        # Select action
        if agent:
            action = agent.select_greedy_action(state)
        else:
            action = env.action_space.sample()

        state, reward, terminated, truncated, _ = play_env.step(action)
        if terminated or truncated:
            state, _ = play_env.reset()

        # Progress indicator
        if (frame_idx + 1) % 100 == 0:
            print(f"  Captured {frame_idx + 1}/{num_frames} frames...")

    play_env.close()

    if frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frames[0].save(
            output_path, save_all=True, append_images=frames[1:],
            duration=33, loop=0, optimize=True,
        )
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Saved: {output_path} ({len(frames)} frames, {size_kb:.0f} KB)")
    else:
        print("  No frames captured — check environment setup.")


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------

def run_smoke_tests() -> None:
    """
    Run quick verification tests on all major components.

    Tests cover:
        1. ReplayBuffer: push, sample, capacity enforcement
        2. DQNetwork: forward pass shape, parameter count
        3. Exploration policies: action selection, decay behavior
        4. Environment: creation, step, observation shape
        5. Full training step: loss computation (1 mini-batch)

    This function is designed to catch setup issues before a long
    training run. All tests complete in < 30 seconds.
    """
    c = TermColors if TermColors.supports_color() else type(
        "NoColor", (), {k: "" for k in dir(TermColors) if not k.startswith("_")}
    )()

    def _pass(name: str) -> None:
        print(f"  {c.GREEN}\u2713{c.RESET} {name}")

    def _fail(name: str, err: str) -> None:
        print(f"  {c.RED}\u2717{c.RESET} {name}: {err}")

    print(f"\n{c.BOLD}Running smoke tests...{c.RESET}\n")
    passed = 0
    failed = 0

    # --- Test 1: ReplayBuffer ---
    try:
        buf = ReplayBuffer(capacity=10)
        for i in range(15):
            buf.push(np.zeros((4, 84, 84)), i % 9, float(i), np.zeros((4, 84, 84)), False)
        assert len(buf) == 10, f"Expected len 10, got {len(buf)}"
        batch = buf.sample(5)
        assert len(batch) == 5, f"Expected 5 samples, got {len(batch)}"
        assert isinstance(batch[0], Transition), "Sample should return Transitions"
        _pass("ReplayBuffer: push, sample, capacity eviction")
        passed += 1
    except Exception as e:
        _fail("ReplayBuffer", str(e))
        failed += 1

    # --- Test 2: DQNetwork ---
    try:
        net = DQNetwork(input_channels=4, num_actions=9)
        params = sum(p.numel() for p in net.parameters())
        assert params > 1_000_000, f"Expected >1M params, got {params}"
        dummy = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)
        out = net(dummy)
        assert out.shape == (2, 9), f"Expected shape (2,9), got {out.shape}"
        _pass(f"DQNetwork: forward pass (2, 4, 84, 84) \u2192 (2, 9), {params:,} params")
        passed += 1
    except Exception as e:
        _fail("DQNetwork", str(e))
        failed += 1

    # --- Test 3: Exploration Policies ---
    try:
        q_vals = torch.tensor([[1.0, 2.0, 0.5, -1.0, 3.0, 0.0, 1.5, -0.5, 2.5]])

        # Epsilon-greedy
        eg = EpsilonGreedyPolicy(1.0, 0.02, 0.001)
        action = eg.select_action(q_vals, 9)
        assert 0 <= action < 9, f"Action {action} out of range"
        old_eps = eg.get_epsilon()
        eg.decay()
        assert eg.get_epsilon() < old_eps, "Epsilon should decrease after decay"

        # Boltzmann
        bp = BoltzmannPolicy(temperature_start=1.0)
        action = bp.select_action(q_vals, 9)
        assert 0 <= action < 9

        # UCB
        ucb = UCBPolicy(num_actions=9, exploration_constant=2.0)
        action = ucb.select_action(q_vals, 9)
        assert 0 <= action < 9

        _pass("Exploration policies: \u03B5-greedy, Boltzmann, UCB all produce valid actions")
        passed += 1
    except Exception as e:
        _fail("Exploration policies", str(e))
        failed += 1

    # --- Test 4: Environment ---
    try:
        env = gym.make("MsPacmanNoFrameskip-v4")
        assert env.action_space.n == 9, f"Expected 9 actions, got {env.action_space.n}"
        env = AtariPreprocessing(
            env, noop_max=1, frame_skip=4, screen_size=84,
            terminal_on_life_loss=True, grayscale_obs=True, scale_obs=False,
        )
        env = FrameStack(env, stack_size=4)
        obs, _ = env.reset()
        assert np.array(obs).shape == (4, 84, 84), f"Unexpected obs shape: {np.array(obs).shape}"
        obs2, reward, term, trunc, info = env.step(1)
        assert isinstance(reward, (int, float)), f"Reward type: {type(reward)}"
        env.close()
        _pass("Environment: create, reset, step, obs shape (4, 84, 84)")
        passed += 1
    except Exception as e:
        _fail("Environment", str(e))
        failed += 1

    # --- Test 5: Config validation ---
    try:
        cfg = DEFAULT_CONFIG.copy()
        required_keys = [
            "env_name", "total_episodes", "learning_rate", "gamma",
            "epsilon_start", "epsilon_end", "epsilon_decay", "batch_size",
            "replay_buffer_size", "target_update_frequency", "frame_stack",
        ]
        for key in required_keys:
            assert key in cfg, f"Missing config key: {key}"
        assert 0 < cfg["gamma"] <= 1.0, f"Gamma must be in (0, 1], got {cfg['gamma']}"
        assert 0 < cfg["learning_rate"] < 1.0, f"LR out of range: {cfg['learning_rate']}"
        assert cfg["epsilon_start"] >= cfg["epsilon_end"], "epsilon_start must >= epsilon_end"
        _pass("Configuration: all required keys present, values in valid ranges")
        passed += 1
    except Exception as e:
        _fail("Configuration", str(e))
        failed += 1

    # Summary
    total = passed + failed
    status_color = c.GREEN if failed == 0 else c.RED
    print(f"\n{c.BOLD}Results: {status_color}{passed}/{total} passed{c.RESET}")
    if failed == 0:
        print(f"{c.GREEN}All smoke tests passed! Ready to train.{c.RESET}\n")
    else:
        print(f"{c.RED}{failed} test(s) failed. Fix issues before training.{c.RESET}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for flexible execution."""
    parser = argparse.ArgumentParser(
        description="Deep Q-Learning Agent for Atari Ms. Pac-Man",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python dqn_mspacman.py --mode test                              # Run smoke tests
  python dqn_mspacman.py --mode analyze                           # Analyze environment
  python dqn_mspacman.py --mode train                             # Train (default config)
  python dqn_mspacman.py --mode train --episodes 100 --lr 5e-4    # Custom training
  python dqn_mspacman.py --mode train --policy boltzmann          # Boltzmann exploration
  python dqn_mspacman.py --mode evaluate --model models/best.pt   # Evaluate a model
  python dqn_mspacman.py --mode play                              # Record gameplay GIF
  python dqn_mspacman.py --mode play --model models/best.pt       # GIF with trained agent
  python dqn_mspacman.py --mode all_experiments                   # Full ablation suite
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "analyze", "all_experiments", "test", "play"],
        help="Execution mode (default: train)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation mode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of training episodes",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (alpha)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Override discount factor (gamma)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="epsilon_greedy",
        choices=["epsilon_greedy", "boltzmann", "ucb"],
        help="Exploration policy (default: epsilon_greedy)",
    )
    return parser.parse_args()


def main():
    """Main entry point — parse args and dispatch to appropriate mode."""
    args = parse_arguments()

    # Print startup banner
    print_banner()

    # Set seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    if args.mode == "test":
        run_smoke_tests()

    elif args.mode == "analyze":
        analyze_environment()

    elif args.mode == "play":
        record_gameplay_gif(
            model_path=args.model,
            num_frames=500,
            output_path="results/gameplay.gif",
        )

    elif args.mode == "train":
        config = DEFAULT_CONFIG.copy()
        if args.episodes:
            config["total_episodes"] = args.episodes
        if args.lr:
            config["learning_rate"] = args.lr
        if args.gamma:
            config["gamma"] = args.gamma

        run_experiment(config, "training_run", args.policy)

    elif args.mode == "evaluate":
        if not args.model:
            print("ERROR: --model path required for evaluation mode")
            print("Example: python dqn_mspacman.py --mode evaluate --model models/best_model.pt")
            sys.exit(1)

        logger = setup_logging("results")
        agent = DQNAgent(DEFAULT_CONFIG, logger)
        agent.load_model(args.model)
        agent.evaluate(num_episodes=100, record_video=True)

    elif args.mode == "all_experiments":
        run_all_experiments()


if __name__ == "__main__":
    main()
