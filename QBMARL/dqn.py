"""
DQN brain for one agent: Q-network + replay buffer + training step.
Two of these, trained independently, are the two "minds" in the simulation.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """One independent brain: its own network, target network, replay buffer,
    and exploration schedule. Two instances of this = two independent minds."""

    def __init__(self, obs_dim, num_actions, seed=0, lr=1e-4, gamma=0.99,
                 buffer_size=20000, batch_size=128):
        torch.manual_seed(seed)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.q_net = QNetwork(obs_dim, num_actions)
        self.target_net = QNetwork(obs_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.998

        self.train_steps = 0
        # With up to 500 env steps per episode, train_step() fires up to
        # 500 times per episode, so lr and the target-update cadence need
        # to be conservative or the online network chases a fast-moving
        # target and diverges instead of converging. lr=1e-4 with
        # batch_size=128 (a steadier gradient estimate) and syncing the
        # target net every 1000 steps (relative to the slower-moving
        # online net) keeps training loss low and flat.
        self.target_update_every = 1000

    # DECLARE isn't catastrophic (a wrong guess is just a small penalty,
    # not episode-ending -- see env.py), so it doesn't need to be
    # suppressed during random exploration. PICKUP/DROP are kept rarer
    # since they only make sense in specific states.
    # SENSE is cheap, only usable once per episode, and genuinely
    # informative (narrows the search to half the map) -- weighted close
    # to SEARCH so random exploration actually discovers it's useful.
    EXPLORE_WEIGHTS = [0.16, 0.16, 0.16, 0.16, 0.14, 0.04, 0.02, 0.02, 0.14]
    # up   down  left  right search pickup drop declare sense

    def act(self, obs, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choices(range(self.num_actions), weights=self.EXPLORE_WEIGHTS, k=1)[0]
        with torch.no_grad():
            q = self.q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s = torch.tensor(s)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r)
        s2 = torch.tensor(s2)
        done = torch.tensor(done)

        q_values = self.q_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            # Double DQN: select the best next action with the online network,
            # but evaluate its value with the target network -- avoids the
            # overestimation that otherwise makes DQN collapse onto a single
            # repeated bad action.
            next_actions = self.q_net(s2).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            target = r + self.gamma * next_q * (1 - done)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        self.target_net.load_state_dict(self.q_net.state_dict())
