"""Train a Deep Q-Network agent to play Atari Pong."""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = 50000
LEARNING_STARTS = 10000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1000
GRAD_CLIP = 10


# Read the command-line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--out", type=str, default="dqn_model")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Build the Pong environment
def make_env(render_mode=None):
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    return env


# The Q-network
class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x).flatten(start_dim=1)
        return self.fc(x)


# The replay buffer
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.position = 0
        self.count = 0
        self.states = np.zeros((size, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((size, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.position = (self.position + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.count, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )


# Epsilon-greedy action selection
def get_epsilon(step, eps_start, eps_end, eps_decay):
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def choose_action(net, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device)
        return net(state_tensor).argmax(dim=1).item()


# One learning update
def learn(online_net, target_net, buffer, optimizer, loss_fn, batch_size, gamma):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1).values
        target = rewards + gamma * max_next_q * (1 - dones)

    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
    optimizer.step()


# Training loop
def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    n_actions = env.action_space.n
    online_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(BUFFER_SIZE)

    out_folder = os.path.dirname(args.out)
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
    log_file = open(args.out + "_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["episode", "episode_length", "total_steps", "reward", "epsilon"])

    state, _ = env.reset(seed=args.seed)
    episode = 0
    episode_reward = 0.0
    episode_length = 0
    recent_rewards = []

    for step in range(1, args.steps + 1):
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = choose_action(online_net, state, epsilon, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Only terminated counts as done, a truncated episode still has a future
        buffer.add(state, action, reward, next_state, float(terminated))
        state = next_state
        episode_reward += reward
        episode_length += 1

        if terminated or truncated:
            episode += 1
            log.writerow([episode, episode_length, step, episode_reward, round(epsilon, 4)])
            log_file.flush()
            recent_rewards.append(episode_reward)
            if episode % 10 == 0:
                average = np.mean(recent_rewards[-100:])
                print(f"Episode {episode} | steps {step} | epsilon {epsilon:.3f} | "
                      f"avg reward (last 100) {average:.1f}")
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        if buffer.count >= LEARNING_STARTS and step % TRAIN_EVERY == 0:
            learn(online_net, target_net, buffer, optimizer, loss_fn,
                  args.batch_size, args.gamma)

        if step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(online_net.state_dict())

    log_file.close()
    env.close()
    torch.save(online_net.state_dict(), args.out + ".pth")
    print("Saved", args.out + ".pth")


if __name__ == "__main__":
    main()
