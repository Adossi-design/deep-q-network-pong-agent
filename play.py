
"""
DQN Pong Agent - Gameplay Evaluation
Author: Fidel Chretien Impano

Loads a trained DQN state dictionary and evaluates the agent
using greedy action selection in the ALE/Pong-v5 environment.
"""

import argparse
from pathlib import Path

import torch

from train import QNetwork, make_env


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN Pong agent."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained .pth model file."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of gameplay episodes."
    )

    return parser.parse_args()


def load_agent(model_path, n_actions, device):
    """Load the trained DQN network."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    network = QNetwork(n_actions).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    network.load_state_dict(state_dict)
    network.eval()

    return network


def select_action(network, observation, device):
    """Choose the action with the highest predicted Q-value."""

    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state)
        action = q_values.argmax(dim=1).item()

    return action


def evaluate_agent(network, env, episodes, device):
    """Run the trained agent for the requested episodes."""

    episode_rewards = []

    for episode in range(1, episodes + 1):

        observation, _ = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = select_action(
                network,
                observation,
                device
            )

            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps}"
        )

    average_reward = sum(episode_rewards) / len(episode_rewards)

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Worst reward: {min(episode_rewards):.2f}")
    print("========================================")


def main():
    args = get_args()

    if args.episodes < 1:
        raise ValueError("Episodes must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Evaluation device: {device}")

    env = make_env(render_mode="human")

    try:
        n_actions = env.action_space.n

        network = load_agent(
            args.model,
            n_actions,
            device
        )

        print(f"Loaded model: {args.model}")
        print(f"Available actions: {n_actions}")
        print("Starting gameplay evaluation...\n")

        evaluate_agent(
            network,
            env,
            args.episodes,
            device
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
