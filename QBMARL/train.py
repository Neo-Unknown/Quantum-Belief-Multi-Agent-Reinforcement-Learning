"""
Trains two independent DQN agents (Agent A and Agent B) on the SAME shared
world each episode -- same target room, same clue set -- but each explores
and learns from its own separate experience.

A wrong DECLARE doesn't end the episode (see env.py) -- it's just a
small penalty -- so random exploration doesn't need to avoid DECLARE to
survive long enough to learn anything.

Run: python train.py --episodes 3000
Saves: agent_a.pt, agent_b.pt, training_log.csv (checkpointed, resumable)
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from env import QuantumBeliefEnv, World, OBS_DIM, NUM_ACTIONS
from dqn import DQNAgent

NUM_EPISODES = 400
LOG_EVERY = 50


def run_episode(env, agent, train=True):
    obs = env.reset(world=env.world)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs, greedy=not train)
        next_obs, reward, done, _ = env.step(action)
        if train:
            agent.remember(obs, action, reward, next_obs, done)
            agent.train_step()
        obs = next_obs
        total_reward += reward
    if train:
        agent.decay_epsilon()
    success = env.opened
    return total_reward, success, env.steps


def main(num_episodes=NUM_EPISODES, resume=False):
    out_dir = os.path.dirname(__file__)
    agent_a = DQNAgent(OBS_DIM, NUM_ACTIONS, seed=1)
    agent_b = DQNAgent(OBS_DIM, NUM_ACTIONS, seed=2)

    start_ep = 0
    if resume and os.path.exists(os.path.join(out_dir, "agent_a.pt")):
        agent_a.load(os.path.join(out_dir, "agent_a.pt"))
        agent_b.load(os.path.join(out_dir, "agent_b.pt"))
        if os.path.exists(os.path.join(out_dir, "progress.txt")):
            with open(os.path.join(out_dir, "progress.txt")) as f:
                start_ep = int(f.read().strip())
        agent_a.epsilon = max(agent_a.epsilon_min, 1.0 * (agent_a.epsilon_decay ** start_ep))
        agent_b.epsilon = max(agent_b.epsilon_min, 1.0 * (agent_b.epsilon_decay ** start_ep))
        print(f"Resumed from episode {start_ep}, epsilon={agent_a.epsilon:.3f}")

    # start_room=None -> each reset() drops the agent in a fresh random
    # room (excluding the target) instead of the same fixed corner every
    # episode -- see env.py's QuantumBeliefEnv docstring/_random_start_room
    env_a = QuantumBeliefEnv(start_room=None, agent_id="A")
    env_b = QuantumBeliefEnv(start_room=None, agent_id="B")

    log_rows = []
    window_a, window_b = [], []

    for ep in range(start_ep + 1, start_ep + num_episodes + 1):
        world = World()  # same world for both agents this episode
        env_a.world = world
        env_b.world = world

        r_a, s_a, steps_a = run_episode(env_a, agent_a, train=True)
        r_b, s_b, steps_b = run_episode(env_b, agent_b, train=True)

        window_a.append(s_a)
        window_b.append(s_b)
        window_a = window_a[-50:]
        window_b = window_b[-50:]

        log_rows.append({
            "episode": ep,
            "reward_a": r_a, "success_a": int(s_a), "steps_a": steps_a, "eps_a": agent_a.epsilon,
            "reward_b": r_b, "success_b": int(s_b), "steps_b": steps_b, "eps_b": agent_b.epsilon,
        })

        if ep % LOG_EVERY == 0:
            sr_a = 100 * sum(window_a) / len(window_a)
            sr_b = 100 * sum(window_b) / len(window_b)
            print(f"ep {ep:4d} | A: reward={r_a:6.2f} success_rate(50)={sr_a:5.1f}% eps={agent_a.epsilon:.2f} "
                  f"| B: reward={r_b:6.2f} success_rate(50)={sr_b:5.1f}% eps={agent_b.epsilon:.2f}")
            agent_a.save(os.path.join(out_dir, "agent_a.pt"))
            agent_b.save(os.path.join(out_dir, "agent_b.pt"))
            with open(os.path.join(out_dir, "progress.txt"), "w") as f:
                f.write(str(ep))

    agent_a.save(os.path.join(out_dir, "agent_a.pt"))
    agent_b.save(os.path.join(out_dir, "agent_b.pt"))
    with open(os.path.join(out_dir, "progress.txt"), "w") as f:
        f.write(str(start_ep + num_episodes))

    log_path = os.path.join(out_dir, "training_log.csv")
    write_header = not (resume and os.path.exists(log_path))
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nSaved agent_a.pt, agent_b.pt, training_log.csv (total episodes so far: {start_ep + num_episodes})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(num_episodes=args.episodes, resume=args.resume)
