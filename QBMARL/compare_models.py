"""
Model A vs Model B vs Model C comparison experiment: a 3-way ablation
over the entropy-threshold collapse mechanism and the metacognitive
control layer built on top of it.

Model A -- "conventional": QuantumBeliefEnv(enable_collapse=False,
    metacognition=False). Interference/entanglement/disturbance are still
    part of the quantum belief model underneath (removing those would mean
    testing a different, much simpler environment than the rest of this
    project) -- what's specifically ablated is the entropy-threshold
    collapse event, and the second-order self-monitoring signal built on
    top of it, so the comparison isn't overclaimed as "quantum vs
    non-quantum."

Model B -- "collapse + metacognition (observation only)":
    QuantumBeliefEnv(enable_collapse=True, metacognition=True).
    Unprompted entropy-threshold collapse, plus entropy_trend()/
    confidence() exposed as two extra numbers in the observation vector
    for the DQN to (maybe) learn to use on its own. As
    metacognitive_controller.py's own docstring points out, a DQN
    treats those two numbers exactly like any other input dimension --
    there's no guarantee they influence anything. Model B alone can
    only ever show "did the DQN learn to use these extra features,"
    which is a narrower and weaker claim than "does metacognitive
    self-monitoring change behavior."

Model C -- "collapse + metacognition (explicit controller)": same
    env config as Model B (enable_collapse=True, metacognition=True),
    but each agent's action selection is wrapped in a REAL
    MetacognitiveController (metacognitive_controller.py) that reads
    confidence()/entropy_trend() and explicitly, auditably overrides
    the DQN's chosen action when its deferral regimes fire -- rather
    than hoping the network picks up on the signal by itself. This is
    what actually exercises the mechanism the "metacognition" claim in
    this project depends on. Off-policy Double DQN (dqn.py) learns
    correctly from transitions gathered under any behavior policy,
    including one occasionally overridden by the controller, since
    what's stored via agent.remember is the action actually taken,
    whichever produced it.

Both models see the SAME sequence of worlds (same target room, same
clues, episode-for-episode) by reseeding Python's `random` to the same
per-episode seed before constructing each model's World -- so any
measured difference in behavior comes from the mechanism being tested,
not from harder or easier worlds landing on one model by chance.

Run: python compare_models.py --episodes 400

Saves: compare_log.csv (per-episode rows tagged by model), and prints a
summary table at the end.

Honest scope note: this script measures whether the mechanisms change
LEARNED BEHAVIOR (success rate, reward, steps-to-solve, collapse
frequency) under this specific environment and these specific
hyperparameters, from a single seed per condition -- see README.md's
scope note for what would be needed to generalize the result.
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from env import QuantumBeliefEnv, World, NUM_ACTIONS, OBS_DIM_BASE, OBS_DIM_META, GRID_SIZE
from dqn import DQNAgent
from metacognitive_controller import MetacognitiveController

LOG_EVERY = 25


def run_episode(env, agent, world, start_room, train=True):
    env.world = world
    obs = env.reset(world=world, start_room=start_room)
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
    return total_reward, env.opened, env.steps, env.or_triggered


def make_training_policy(agent):
    """Epsilon-greedy policy usable as a MetacognitiveController
    base_policy DURING TRAINING -- distinct from policies.py's
    make_dqn_policy, which always acts greedy(=True) because it's meant
    for evaluating an already-trained checkpoint, not for training with
    exploration. MetacognitiveController.act(env, carry_state) doesn't
    pass obs in, so this reads it straight off env via env._obs()."""
    def policy(env, carry_state):
        return agent.act(env._obs(), greedy=False)
    return policy


def run_episode_controlled(env, agent, controller, world, start_room, train=True):
    """Same contract as run_episode, but actions come from a
    MetacognitiveController wrapping the agent's own epsilon-greedy
    policy, so the controller can actually intercept and override the
    action when its deferral regimes fire (see
    metacognitive_controller.py), instead of the confidence/
    entropy_trend signal only sitting passively in the observation
    vector. Returns an extra `overrides` count (how many times this
    episode the controller diverged from the base DQN's choice) on top
    of run_episode's normal return values, for logging/auditing."""
    env.world = world
    obs = env.reset(world=world, start_room=start_room)
    carry_state = {}
    controller.reset_log()
    total_reward = 0.0
    done = False
    while not done:
        action = controller.act(env, carry_state)
        next_obs, reward, done, _ = env.step(action)
        if train:
            agent.remember(obs, action, reward, next_obs, done)
            agent.train_step()
        obs = next_obs
        total_reward += reward
    if train:
        agent.decay_epsilon()
    return total_reward, env.opened, env.steps, env.or_triggered, len(controller.overrides)


def make_model(tag, enable_collapse, metacognition, obs_dim, seed_offset):
    agent_a = DQNAgent(obs_dim, NUM_ACTIONS, seed=1 + seed_offset)
    agent_b = DQNAgent(obs_dim, NUM_ACTIONS, seed=2 + seed_offset)
    # start_room=None -> random spawn each reset, same reasoning as
    # train.py. Important here specifically: with fixed corners, any
    # success-rate gap between Model A and Model B could partly reflect
    # "who happened to start closer to the target this seed" rather than
    # the collapse/metacognition mechanism actually being compared.
    env_a = QuantumBeliefEnv(start_room=None, agent_id="A",
                               enable_collapse=enable_collapse, metacognition=metacognition)
    env_b = QuantumBeliefEnv(start_room=None, agent_id="B",
                               enable_collapse=enable_collapse, metacognition=metacognition)
    return {
        "tag": tag, "agent_a": agent_a, "agent_b": agent_b,
        "env_a": env_a, "env_b": env_b,
        "window_a": [], "window_b": [],
    }


def make_model_c(tag, obs_dim, seed_offset):
    """Model C: same env config as Model B (collapse + metacognition
    enabled), but wraps each agent's action selection with the real
    MetacognitiveController rather than only exposing entropy_trend/
    confidence as passive observation features. See module docstring
    for why this is a meaningfully different, stronger test of the
    "metacognition changes behavior" claim than Model B alone."""
    agent_a = DQNAgent(obs_dim, NUM_ACTIONS, seed=3 + seed_offset)
    agent_b = DQNAgent(obs_dim, NUM_ACTIONS, seed=4 + seed_offset)
    env_a = QuantumBeliefEnv(start_room=None, agent_id="A",
                               enable_collapse=True, metacognition=True)
    env_b = QuantumBeliefEnv(start_room=None, agent_id="B",
                               enable_collapse=True, metacognition=True)
    controller_a = MetacognitiveController(make_training_policy(agent_a), name="dqn+controller-A")
    controller_b = MetacognitiveController(make_training_policy(agent_b), name="dqn+controller-B")
    return {
        "tag": tag, "agent_a": agent_a, "agent_b": agent_b,
        "env_a": env_a, "env_b": env_b,
        "controller_a": controller_a, "controller_b": controller_b,
        "window_a": [], "window_b": [],
    }


def _random_start_room(world):
    """Uniformly random room, excluding the world's target room -- same
    rule as env.py's QuantumBeliefEnv._random_start_room, duplicated
    here (rather than reused) because this needs to draw a room BEFORE
    any QuantumBeliefEnv exists for this episode."""
    target = world.target_room
    while True:
        room = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if room != target:
            return room


def matched_world_set(seed, n_conditions):
    """Generalizes the original matched_world_pair to N conditions
    (originally 2, now 3 with Model C added). Builds N separately-
    instantiated World objects that are configuration-identical (same
    target room, same clue set/placement) by reseeding to the same
    value immediately before each construction. Separate instances
    matter because PICKUP/DROP mutates a World's live clue positions --
    different models' agents must not accidentally share (and fight
    over) one mutable object.

    Draws each agent's random start room ONCE, off the first world's
    seeded stream, and returns it for ALL conditions to reuse verbatim
    (rather than letting each QuantumBeliefEnv randomize its own start
    independently). Every condition's agent A needs to start this
    episode in the SAME room -- otherwise randomizing start position
    would reintroduce the very start-position confound this function
    exists to eliminate. All N worlds are configuration-identical (same
    seed => same target room), so a room drawn against world[0]'s
    target is equally valid for every other world; no need to draw more
    than once."""
    random.seed(seed)
    first_world = World()
    start_room_a = _random_start_room(first_world)
    start_room_b = _random_start_room(first_world)

    worlds = [first_world]
    for _ in range(n_conditions - 1):
        random.seed(seed)
        worlds.append(World())

    return worlds, start_room_a, start_room_b


def main(num_episodes, base_seed, log_path, resume=False):
    out_dir = os.path.dirname(os.path.abspath(log_path)) or "."
    ckpt = {
        "A_no_collapse": ("cmp_a_agent_a.pt", "cmp_a_agent_b.pt"),
        "B_collapse_metacog": ("cmp_b_agent_a.pt", "cmp_b_agent_b.pt"),
        "C_meta_controller": ("cmp_c_agent_a.pt", "cmp_c_agent_b.pt"),
    }
    progress_path = os.path.join(out_dir, "compare_progress.txt")

    model_a = make_model("A_no_collapse", enable_collapse=False, metacognition=False,
                          obs_dim=OBS_DIM_BASE, seed_offset=0)
    model_b = make_model("B_collapse_metacog", enable_collapse=True, metacognition=True,
                          obs_dim=OBS_DIM_META, seed_offset=100)
    model_c = make_model_c("C_meta_controller", obs_dim=OBS_DIM_META, seed_offset=200)
    all_models = (model_a, model_b, model_c)
    controlled_tags = {"C_meta_controller"}  # models run via run_episode_controlled

    start_ep = 0
    write_header = True
    if resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            start_ep = int(f.read().strip())
        for model in all_models:
            fa, fb = ckpt[model["tag"]]
            pa, pb = os.path.join(out_dir, fa), os.path.join(out_dir, fb)
            if os.path.exists(pa) and os.path.exists(pb):
                model["agent_a"].load(pa)
                model["agent_b"].load(pb)
                model["agent_a"].epsilon = max(model["agent_a"].epsilon_min,
                                                1.0 * (model["agent_a"].epsilon_decay ** start_ep))
                model["agent_b"].epsilon = max(model["agent_b"].epsilon_min,
                                                1.0 * (model["agent_b"].epsilon_decay ** start_ep))
        write_header = not os.path.exists(log_path)
        print(f"Resumed from episode {start_ep}, eps_a(A)={model_a['agent_a'].epsilon:.3f}")

    f_log = open(log_path, "a", newline="")
    writer = None
    all_rows_this_run = []
    # unified schema across all 3 models -- overrides_a/overrides_b are 0
    # for models A and B (they have no controller), and the real
    # per-episode override count for model C. Keeping one fixed schema
    # avoids csv.DictWriter choking on rows with different fields.
    fieldnames = ["episode", "model", "reward_a", "success_a", "steps_a", "or_a",
                  "reward_b", "success_b", "steps_b", "or_b",
                  "overrides_a", "overrides_b", "eps_a", "eps_b"]

    for ep in range(start_ep + 1, start_ep + num_episodes + 1):
        seed = base_seed + ep
        worlds, start_room_a, start_room_b = matched_world_set(seed, len(all_models))

        for model, world in zip(all_models, worlds):
            if model["tag"] in controlled_tags:
                r_a, s_a, steps_a, or_a, ov_a = run_episode_controlled(
                    model["env_a"], model["agent_a"], model["controller_a"], world, start_room_a, train=True)
                r_b, s_b, steps_b, or_b, ov_b = run_episode_controlled(
                    model["env_b"], model["agent_b"], model["controller_b"], world, start_room_b, train=True)
            else:
                r_a, s_a, steps_a, or_a = run_episode(model["env_a"], model["agent_a"], world, start_room_a, train=True)
                r_b, s_b, steps_b, or_b = run_episode(model["env_b"], model["agent_b"], world, start_room_b, train=True)
                ov_a, ov_b = 0, 0

            model["window_a"] = (model["window_a"] + [s_a])[-50:]
            model["window_b"] = (model["window_b"] + [s_b])[-50:]
            row = {
                "episode": ep, "model": model["tag"],
                "reward_a": r_a, "success_a": int(s_a), "steps_a": steps_a, "or_a": int(or_a),
                "reward_b": r_b, "success_b": int(s_b), "steps_b": steps_b, "or_b": int(or_b),
                "overrides_a": ov_a, "overrides_b": ov_b,
                "eps_a": model["agent_a"].epsilon, "eps_b": model["agent_b"].epsilon,
            }
            all_rows_this_run.append(row)
            if writer is None:
                writer = csv.DictWriter(f_log, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
            writer.writerow(row)

        if ep % LOG_EVERY == 0:
            f_log.flush()
            for model in all_models:
                sr_a = 100 * sum(model["window_a"]) / len(model["window_a"])
                sr_b = 100 * sum(model["window_b"]) / len(model["window_b"])
                print(f"[{model['tag']:18s}] ep {ep:4d} | "
                      f"A success_rate(50)={sr_a:5.1f}% eps={model['agent_a'].epsilon:.2f} | "
                      f"B success_rate(50)={sr_b:5.1f}% eps={model['agent_b'].epsilon:.2f}")
                fa, fb = ckpt[model["tag"]]
                model["agent_a"].save(os.path.join(out_dir, fa))
                model["agent_b"].save(os.path.join(out_dir, fb))
            with open(progress_path, "w") as pf:
                pf.write(str(ep))

    f_log.close()
    final_ep = start_ep + num_episodes
    with open(progress_path, "w") as pf:
        pf.write(str(final_ep))
    for model in all_models:
        fa, fb = ckpt[model["tag"]]
        model["agent_a"].save(os.path.join(out_dir, fa))
        model["agent_b"].save(os.path.join(out_dir, fb))

    print(f"\n=== Summary (last 50 episodes seen this run, episode {final_ep}) ===")
    for model in all_models:
        sr_a = 100 * sum(model["window_a"]) / len(model["window_a"])
        sr_b = 100 * sum(model["window_b"]) / len(model["window_b"])
        last50 = [r for r in all_rows_this_run if r["model"] == model["tag"]][-50:]
        if last50:
            avg_reward = sum(r["reward_a"] + r["reward_b"] for r in last50) / (2 * len(last50))
            solved_steps = [r["steps_a"] for r in last50 if r["success_a"]] + \
                            [r["steps_b"] for r in last50 if r["success_b"]]
            avg_steps = sum(solved_steps) / len(solved_steps) if solved_steps else float("nan")
            or_rate = 100 * sum(r["or_a"] + r["or_b"] for r in last50) / (2 * len(last50))
            avg_overrides = sum(r["overrides_a"] + r["overrides_b"] for r in last50) / (2 * len(last50))
            extra = f" | avg_overrides/ep={avg_overrides:4.1f}" if model["tag"] in controlled_tags else ""
            print(f"{model['tag']:18s} | success A={sr_a:5.1f}% B={sr_b:5.1f}% | "
                  f"avg_reward={avg_reward:6.2f} | avg_steps_to_solve={avg_steps:6.1f} | "
                  f"collapse_fired={or_rate:4.1f}% of episodes{extra}")

    print(f"\nSaved {log_path} (total episodes so far: {final_ep})")
    print("\nNote: if you change env.py's observation/action space or")
    print("OR_ENTROPY_THRESHOLD, delete old .pt/.csv/progress files before")
    print("running fresh, or --resume will load stale/invalid state.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "compare_log.csv"))
    args = parser.parse_args()
    main(args.episodes, args.seed, args.out, resume=args.resume)
