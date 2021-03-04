import gym
import argparse
import agents
from envs.tetris import Tetris
from utils.plots import plot_reward_loss

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument("-t", "--horizon", default=500, help="Maximum time steps")
cmd_parser.add_argument(
    "-m",
    "--model",
    choices=["cnn", "linear"],
    default="linear",
    help="Model used for DQN",
)
cmd_parser.add_argument(
    "-n", "--num-frames", default=500000, help="Number of frames for training"
)


def render(env, t, n):
    print("Episode: ", n, "Step:", t)
    env.render()


def train(env, agent, num_frames):
    rewards = []
    state = env.reset()
    episode_reward = 0
    for frame_idx in range(1, num_frames + 1):
        action = agent.take_action(state)
        next_state, reward, done = env.step(action)
        agent.update((state, next_state, action, reward, done))
        episode_reward += reward
        if done:
            rewards.append(episode_reward)
            state = env.reset()
            episode_reward = 0
            continue
        if frame_idx % 100000 == 0:
            plot_reward_loss(frame_idx, rewards, agent.losses)
        state = next_state


if __name__ == "__main__":
    args = cmd_parser.parse_args()
    config = agents.DQN.Config()
    config.model_arch = args.model

    if args.model == "linear":
        env = Tetris(horizon=args.horizon, flattened_observation=True)
    elif args.model == "cnn":
        env = Tetris(horizon=args.horizon)

    agent = agents.DQN.DQNAgent(env, config)
    train(env, agent, int(args.num_frames))
