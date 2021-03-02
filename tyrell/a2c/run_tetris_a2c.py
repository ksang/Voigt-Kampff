import gym
import argparse
import agents
import torch
from envs.tetris import Tetris
from utils.plots import plot_reward_loss

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument('-t', '--horizon', default=500,
                        help='Maximum time steps')
cmd_parser.add_argument('-n', '--num-frames', default=50000,
                        help='Number of frames for training')
cmd_parser.add_argument('-cuda', '--enable-cuda', default=False, action='store_true',
                        help='Enable CUDA if available')

def train(env, agent, num_frames, device):
    total_rewards = []
    state = env.reset()
    episode_reward = 0
    entropy = 0
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    for frame_idx in range(1, num_frames+1):
        dist, value = agent.actions_value(state)
        action = dist.sample()

        next_state, reward, done = env.step(action)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob.unsqueeze(0).to(device))
        values.append(value)
        rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

        episode_reward += reward

        if done:
            agent.update((next_state, log_probs, rewards, masks, values, entropy))
            total_rewards.append(episode_reward)

            state = env.reset()
            episode_reward = 0
            log_probs = []
            values    = []
            rewards   = []
            masks     = []
            continue
        if frame_idx % 10000 == 0:
            plot_reward_loss(frame_idx, total_rewards, agent.losses)
        state = next_state

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    use_cuda = torch.cuda.is_available() and args.enable_cuda
    device   = torch.device("cuda" if use_cuda else "cpu")
    env = Tetris(horizon=args.horizon, flattened_observation=True)
    agent = agents.A2C.A2CAgent(env)
    train(env, agent, int(args.num_frames), device)
