import gym
import argparse
import agents
from envs.tetris import Tetris

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument('-t', '--horizon', default=500,
                        help='Maximum time steps')
cmd_parser.add_argument('-m', '--model', choices=['cnn','linear'], default='linear',
                        help='Model used for DQN')
cmd_parser.add_argument('-n', '--num-episode', default=1000,
                        help='Number of episode for training')

def render(env, t, n):
    print("Episode: ", n, "Step:", t)
    env.render()

def train(env, agent, num_episode):

    for n in range(num_episode):
        state = env.reset()
        i = 0
        while True:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.update((state, next_state, action, reward, done))
            if done:
                break
            state = next_state
        render(env, i, n)

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    if args.model == 'linear':
        env = Tetris(horizon=args.horizon, flattened_observation=True)
    agent = agents.DQN.DQNAgent(env)
    train(env, agent, args.num_episode)
