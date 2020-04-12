import gym
import argparse
import agents
from envs import Tetris

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument('-t', '--horizon', default=500,
                        help='Maximum time steps')

def render(env, t):
    print("Step:", t)
    env.render()

def run(env, agent):
    env.reset()
    i = 0
    render(env, i)
    while True:
        (_, _), _, done  = env.step(agent.sample_action())
        if done:
            break
        i += 1
    render(env, i)

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    env = Tetris(horizon=args.horizon)
    agent = agents.Random.RandomAgent(env, None)
    run(env, agent)
