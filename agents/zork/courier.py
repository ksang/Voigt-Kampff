"""
Courer is an agent that not learning to play Zork but exploring the environment.
The observations will be recorded in output file.
"""
import sys,os
import argparse
import itertools
from collections import defaultdict
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.zork import ZorkEnv

cmd_parser = argparse.ArgumentParser(description=None)
# Data arguments
cmd_parser.add_argument('-a', '--action_file', default='data/131_actions.txt',
                        help='Line sapareted text file of actions.')
cmd_parser.add_argument('-o', '--output', default='output.zip',
                        help='Output text filename.')
# Training arguments
cmd_parser.add_argument('-i', '--episode_num', default=100, type=int,
                        help='Number of episode to run.')
cmd_parser.add_argument('-msn', '--max_step_num', default=10000, type=int,
                        help='Maximun step number for each episode.')

def load_actions(filename):
    """
    Load a list of actions from a line sapareted text file
    """
    with open(filename, "r") as fd:
        actions = fd.read().splitlines()
    return actions

def make_rand_policy(action_num):
    """
    Make a policy that takes any observation and returns a random action id.
    """
    def rand_policy(observation):
        return np.random.randint(action_num)
    return rand_policy

def save_observations(observation_dictionary):
    for k in observation_dictionary:
        print(k,':',observation_dictionary[k])

class Courier(object):
    def __init__(self, actions):
        self.actions = actions
        self.action_num = len(self.actions)
        self.policy = make_rand_policy(self.action_num)
        self.observation_dictionary = defaultdict(lambda: 0)

    def act(self, observation):
        return self.actions[self.policy(observation)]

    def update(self, action, state, reward, done):
        self.observation_dictionary[state.description] += 1
        self.observation_dictionary[action] += 1


def run(args):
    env_wrapper = ZorkEnv()
    env = env_wrapper.start()
    actions = load_actions(args.action_file)
    print('Actions loaded:', len(actions))
    agent = Courier(actions)
    for i in range(args.episode_num):
        state = env.reset()
        for t in itertools.count():
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            print('action:', action)
            print('ob:', next_state.description)
            agent.update(action, next_state, reward, done)
            if done or t >= args.max_step_num:
                break
            state = next_state

    save_observations(agent.observation_dictionary)

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    run(args)
