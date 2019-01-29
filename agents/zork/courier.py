"""
Courer is an agent that not learning to play Zork but exploring the environment.
The observations will be recorded in output file.
"""
import sys,os
import json
import argparse
import timeit
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
cmd_parser.add_argument('-o', '--output', default='output.txt',
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
    def rand_policy(observation, exclude_list):
        if len(exclude_list) >= action_num:
            return np.random.randint(action_num)
        action_list = [x for x in range(action_num) if x not in exclude_list]
        return np.random.choice(action_list)
    return rand_policy

def save_observations(observation_dictionary, out_file):
    output = json.dumps(observation_dictionary, sort_keys=True, indent=4)
    print("Saving output to file:", out_file)
    with open(out_file, 'w') as f:
        f.write(output)
    print('Total observation num:', len(observation_dictionary))

class Courier(object):
    def __init__(self, actions):
        self.actions = actions
        self.action_num = len(self.actions)
        self.observation_dictionary = defaultdict(lambda: 0)
        self.policy = make_rand_policy(self.action_num)
        self.last_action = -1
        self.tried_actions = []
        self.last_observation = ''

    def act(self, observation):
        if self.last_observation != observation.description:
            last_observation = observation.description
            self.tried_actions = []
        else:
            self.tried_actions += [self.last_action]
        action = self.policy(observation, self.tried_actions)
        self.last_action = action
        return self.actions[action]

    def update(self, action, state, reward, done):
        self.observation_dictionary[state.description] += 1
        self.observation_dictionary[action] += 1

def run(args):
    env_wrapper = ZorkEnv()
    env = env_wrapper.start()
    actions = load_actions(args.action_file)
    print('Actions loaded:', len(actions))
    agent = Courier(actions)
    total_steps = 0
    start_time = timeit.default_timer()
    for i in range(args.episode_num):
        state = env.reset()
        for t in itertools.count():
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(action, next_state, reward, done)
            #print('action:', action)
            #print('ob:', next_state.description)
            if done or t >= args.max_step_num:
                total_steps += (t+1)
                break
            state = next_state
    print('Running time:', timeit.default_timer() - start_time, 'Seconds')
    print('Total steps:', total_steps)
    save_observations(agent.observation_dictionary, args.output)

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    run(args)
