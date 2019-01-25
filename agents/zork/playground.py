import sys,os
import logging
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.zork import ZorkEnv

def print_state(state, indention=' '):
    members = [
        'state.command',
        'state.description',
        'state.feedback',
        'state.game_ended',
        'state.has_lost',
        'state.has_won',
        'state.init',
        'state.inventory',
        'state.location',
        'state.max_score',
        'state.nb_deaths',
        'state.nb_moves',
        'state.previous_state',
        'state.score',
        'state.update'
    ]
    for member in members:
        try:
            print(indention, member+":", eval(member))
        except NotImplementedError as e:
            print(indention, member+":", 'NotImplemented')

env_wrapper = ZorkEnv()
env = env_wrapper.start()
state = env.reset()
env.render()

print_state(state)

print("Tak action: open mailbox")
state, reward, done = env.step('open mailbox')
print(state)
print_state(state)
