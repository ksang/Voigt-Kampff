import sys,os
import logging
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.zork import ZorkEnv

env_wrapper = ZorkEnv()
env = env_wrapper.start()
env.reset()
env.render()
