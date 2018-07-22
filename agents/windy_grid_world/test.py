import gym
import numpy as np
import sys,os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.grid_world import WindyGridworldEnv

env = WindyGridworldEnv()

print(env.reset())
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()

print(env.step(3))
env.render()

print(env.step(0))
env.render()

new_shape = (8, 8)
w = np.zeros(new_shape)
w[:,[3,4]] = 1
w[:,[6,7]] = 3

print("changing")
env.change(new_shape, w, (5,5))

print(env.step(0))
env.render()
