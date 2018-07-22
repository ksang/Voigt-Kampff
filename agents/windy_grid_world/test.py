import gym
import numpy as np
import sys,os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.grid_world import WindyGridworldEnv

env = WindyGridworldEnv()
episode = []
actions = [1, 2, 3, 0]

state = env.reset()
reward = 0
for a in actions:
    episode.append((state, reward, a))
    env.render()
    state, reward, _, _ = env.step(a)

episode.append((state, reward, a))
env.render_episode(episode)

new_shape = (8, 8)
w = np.zeros(new_shape)
w[:,[3,4]] = 1
w[:,[6,7]] = 3

print("change env:")
env.change(new_shape, w, (5,5))

print(env.step(0))
env.render()
