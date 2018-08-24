import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys,os
import logging
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.grid_world import WindyGridworldEnv

from collections import defaultdict
from libs import plotting

matplotlib.style.use('ggplot')

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('q_learning')

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, start_idx, stats, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        state = env.reset()
        if i_episode % 20 == 0:
            logger.warning('Episode %d', i_episode)
        for t in range(2000):
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            alt_action = np.argmax(Q[next_state])
            # update Q values
            delta = alpha * (reward + discount_factor * Q[next_state][alt_action] - Q[state][action])
            Q[state][action] += delta
            # update stats
            stats.episode_lengths[start_idx+i_episode] += 1
            stats.episode_rewards[start_idx+i_episode] += reward
            if done:
                break
            state = next_state

    return Q

def gen_episode(env, Q):
    """
    Generates episode experience [(state, reward, action)] from Q function

    Args:
        env: OpenAI environment.
        Q: action-value function, a dictionary mapping state -> action values.

    Returns:
        episode: experience under Q, a list of (state, reward, action)
    """
    state = env.reset()
    action = np.argmax(Q[state])
    reward = 0
    episode = [(state, reward, action)]
    for t in itertools.count():
        state, reward, done, _ = env.step(action)
        action = np.argmax(Q[state])
        if done:
            episode.append((state, reward, -1))
            break
        episode.append((state, reward, action))
    return episode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-x', type=int, default=7)
    parser.add_argument('-y', type=int, default=10)
    args = parser.parse_args()

    env = WindyGridworldEnv(shape=(args.x, args.y), terminate=(args.x-3, args.y-1))
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(500),
        episode_rewards=np.zeros(500))

    Q = q_learning(env, 500, 0, stats)

    #episode = gen_episode(env, Q)
    #env.render_episode(episode)

    '''
    new_shape = (8, 8)
    w = np.zeros(new_shape)
    w[:,[3,4]] = 1
    w[:,[6,7]] = 3

    print("changing env")
    env.change(new_shape, w, (5,5))
    Q = q_learning(env, 500, 500, stats)
    print("After change:")
    print(Q)
    episode = gen_episode(env, Q)
    env.render_episode(episode)
    '''
    # plot statistics
    plotting.plot_episode_stats(stats)
