import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.terminate
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self, shape=(7, 10), winds=np.zeros(0), terminate=(3, 7)):
        self._change(shape, winds, terminate)

    def change(self, shape, winds, terminate):
        self._change(shape, winds, terminate)

    def _change(self, shape, winds, terminate):
        self.shape = shape
        self.terminate = terminate

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        if winds.size == 0:
            winds = np.zeros(self.shape)
            winds[:,[3,4,5,8]] = 1
            winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == self.terminate:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

    def render_episode(self, episode, mode='human', close=False):
        """
        episde is a list of tuple (state, reward, action)

        """
        if close:
            return
        state_action = {}
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        total_reward = 0
        for state, reward, action in episode:
            print("state: %s\treward: %s\taction: %s" % (state, reward, action))
            state_action[state] = action
            total_reward += reward
        print("total reward: %s" % total_reward)

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if s in state_action:
                action = state_action[s]
                if action == 0:
                    output = " U "
                elif action == 1:
                    output = " R "
                elif action == 2:
                    output = " D "
                elif action == 3:
                    output = " L "
                else:
                    output = " T "
            elif position == self.terminate:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
