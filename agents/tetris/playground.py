import sys,os
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.tetris import Game

def print_board(board, nq):
    print("="*20)
    print("Board:")
    print(board)
    print("Next queue:")
    print(nq)

env = Game()
(initial_board, initial_next_queue) = env.reset()
print_board(initial_board, initial_next_queue)
for t in range(10):
    action = env.action_space.sample()
    (board, next_queue), score, done = env.step(action)
    print_board(board, next_queue)
    print("Score: %d, Action %d" % (score, action))
