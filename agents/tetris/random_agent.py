import sys,os
import time
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.tetris import Game
from envs.tetris import GameGUI


cmd_parser = argparse.ArgumentParser(description=None)
# Data arguments
cmd_parser.add_argument('-g', '--gui', default=False, action='store_true',
                        help='Enable GUI display.')
cmd_parser.add_argument('-i', '--interval', default=100, type=int,
                        help='Action interval in ms, only in GUI mode.')
cmd_parser.add_argument('-n', '--num', default=100, type=int,
                        help='Number of steps to take.')
cmd_parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Number of steps to take.')

def print_board(board, nq):
    print("="*20)
    print("Board:")
    print(board)
    print("Next queue:")
    print(nq)

def one_step(env, interval, verbose):
    action = env.action_space.sample()
    (board, next_queue), score, done = env.step(action)
    if verbose:
        print_board(board, next_queue)
        print("Score: %d, Action %d, Done: %s" % (score, action, done))
    if done:
        print_board(board, next_queue)
        return score, done
    if interval > 0:
        time.sleep(interval/1000)
    return score, done

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    if args.gui:
        g = GameGUI(mode='agent')
        g.play()
        for t in range(args.num):
            score, done = one_step(g.tetris, args.interval, args.verbose)
            g.update_window()
            if done:
                break
        g.close()
    else:
        env = Game()
        for t in range(args.num):
            score, done = one_step(env, 0, args.verbose)
            if done:
                break

    print("Score: %d, Steps: %d, Done: %s" % (score, t+1, done))
