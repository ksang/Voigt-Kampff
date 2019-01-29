import numpy as np
import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir)))

from envs.netgraph import NetGraphEnv

"""
Topology:
                           +---------+       +---------+       +---------+
                           |         |       |         |       |         |
                           /   104   ---------   105   ---------   106   \
                          /|         |       |         |       |         |\
                         / +---------+       +----|----+       +---------+ \
                        /                         |                         \
                       /                          |                          \
                      /                           |                           \
                     /                            |                            \
        +---------+ /                        +----|----+                        \ +---------+
        |         |/                         |         |                         \|         |
  SRC   |   101   ----------------------------   102   ----------------------------   103   |  DST
        |         |\                         |         |                         /|         |
        +---------+ \                        +----|----+                        / +---------+
                     \                            |                            /
                      \                           |                           /
                       \                          |                          /
                        \                         |                         /
                         \ +---------+       +----|----+       +---------+ /
                          \|         |       |         |       |         |/
                           \   107   ---------   108   ---------   109   /
                           |         |       |         |       |         |
                           +---------+       +---------+       +---------+
"""

def nine_nodes_topo(rand_rtl=False):
    env = NetGraphEnv()
    nodes = []
    for i in range(1,10):
        nodes += [100+i]
    pairs = [
        (101,104,50),
        (101,102,100),
        (101,107,50),
        (104,105,50),
        (105,106,50),
        (105,102,50),
        (106,103,50),
        (102,103,100),
        (102,108,50),
        (107,108,50),
        (108,109,50),
        (109,103,50)
    ]
    if rand_rtl:
        for d in l:
            d[2] = np.random.randint(1000)
    env.fromlist(nodes, pairs)
    return env
