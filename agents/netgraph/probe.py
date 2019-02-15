# Probe agent to discover and gather performance metrics
# from NetGraph env


# Pseudo code for exploration:
#    Init netgraph environment env
#    Init stack node_q
#    Init LINKS and NODES dictionary for saving results
#    node_q.push((gateway, empty path))
#
#    while node_q is not empty:
#        node, path = node_q.pop()
#        next_nodes = env.neighbors(node)
#        for next_node in next_nodes:
#            if next_node in NODES:
#                continue
#            next_path = path + (node, next_node)
#            node_q.push((next_node, next_path))
#            LINKS[(node, next_node)] = env.path_rtl(next_path) - env.path_rtl(path)
#        NODES[node] = True

import argparse
import itertools
from collections import deque
from nine import nine_nodes_topo

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument('-t', '--topo', default='nine',
                        help='Toplogy name to run.')
cmd_parser.add_argument('-r', '--random_gateway', default=False, action='store_true',
                        help='Randomly pick a node in toplogy to start with.')
cmd_parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Verbosely output.')


class Probe(object):
    """
    Probe agent that interating with netgraph.
    Probe doesn't aware of the entire topology, it starts from its neighbor node (gateway)
    and explores network, saves nodes/links into its own database.
    """
    def __init__(self, env):
        self.env = env
        self.nodes = {}
        self.links = {}
        self.gate = self.env.hello()

    def __link_key(self, pair):
        if pair[0] > pair[1]:
            return (pair[1], pair[0])
        return pair

    def explore(self, verbose=False):
        node_q = deque()
        gate_neib = self.env.neighbors(self.gate)
        for neib in gate_neib:
            path = [(self.gate, neib)]
            node_q.append((neib, path))
            self.links[self.__link_key(path[0])] = self.env.path_rtl(path)
        self.nodes[self.gate] = True
        if verbose:
            print("Init:")
            self.render_net()
        for t in itertools.count():
            if len(node_q) == 0: break
            if verbose:
                print("Step:", t+1)
            self.step_explore(node_q, verbose)
            if verbose:
                self.render_net()

    def step_explore(self, node_q, verbose=False):
        node, path = node_q.pop()
        srtl, neighbors = self.env.path_rtl_neighbors(path)
        if verbose:
            print("Path RTL Neighbors:", path)
        for neib in neighbors:
            if neib in self.nodes:
                continue
            p = path + [(node, neib)]
            node_q.append((neib, p))
            drtl = self.env.path_rtl(p)
            if verbose:
                print("Path RTL:", p)
            self.links[self.__link_key(p[-1])] = drtl - srtl
        self.nodes[node] = True

    def render_net(self):
        print("Gateway:", self.gate)
        print(len(self.nodes), "Nodes:")
        for n in self.nodes.keys():
            print("  ", n)
        print(len(self.links), "Links:")
        for l in self.links.keys():
            print("  ", l, "RTL:", self.links[l])

def run(args):
    if args.topo == 'nine':
        env = nine_nodes_topo(args.random_gateway)
    else:
        raise NotImplementedError
    agent = Probe(env)
    agent.explore(args.verbose)
    print("Final result:")
    agent.render_net()
    env.close()

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    run(args)
