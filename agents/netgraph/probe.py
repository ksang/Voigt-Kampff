# Probe agent to discover and gather performance metrics
# from NetGraph env
import argparse
from collections import deque
from nine import nine_nodes_topo

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument('-t', '--topo', default='nine',
                        help='Toplogy name to run.')
cmd_parser.add_argument('-r', '--random_gateway', default=False, action='store_true',
                        help='Randomly pick a node in toplogy to start with.')


class Probe(object):
    def __init__(self, env):
        self.env = env
        self.nodes = {}
        self.links = {}
        self.gate = self.env.hello()

    def __link_key(self, pair):
        if pair[0] > pair[1]:
            return (pair[1], pair[0])
        return pair

    def explore(self):
        node_q = deque()
        gate_neib = self.env.neighbors(self.gate)
        for neib in gate_neib:
            path = [(self.gate, neib)]
            node_q.append((neib, path))
            self.links[self.__link_key(path[0])] = self.env.path_rtl(path)
        self.nodes[self.gate] = True
        while len(node_q) > 0:
            node, path = node_q.pop()
            srtl, neighbors = self.env.path_rtl_neighbors(path)
            for neib in neighbors:
                if neib in self.nodes:
                    continue
                p = path + [(node, neib)]
                node_q.append((neib, p))
                drtl = self.env.path_rtl(p)
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
    agent.explore()
    agent.render_net()

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    run(args)
