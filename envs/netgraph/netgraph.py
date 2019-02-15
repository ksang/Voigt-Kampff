import json
import numpy as np

def parse_link(pair):
    if pair[0] < pair[1]:
        return pair
    return (pair[1], pair[0])

class NetGraphEnv(object):
    """
    NetGraph is an environment that simulates a MPLS-like network topology.
    The nodes in network have unique labels assigned to them。
    Network topology is an undirected cyclic graph.
    """
    def __init__(self, nodes={}, links={}, gate=None):
        self.nodes = nodes
        self.links = links
        self.gate = gate
        self.packet_count = 0

    def fromfile(self, filename):
        with open(filename, 'r') as f:
            data = f.read()
            topo = json.loads(data)
            self.nodes = topo["nodes"]
            self.links = topo["links"]
            self.gate = topo.get("gate")
            if self.gate is None:
                self.gate = np.random.choice(list(self.nodes.keys()))

    def fromlist(self, nodes, pairs, gate=None):
        ns = {}
        for n in nodes:
            ns[n] = True
        ls = {}
        for d in pairs:
            assert d[0] in ns and d[1] in ns
            link = parse_link((d[0], d[1]))
            ls[link] = d[2]
        self.nodes = ns
        self.links = ls
        self.gate = gate
        if self.gate is None:
            self.gate = np.random.choice(nodes)

    def close(self):
        print("env closed, packet count:", self.packet_count)

    def hello(self):
        """
        Return the link peer label id in the toplogy.
        This is the start point of exploring the network.
        """
        self.packet_count += 1
        return self.gate

    def neighbors(self, node):
        """
        Return the neighbor label list of a node.
        """
        self.packet_count += 1
        neighbors = []
        for l in self.links:
            if l[0] == node:
                neighbors += [l[1]]
            if l[1] == node:
                neighbors += [l[0]]
        return neighbors

    def path_rtl(self, path):
        """
        Return the round-trip latency(RTL) for the path given.
        path:   a list of tuple that contains src and dst node labels
        """
        self.packet_count += 1
        path_rtl = 0
        for pair in path:
            link = parse_link(pair)
            rtl = self.links.get(link)
            assert rtl is not None, link
            path_rtl += rtl
        return path_rtl

    def path_rtl_neighbors(self, path):
        """
        Return the round-trip latency(RTL) for the path given and
        neighbors of last hop in the path
        path:   a list of tuple that contains src and dst node labels
        """
        self.packet_count += 1
        node = path[-1][1]
        neighbors = []
        for l in self.links:
            if l[0] == node:
                neighbors += [l[1]]
            if l[1] == node:
                neighbors += [l[0]]
        return self.path_rtl(path), neighbors
