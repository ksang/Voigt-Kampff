import json

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
    def __init__(self, nodes={}, links={}):
        self.nodes = nodes
        self.links = links

    def fromfile(self, filename):
        with open(filename, 'r') as f:
            data = f.read()
            topo = json.loads(data)
            self.nodes = topo["nodes"]
            self.links = topo["links"]

    def fromlist(self, nodes, pairs):
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

    def path_rtl(self, path):
        """
        Return the round-trip latency(RTL) for the path given.
        path:   a list of tuple that contains src and dst node labels
        """
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
        node = path[-1][1]
        neighbors = []
        for l in self.links:
            if l[0] == node:
                neighbors += [l[1]]
            if l[1] == node:
                neighbors += [l[0]]
        return self.path_rtl(path), neighbors
