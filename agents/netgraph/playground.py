from nine import nine_nodes_topo

env = nine_nodes_topo()

test_paths = [
    [(101, 102), (102, 103)],
    [(101, 107), (107, 108), (108, 109), (109, 103)]
]

for path in test_paths:
    print('Gate:', env.hello())
    print('Gate neighbor:', env.neighbors(env.hello()))
    print('Path:', path)
    print('Path RTL:', env.path_rtl(path))
    print('Path RTL neighbors:', env.path_rtl_neighbors(path))
