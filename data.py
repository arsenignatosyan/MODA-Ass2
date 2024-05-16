import numpy as np
import networkx as nx

def create_data():
  return np.random.randint([1,1,1,1,50], [6,6,6,6,121])

def main():
    G = nx.triangular_lattice_graph(6,7)
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    node_dict = {}
    num = 1
    for n in nodes:
        node_dict[n] = num
        num += 1
    
    edge_list = []
    for e in edges:
        edge_list.append((node_dict[e[0]], node_dict[e[1]]))

    graph = nx.from_edgelist(edge_list)

    edge_data = {}

    for e in edge_list:
        d = create_data()
        edge_data[(e[0], e[1])] = d
        edge_data[(e[1], e[0])] = d

    paths = list(nx.all_simple_paths(graph, 1, 32, cutoff=10))

    data = []
    for p in paths:
        path_data = edge_data[(p[0],p[1])].copy()
        for i in range(1, len(p)-1):
            path_data += edge_data[(p[i], p[i+1])]
        path_data[:4] = (path_data[:4] * 10) / (len(p) - 1)
        path_data = list(path_data)
        path_data.append(p)
        data.append(path_data)


    output = np.array(data)
    np.save("output.npy", output)

if __name__ == '__main__':
    main()