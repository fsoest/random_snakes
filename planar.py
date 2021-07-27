import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx
from random_snakes.snek import diff


def plot_edges(nodes, edges, c='k'):
    node_1 = nodes[edges[0]]
    node_2 = nodes[edges[1]]
    x_1 = node_1[0]
    x_2 = node_2[0]
    y_1 = node_1[1]
    y_2 = node_2[1]
    plt.plot(np.array([x_1, x_2]), np.array([y_1, y_2]), c=c)


def make_planar_graph(n_points):
    a = [0, 1, -1]
    n_points = 10
    points = np.random.uniform(0, 1, (n_points, 2))
    periodic = np.zeros((9 * n_points, 2))

    k = 0
    for i in a:
        for j in a:
            shift = np.zeros((n_points, 2))
            shift[:, 0] = points[:, 0] + i
            shift[:, 1] = points[:, 1] + j
            periodic[k * n_points: (k + 1) * n_points] = shift
            k += 1
    tri = Delaunay(periodic)
    simplices = []
    for simplex in tri.simplices:
        if min(simplex) < n_points:
            simplices.append(simplex)
    simplices = np.array(simplices) % n_points
    simplices = np.sort(simplices, axis=1)
    simplices = np.unique(simplices, axis=0)

    edges = []
    for simplex in simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[0], simplex[2]))
    edges = np.unique(np.array(edges), axis=0)

    # Create the networkx Graph
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(np.arange(n_points))
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=np.linalg.norm(diff(points[edge[0]], points[edge[1]], 1)))
    return G, points

if __name__ == '__main__':
    a = [0, 1, -1]
    n_points = 10
    np.random.seed(42)
    points = np.random.uniform(0, 1, (n_points, 2))
    periodic = np.zeros((9 * n_points, 2))

    k = 0
    for i in a:
        for j in a:
            shift = np.zeros((n_points, 2))
            shift[:, 0] = points[:, 0] + i
            shift[:, 1] = points[:, 1] + j
            periodic[k * n_points: (k+1) * n_points] = shift
            k += 1

    plt.plot(periodic[:, 0], periodic[:, 1], '.')
    tri = Delaunay(periodic)
    plt.triplot(periodic[:, 0], periodic[:, 1], tri.simplices)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    simplices = []

    for simplex in tri.simplices:
        if min(simplex) < n_points:
            simplices.append(simplex)
    simplices = np.array(simplices) % n_points
    simplices = np.sort(simplices, axis=1)
    simplices = np.unique(simplices, axis=0)

    edges = []
    for simplex in simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[0], simplex[2]))
    edges = np.unique(np.array(edges), axis=0)

    for edge in edges:
        plot_edges(points, edge)

    plt.show()

    # Create the networkx Graph
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(np.arange(n_points))
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=np.linalg.norm(diff(points[edge[0]], points[edge[1]], 1)))
    nx.draw(G)
    plt.show()
