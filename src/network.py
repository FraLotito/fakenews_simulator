from enum import Enum
import random
import math
import numpy as np

from igraph import *


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def norm_sample(n=None):
    """
    Sample from a normal distribution centered in 0.5. Results are limited in [0, 1]
    """
    norm_vals = np.random.normal(0.5, 0.2, n)  # sample from a normal distribution
    return np.clip(norm_vals, 0, 1)  # limit the results into [0, 1]


class NodeType(Enum):
    Common = 0,
    Conspirator = 1,
    Influencer = 2,
    Debunker = 3,
    Media = 4


class Edge:
    def __init__(self, dest, weight):
        self.dest = dest
        self.weight = weight


class Node:
    def __init__(self, _id, node_type: NodeType, n_interests):
        self.id = _id
        self.type = node_type
        self.adj = []

        self.interests = norm_sample(n_interests)

        self.education_rate = norm_sample()
        self.reshare_rate = norm_sample()

        self.position_x = np.random.uniform(0, 1)
        self.position_y = np.random.uniform(0, 1)

    def add_adj(self, edge):
        self.adj.append(edge)

    def compute_distance(self, node_b):
        return euclidean_distance(self.interests, node_b.interests) / math.sqrt(len(self.interests))

    def compute_physical_distance(self, node_b):
        a = np.array([self.position_x, self.position_y])
        b = np.array([node_b.position_x, node_b.position_y])
        return euclidean_distance(a, b) / math.sqrt(2)

    def __str__(self):
        return '{} - deg: {}\n' \
               '\tinterests: {}\n' \
               '\teducation_rt: {}, reshare_rt: {}\n' \
               '\tx: {} y: {}'.format(
                self.id, len(self.adj), self.interests, self.education_rate, self.reshare_rate,
                self.position_x, self.position_y)


class Network:
    def __init__(self, N_nodes, n_interests, random_const, random_phy_const):
        self.N_nodes = N_nodes
        self.nodes = {}
        self.available_id = 0
        self.n_interests = n_interests
        self.generate(random_const, random_phy_const)

    def gen_node(self, node_type):
        idx = self.available_id
        node = Node(idx, node_type, self.n_interests)
        self.available_id += 1
        self.nodes[node.id] = node
        return idx

    def generate(self, random_const, random_phy_const):
        g = Graph()

        g.add_vertices(self.N_nodes)
        g.es["weight"] = 1.0

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            p = random.uniform(0, 1)
            if p < prox * random_const:
                edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
                weight = prox
                if len(edge) == 0:
                    self.nodes[idx_a].add_adj(Edge(idx_b, prox))
                    self.nodes[idx_b].add_adj(Edge(idx_a, prox))
                    g.add_edges([(idx_a, idx_b)])
                else:
                    weight = np.mean([weight, edge[0].weight])
                    edge_b = list(filter(lambda x: x.dest == idx_a, self.nodes[idx_b].adj))
                    edge[0].weight = weight
                    edge_b[0].weight = weight
                g[idx_a, idx_b] = weight

        n = 0
        while n < self.N_nodes:
            idx = self.gen_node(NodeType.Common)

            for b in self.nodes.keys():
                if idx == b:
                    continue
                dist = self.nodes[idx].compute_distance(self.nodes[b])
                add_proximity_edge(idx, b, dist, random_const)

                phys_dist = self.nodes[idx].compute_physical_distance(self.nodes[b])
                add_proximity_edge(idx, b, phys_dist, random_phy_const)
            n += 1

        clusters = g.community_multilevel()
        member = clusters.membership
        new_cmap = ['#' + ''.join([random.choice('0123456789abcdef') for x in range(6)]) for z in range(len(clusters))]

        vcolors = {v: new_cmap[i] for i, c in enumerate(clusters) for v in c}
        g.vs["color"] = [vcolors[v] for v in g.vs.indices]

        g.vs["label"] = [v for v in g.vs.indices]
        g.es["label"] = np.around(g.es["weight"], decimals=1)

        for i in range(self.N_nodes):
            print(self.nodes[i])

        plot(g, layout='circle')


if __name__ == "__main__":
    a = Network(20, 4, random_const=0.15, random_phy_const=0.1)
