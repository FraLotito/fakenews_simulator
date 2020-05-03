from enum import IntEnum
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from igraph import *


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def norm_sample(avg=0, var=0.4, n=None):
    """
    Sample from a normal distribution centered in 0.5. Results are limited in [0, 1]
    """
    norm_vals = np.random.normal(avg, var, n)  # sample from a normal distribution
    return np.clip(norm_vals, -1, 1)  # limit the results into [0, 1]


class NodeType(IntEnum):
    Common = 0,
    Conspirator = 1,
    Influencer = 2,
    Debunker = 3,
    Media = 4


class Edge:
    def __init__(self, start, dest, weight):
        self.start = start
        self.dest = dest
        self.weight = weight


class Node:
    def __init__(self, _id, node_type: NodeType, n_interests):
        self.id = _id
        self.type = node_type
        self.adj = []

        self.score = norm_sample(avg=0, var=0.4)
        self.initial_score = self.score

        self.interests = norm_sample(n=n_interests)
        np.append(self.interests, self.score)

        self.position_x = np.random.uniform(0, 1)
        self.position_y = np.random.uniform(0, 1)

        self.message_queue = []

    def visit_queue(self):
        s = sum(self.message_queue)
        l = len(self.message_queue)
        self.message_queue = []
        if l > 0:
            return s / l
        else:
            return 0

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
               '\tscore: {}\n' \
               '\tx: {} y: {}'.format(
                self.id, len(self.adj), self.interests, self.score,
                self.position_x, self.position_y)


class Network:
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const):
        self.N_common = N_common
        self.N_influencers = N_influencers
        self.nodes = {}
        self.available_id = 0
        self.N_interests = N_interests
        self.g = Graph(directed=True)
        self.generate_common(random_const, random_phy_const)
        self.generate_influencers(random_const*2, random_phy_const*2)

    def gen_node(self, node_type):
        idx = self.available_id
        node = Node(idx, node_type, self.N_interests)
        self.available_id += 1
        self.nodes[node.id] = node
        return idx

    def generate_common(self, random_const, random_phy_const):
        self.g.add_vertices(self.N_common)
        self.g.es["weight"] = 1.0

        layout = []

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
            if dist < random_const or len(edge) > 0:
                weight = prox
                if len(edge) == 0:
                    self.nodes[idx_a].add_adj(Edge(idx_a, idx_b, prox))
                    self.nodes[idx_b].add_adj(Edge(idx_b, idx_a, prox))
                    self.g.add_edges([(idx_a, idx_b)])
                else:
                    weight = np.min([weight, edge[0].weight])
                    edge_b = list(filter(lambda x: x.dest == idx_a, self.nodes[idx_b].adj))
                    edge[0].weight = weight
                    edge_b[0].weight = weight
                self.g[idx_a, idx_b] = weight

        n = 0
        while n < self.N_common:
            idx = self.gen_node(NodeType.Common)
            layout.append((self.nodes[idx].position_x, self.nodes[idx].position_y))

            for b in self.nodes.keys():
                if idx == b:
                    continue
                phys_dist = self.nodes[idx].compute_physical_distance(self.nodes[b])
                add_proximity_edge(idx, b, phys_dist, random_phy_const)
            n += 1

        for a in self.nodes.keys():
            for b in self.nodes.keys():
                if a == b:
                    continue
                dist = self.nodes[a].compute_distance(self.nodes[b])
                add_proximity_edge(a, b, dist, random_const)

    def generate_influencers(self, random_const, random_phy_const):
        self.g.add_vertices(self.N_influencers)

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            if dist < random_const:
                edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
                weight = prox
                if len(edge) == 0:
                    self.nodes[idx_a].add_adj(Edge(idx_a, idx_b, prox))
                    #self.nodes[idx_b].add_adj(Edge(idx_a, prox))
                    self.g.add_edges([(idx_a, idx_b)])
                else:
                    weight = np.mean([weight, edge[0].weight])
                    edge_b = list(filter(lambda x: x.dest == idx_a, self.nodes[idx_b].adj))
                    edge[0].weight = weight
                    #edge_b[0].weight = weight
                self.g[idx_a, idx_b] = weight

        n = 0
        while n < self.N_influencers:
            idx = self.gen_node(NodeType.Influencer)

            for b in self.nodes.keys():
                if idx == b:
                    continue
                dist = self.nodes[idx].compute_distance(self.nodes[b])
                add_proximity_edge(idx, b, dist, random_const)

                phys_dist = self.nodes[idx].compute_physical_distance(self.nodes[b])
                add_proximity_edge(idx, b, phys_dist, random_phy_const)
            n += 1
        
    def plot(self):
        #clusters = self.g.community_multilevel()
        new_cmap = ['#' + ''.join([random.choice('0123456789abcdef') for x in range(6)]) for z in range(5)]

        vcolors = {key: new_cmap[int(self.nodes[key].type)] for key in self.nodes.keys()}
        self.g.vs["color"] = [vcolors[v] for v in self.g.vs.indices]

        self.g.vs["label"] = [v for v in self.g.vs.indices]
        self.g.es["label"] = np.around(self.g.es["weight"], decimals=1)

        deg = []
        layout = []
        score = []
        for i in range(self.N_common + self.N_influencers):
            print(self.nodes[i])
            layout.append((self.nodes[i].position_x, self.nodes[i].position_y))
            deg.append(len(self.nodes[i].adj))
            score.append(self.nodes[i].score)
        print("AVG DEG: {}".format(sum(deg) / len(deg)))

        #plt.hist(deg)
        #plt.show()
        #plot(self.g, layout=Layout(layout))
        cont = 0
        for i in score:
            if i > 0.8:
                print(i)
            elif i < -0.8:
                print(i)
            elif i < 0.5 and i > -0.5:
                cont+=1
        print(cont)


if __name__ == "__main__":
    a = Network(100, 5, 5, random_const=0.1, random_phy_const=0.1)
    a.plot()
