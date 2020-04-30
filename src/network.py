from enum import Enum
import random
import math

from igraph import *


def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


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
    def __init__(self, _id, node_type: NodeType):
        self.id = _id
        self.type = node_type
        self.a1 = random.triangular(0, 1, 0.5)
        self.a2 = random.triangular(0, 1, 0.5)
        self.adj = []

    def add_adj(self, edge):
        self.adj.append(edge)

    def compute_distance(self, node_b):
        a = (self.a1, self.a2)
        b = (node_b.a1, node_b.a2)
        return euclidean_distance(a, b) / math.sqrt(2)


class Network:
    def __init__(self, N_nodes, random_const):
        self.N_nodes = N_nodes
        self.nodes = {}
        self.available_id = 0
        self.generate(random_const)

    def gen_node(self, node_type):
        idx = self.available_id
        node = Node(idx, node_type)
        self.available_id += 1
        self.nodes[node.id] = node
        return idx

    def generate(self, random_const):
        g = Graph()

        g.add_vertices(self.N_nodes)

        n = 0
        while n < self.N_nodes:
            idx = self.gen_node(NodeType.Common)

            for b in self.nodes.keys():
                if idx == b:
                    continue
                dist = self.nodes[idx].compute_distance(self.nodes[b])
                prox = (1 - dist)
                p = random.uniform(0, 1)
                if p < prox * random_const:
                    self.nodes[idx].add_adj(Edge(b, 1))
                    self.nodes[b].add_adj(Edge(idx, 1))
                    g.add_edges([(idx, b)])
            n += 1

        plot(g)
        for i in range(self.N_nodes):
            print("a1: {}, a2: {}, deg: {}".format(self.nodes[i].a1, self.nodes[i].a2, len(self.nodes[i].adj)))



if __name__ == "__main__":
    a = Network(100, 0.2)

