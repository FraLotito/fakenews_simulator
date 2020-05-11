from enum import IntEnum
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from igraph import *


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def norm_sample(avg=0, var=0.4, clip1=-1, clip2=1, n=None):
    """
    Sample from a normal distribution centered in 0.5. Results are limited in [0, 1]
    """
    norm_vals = np.random.normal(avg, var, n)  # sample from a normal distribution
    return np.clip(norm_vals, clip1, clip2)  # limit the results into [0, 1]


class NodeType(IntEnum):
    Common = 0,
    Conspirator = 1,
    Influencer = 2,
    Debunker = 3,
    Bot = 4


class Edge:
    def __init__(self, start, dest, weight):
        self.start = start
        self.dest = dest
        self.weight = weight


class Node:
    def __init__(self, _id, node_type: NodeType, n_interests, score_avg, score_var, int_avg, int_var):
        self.id = _id
        self.type = node_type
        self.adj = []

        #self.score = norm_sample(avg=score_avg, var=score_var)
        self.score = 0
        # TODO: convert to parameter
        self.recover_rate = norm_sample(avg=0.2, var=0.2, clip1=0, clip2=1)
        #self.recover_rate = 0.05
        self.vulnerability = norm_sample(avg=score_avg, var=score_var)
        self.reshare_rate = norm_sample(avg=score_avg, var=score_var, clip1=0, clip2=1)

        self.interests = norm_sample(avg=int_avg, var=int_var, n=n_interests, clip1=-1, clip2=1)
        
        np.append(self.interests, self.vulnerability)
        np.append(self.interests, self.recover_rate)
        np.append(self.interests, self.score)

        self.position_x = np.random.uniform(0, 1)
        self.position_y = np.random.uniform(0, 1)
        self.message_queue = []

    def is_recovered(self):
        p = random.uniform(0, 1)
        if p < self.recover_rate:
            self.score = -1
            return True
        else:
            return False

    def update(self):
        s = sum(self.message_queue)
        l = len(self.message_queue)
        self.message_queue = []
        if l > 0:
            avg = s / l
            self.score += avg
            if self.score > 1:
                self.score = 1
            elif self.score < - 1:
                self.score = -1
            return True
        else:
            return False

    def update_sir(self):
        number_of_messages = len(self.message_queue)

        if self.type == NodeType.Bot:
            return

        if number_of_messages != 0 and self.score != -1:
            can_fact_check = False

            for i in range(number_of_messages):
                p = random.uniform(0, 1)
                message_type = self.message_queue[i]

                if message_type == -1:
                    k = 0.2
                else:
                    k = 1
                    can_fact_check = True

                if p < self.vulnerability * k:
                    self.score = message_type
                    can_fact_check = False
                    
            if can_fact_check and self.score == 0:
                self.is_recovered()

        self.message_queue = []
        

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
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const,
                 score_avg, score_var, int_avg, int_var, weighted=True):
        self.N_common = N_common
        self.N_influencers = N_influencers
        self.is_weighted = weighted
        self.nodes = {}
        self.available_id = 0
        self.N_interests = N_interests
        self.score_avg = score_avg
        self.score_var = score_var
        self.int_avg = int_avg
        self.int_var = int_var
        self.generate_common(random_const, random_phy_const)
        self.generate_influencers(random_const, random_phy_const)

        self.generate_bots(3)


        # First node that starts infection. Useful for statistics
        self.infected_node = None

    def gen_node(self, node_type):
        idx = self.available_id
        node = Node(idx, node_type, n_interests=self.N_interests, score_avg=self.score_avg, score_var=self.score_var,
                    int_avg=self.int_avg, int_var=self.int_var)
        self.available_id += 1
        self.nodes[node.id] = node
        return idx

    def generate_common(self, random_const, random_phy_const):

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
            if dist < random_const or len(edge) > 0:
                if self.is_weighted:
                    weight = prox
                else:
                    weight = 1
                if len(edge) == 0:
                    self.nodes[idx_a].add_adj(Edge(idx_a, idx_b, weight))
                    self.nodes[idx_b].add_adj(Edge(idx_b, idx_a, weight))
                else:
                    weight = np.min([weight, edge[0].weight])
                    edge_b = list(filter(lambda x: x.dest == idx_a, self.nodes[idx_b].adj))
                    edge[0].weight = weight
                    edge_b[0].weight = weight

        n = 0
        while n < self.N_common:
            idx = self.gen_node(NodeType.Common)

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


    # TODO: fare attenzione alle random_const (ci sono costanti moltiplicative)

    def generate_influencers(self, random_const, random_phy_const):

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            if dist < random_const:
                edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
                if self.is_weighted:
                    weight = prox
                else:
                    weight = 1
                if len(edge) == 0:
                    self.nodes[idx_a].add_adj(Edge(idx_a, idx_b, weight))
                else:
                    weight = np.mean([weight, edge[0].weight])
                    edge[0].weight = weight

        n = 0
        while n < self.N_influencers:
            idx = self.gen_node(NodeType.Influencer)

            for b in self.nodes.keys():
                if idx == b:
                    continue
                dist = self.nodes[idx].compute_distance(self.nodes[b])
                add_proximity_edge(idx, b, dist, random_const * 2)

                phys_dist = self.nodes[idx].compute_physical_distance(self.nodes[b])
                add_proximity_edge(idx, b, phys_dist, random_phy_const * 2)

            for b in self.nodes.keys():
                if idx == b:
                    continue
                dist = self.nodes[idx].compute_distance(self.nodes[b])
                add_proximity_edge(b, idx, dist, random_const * 0.5)

                phys_dist = self.nodes[idx].compute_physical_distance(self.nodes[b])
                add_proximity_edge(b, idx, phys_dist, random_phy_const * 0.5)

            """
            print("OUT: {}".format(len(self.nodes[idx].adj)))
            cont = 0
            for i in self.nodes.keys():
                for j in self.nodes[i].adj:
                    if j.dest == idx:
                        cont += 1

            print("IN: {}".format(cont))
            """

            n += 1


    def generate_bots(self, N_bots):
        n = 0

        while n < N_bots:
            idx = self.gen_node(NodeType.Bot)
            self.nodes[idx].score = 1

            for b in self.nodes.keys():
                if self.nodes[b].type == NodeType.Bot:
                    continue
                
                p = random.uniform(0, 1)
                if p < 0.1:
                    if self.is_weighted:
                        weight = 0.1
                    else:
                        weight = 1
                    self.nodes[idx].add_adj(Edge(idx, b, weight))
                    self.nodes[idx].add_adj(Edge(b, idx, weight))



            print("OUT: {}".format(len(self.nodes[idx].adj)))
            cont = 0
            for i in self.nodes.keys():
                for j in self.nodes[i].adj:
                    if j.dest == idx:
                        cont += 1

            print("IN: {}".format(cont))
            

            n += 1

    def average_score(self):
        tot = 0
        for n in self.nodes:
            tot += self.nodes[n].score
        return tot / len(self.nodes)

    def average_weight(self):
        tot = 0
        for n in self.nodes:
            weights = list(map(lambda edge: edge.weight, self.nodes[n].adj))
            if len(weights) == 0:
                tot += 0
            else:
                tot += sum(weights) / len(weights)
        return tot / len(self.nodes)

    def count_score_equal(self, value):
        return len(list(filter(lambda n: self.nodes[n].score == value, self.nodes)))
