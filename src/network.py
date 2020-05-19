from enum import IntEnum
import random
import math
import numpy as np


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def norm_sample(avg=0.0, var=0.4, clip1=-1, clip2=1, n=None):
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
    def __init__(self, _id, node_type: NodeType, n_interests, int_avg, int_var, recover_avg, recover_var,
                 vuln_avg, vuln_var, reshare_avg, reshare_var):
        self.id = _id
        self.type = node_type
        self.adj = []

        self.score = 0
        self.recover_rate = norm_sample(avg=recover_avg, var=recover_var, clip1=0, clip2=1)
        self.vulnerability = norm_sample(avg=vuln_avg, var=vuln_var, clip1=-1, clip2=1)
        self.reshare_rate = norm_sample(avg=reshare_avg, var=reshare_var, clip1=0, clip2=1)

        self.interests = norm_sample(avg=int_avg, var=int_var, n=n_interests, clip1=-1, clip2=1)
        
        np.append(self.interests, self.vulnerability)
        np.append(self.interests, self.recover_rate)
        np.append(self.interests, self.score)

        self.position_x = np.random.uniform(0, 1)
        self.position_y = np.random.uniform(0, 1)
        self.message_queue = []

        self.infection_time = None
        if node_type == NodeType.Bot:
            self.infection_time = 0

    def is_recovered(self):
        p = random.uniform(0, 1)
        if p < self.recover_rate:
            self.score = -1
            return True
        else:
            return False

    def get_infection_time(self, not_infect_val):
        if self.infection_time is None:
            return not_infect_val
        else:
            return self.infection_time

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

    def update_sir(self, engagement_news):
        number_of_messages = len(self.message_queue)

        if self.type == NodeType.Bot:
            return

        if number_of_messages != 0 and self.score != -1:
            can_fact_check = False

            for i in range(number_of_messages):
                p = random.uniform(0, 1)
                message_type, weight = self.message_queue[i]

                if message_type == -1:
                    k = 0.1
                else:
                    k = 1
                    can_fact_check = True

                if p < self.vulnerability * k * weight * engagement_news:
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
    def __init__(self, N_common, N_influencers, N_bots, N_interests, random_const, random_phy_const,
                 int_avg, int_var, recover_avg, recover_var, vuln_avg, vuln_var, reshare_avg, reshare_var):
        self.N_common = N_common
        self.N_influencers = N_influencers
        self.N_bots = N_bots

        self.nodes = {}
        self.available_id = 0
        self.N_interests = N_interests
        self.recover_avg, self.recover_var = recover_avg, recover_var
        self.vuln_avg, self.vuln_var = vuln_avg, vuln_var
        self.reshare_avg, self.reshare_var = reshare_avg, reshare_var
        self.int_avg, self.int_var = int_avg, int_var

        # First node that starts infection. Useful for statistics
        self.infected_node = None

    def gen_node(self, node_type):
        idx = self.available_id
        node = Node(idx, node_type, n_interests=self.N_interests, int_avg=self.int_avg, int_var=self.int_var,
                    recover_avg=self.recover_avg, recover_var=self.recover_var, vuln_avg=self.vuln_avg, vuln_var=self.vuln_var,
                    reshare_avg=self.reshare_avg, reshare_var=self.reshare_var)
        self.available_id += 1
        self.nodes[node.id] = node

        if node_type == NodeType.Bot:
            node.reshare_rate = 1

        return idx

    def generate_common(self, random_const, random_phy_const):

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
            p = random.uniform(0, 1)
            if (dist < random_const or len(edge) > 0) and p < 0.5:
                weight = prox
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

    def generate_influencers(self, random_const, random_phy_const, lim=None):

        def add_proximity_edge(idx_a, idx_b, dist, random_const):
            prox = (1 - dist)
            if dist < random_const:
                edge = list(filter(lambda x: x.dest == idx_b, self.nodes[idx_a].adj))
                weight = prox
                if len(edge) == 0:
                    self.nodes[idx_a].add_adj(Edge(idx_a, idx_b, weight))
                else:
                    weight = np.mean([weight, edge[0].weight])
                    edge[0].weight = weight

        n = 0
        if lim is None:
            lim = self.N_influencers

        while n < lim:
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
                add_proximity_edge(b, idx, dist, random_const)

                phys_dist = self.nodes[idx].compute_physical_distance(self.nodes[b])
                add_proximity_edge(b, idx, phys_dist * 0.5, random_phy_const * 0.5)

            n += 1

    def generate_bots(self, lim=None):
        n = 0
        if lim is None:
            lim = self.N_bots

        while n < lim:
            idx = self.gen_node(NodeType.Bot)
            self.nodes[idx].score = 1

            for b in self.nodes.keys():
                if self.nodes[b].type == NodeType.Bot:
                    continue
                
                p = random.uniform(0, 1)
                if p < 0.02:
                    weight = 0.1
                    self.nodes[idx].add_adj(Edge(idx, b, weight))
                    self.nodes[idx].add_adj(Edge(b, idx, weight))
            
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

    def count_score_equal(self, low, up):
        return len(list(filter(lambda n: low < self.nodes[n].score <= up, self.nodes)))
