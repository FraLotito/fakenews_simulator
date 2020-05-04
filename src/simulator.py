from .network import Network, Node
import math
from queue import PriorityQueue
from random import expovariate, shuffle, uniform
import copy


class Simulator:
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const):
        self.N = N_common + N_influencers
        self.engagement_news = 1
        self.network = Network(N_common, N_influencers, N_interests, random_const, random_phy_const)
        self.sim_network = None
        self.events_queue = PriorityQueue()

        """
        print("Computing initial avg..")
        s = 0
        for i in range(self.N):
            s += self.network.nodes[i].score
        print("AVG: {}".format(s / self.N))

        self.simulate()

        print("Computing ending avg..")
        s = 0
        for i in range(self.N):
            s += self.network.nodes[i].score
            #print(self.network.nodes[i].score)
        print("AVG: {}".format(s / self.N))
        """

    def first_population_queue(self, worst_node):
        idx_0 = worst_node
        self.events_queue.put((0, idx_0))
        t = 0
        order = []
        for i in range(self.N):
            if idx_0 != i:
                order.append(i)
        shuffle(order)
        for i in range(self.N - 1):
            t += expovariate(1/3)
            self.events_queue.put((t, order[i]))

    def find_worst_node(self):
        worst = -1
        worst_id = -1
        for i in self.sim_network.nodes.keys():
            if self.sim_network.nodes[i].score > worst:
                worst = self.sim_network.nodes[i].score
                worst_id = i

        return worst_id, worst

    def simulate(self, max_time):
        self.sim_network = copy.deepcopy(self.network)
        worst_node, _ = self.find_worst_node()
        self.first_population_queue(worst_node)

        hist_status = []
        # Add simulation checkpoint
        for i in range(0, max_time, 50):
            self.events_queue.put((i, -1))

        time = 0
        print("WORST NODE: ID {}, DEG: {}".format(worst_node, len(self.sim_network.nodes[worst_node].adj)))
        while time < max_time:
            t, node_id = self.events_queue.get()
            time = t

            if node_id == -1:  # checkpoint
                hist_status.append(copy.deepcopy(self.sim_network))
                continue

            status = self.sim_network.nodes[node_id].update() or worst_node == node_id
            #print("VISITED NODE: {}, {}".format(node_id, status)) 
            if status:
                score = self.sim_network.nodes[node_id].score
                #print("SCORE: {}".format(score))
                #print("EDGES: {}".format(self.network.nodes[node_id].adj))
                for edge in self.sim_network.nodes[node_id].adj:
                    threshold = abs(score)
                    p = uniform(0, 1)
                    if p < threshold:
                        dest = edge.dest
                        weight = edge.weight
                        self.propagate(dest, score, weight)
            self.events_queue.put((time + expovariate(1/3), node_id))

        return hist_status

    def propagate(self, dest, score, weight):
        if score > 0:
            En = self.engagement_news
        else:
            En = self.engagement_news * 0.5
        message = En * score * weight
        if message < - 1:
            message = -1
        elif message > 1:
            message = 1
        self.sim_network.nodes[dest].message_queue.append(message)
