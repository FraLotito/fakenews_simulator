from .network import Network, Node
import math
from queue import PriorityQueue
from random import expovariate, shuffle, uniform


class Simulator:
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const):
        self.N = N_common + N_influencers
        self.engagement_news = 1
        self.network = Network(N_common, N_influencers, N_interests, random_const, random_phy_const)
        self.events_queue = PriorityQueue()

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

    def first_population_queue(self, worst_node):
        idx_0 = worst_node
        self.events_queue.put((0, idx_0))
        #print("{} {}".format(0, idx_0))
        t = 0
        order = []
        for i in range(self.N):
            if idx_0 != i:
                order.append(i)
        shuffle(order)
        for i in range(self.N - 1):
            t += expovariate(1/3)
            self.events_queue.put((t, order[i]))
            #print("{} {}".format(t, order[i]))

    def find_worst_node(self):
        worst = -1
        worst_id = -1
        for i in self.network.nodes.keys():
            if self.network.nodes[i].score > worst:
                worst = self.network.nodes[i].score
                worst_id = i

        return worst_id, worst


    def simulate(self):
        worst_node, _ = self.find_worst_node()
        self.first_population_queue(worst_node)
        time = 0
        while time < 1000:
            t, node_id = self.events_queue.get()
            time = t
            status = self.network.nodes[node_id].update() or worst_node == node_id 
            if status:
                score = self.network.nodes[node_id].score
                for edge in self.network.nodes[node_id].adj:
                    threshold = abs(score)
                    p = uniform(0, 1)
                    if p < threshold:
                        dest = edge.dest
                        weight = edge.weight
                        self.propagate(dest, score, weight)
            self.events_queue.put((t + expovariate(1/4), node_id))
            
            

    def propagate(self, dest, score, weight):
        if score > 0:
            En = self.engagement_news
        else:
            En = self.engagement_news * 0.5
        message = En * score * weight
        #print(message)
        self.network.nodes[dest].message_queue.append(message)

