from .network import Network, Node
import math
from queue import PriorityQueue
from random import expovariate, shuffle


class Simulator:
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const):
        self.N = N_common + N_influencers
        self.network = Network(N_common, N_influencers, N_interests, random_const, random_phy_const)
        self.events_queue = PriorityQueue()
        self.simulate()

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
        worst_node, worst_value = self.find_worst_node())
        self.first_population_queue(worst_node)

    def propagate(self, a : Node, b : Node):
        En = 1
        d = 0.9
        mb = 1
        alpha = 1
        beta = 2
        return En * a.education_rate * ((alpha * d + beta * mb) / (alpha + beta))

if __name__ == "__main__":
    pass
    #a = Node(1, 1, 5)
    #b = Node(1, 1, 5)
    #print(irradiate(a, b))