from .network import Network, Node
import math
from queue import PriorityQueue
from random import expovariate, shuffle, uniform
import copy


class Simulator:
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const, engagement_news,
                 score_avg, score_var, int_avg, int_var, weighted=True):
        self.N = N_common + N_influencers
        self.engagement_news = engagement_news
        self.network = Network(N_common=N_common, N_influencers=N_influencers, N_interests=N_interests,
                               random_const=random_const, random_phy_const=random_phy_const, score_avg=score_avg,
                               score_var=score_var, int_avg=int_avg, int_var=int_var, weighted=weighted)
        self.sim_network = None
        self.events_queue = PriorityQueue()

    def first_population_queue(self, first_infect):
        order = []
        self.events_queue.put((0, first_infect))
        for i in range(self.N):
            if i != first_infect:
                order.append(i)
        shuffle(order)
        for i in range(self.N - 1):
            self.events_queue.put((expovariate(1/4), order[i]))

    def initial_infection(self):
        import random
        infect = random.randint(0, self.N-1)
        self.sim_network.nodes[infect].score = 1
        self.sim_network.nodes[infect].reshare_rate = 1
        self.sim_network.nodes[infect].recover_rate = 0
        return infect

    def simulate(self, max_time, SIR=False):
        self.sim_network = copy.deepcopy(self.network)
        first_infect = self.initial_infection()
        self.first_population_queue(first_infect)

        hist_status = []
        # Add simulation checkpoint
        for i in range(0, max_time, 50):
            self.events_queue.put((i, -1))

        time = 0
        while time < max_time:
            t, node_id = self.events_queue.get()
            
            time = t

            if node_id == -1:  # checkpoint
                hist_status.append((time, copy.deepcopy(self.sim_network)))
                continue
            
            if not SIR:
                status = self.sim_network.nodes[node_id].update()
            else:
                if self.sim_network.nodes[node_id].score == 1:
                    if self.sim_network.nodes[node_id].is_recovered():
                        print("recovered")
                        self.sim_network.nodes[node_id].score = -1
                        status = False
                    else:
                        status = True
                else:
                    status = False
            if status:
                reshare_rate = self.sim_network.nodes[node_id].reshare_rate
                score = self.sim_network.nodes[node_id].score
                for edge in self.sim_network.nodes[node_id].adj:
                    p = uniform(0, 1)
                    if p < reshare_rate:
                        dest = edge.dest
                        weight = edge.weight
                        self.propagate(dest, score, weight, SIR=SIR)
            self.events_queue.put((time + expovariate(1/4), node_id))

        return hist_status

    def propagate(self, dest, score, weight, SIR):
        if SIR:
            if self.sim_network.nodes[dest].score == 0:
                p = uniform(0, 1)
                if p < self.sim_network.nodes[dest].vulnerability:
                    self.sim_network.nodes[dest].score = 1
        else:
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
