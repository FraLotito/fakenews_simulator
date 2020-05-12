from .network import Network, Node, NodeType
import math
from queue import PriorityQueue
from random import expovariate, shuffle, uniform, randint
import copy
import pickle

class Simulator:
    def __init__(self, N_common, N_influencers, N_bots, N_interests, random_const, random_phy_const, engagement_news,
                 int_avg, int_var, recover_avg, recover_var, vuln_avg, vuln_var, reshare_avg, reshare_var, weighted=True):
        
        self.N_common = N_common
        self.N = N_common + N_influencers + N_bots
        self.engagement_news = engagement_news
        self.network = Network(N_common=N_common, N_influencers=N_influencers, N_bots=N_bots, N_interests=N_interests,
                               random_const=random_const, random_phy_const=random_phy_const, int_avg=int_avg, int_var=int_var,
                               recover_avg=recover_avg, recover_var=recover_var, vuln_avg=vuln_avg, vuln_var=vuln_var,
                               reshare_avg=reshare_avg, reshare_var=reshare_var, weighted=weighted)

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
            self.events_queue.put((expovariate(1/16), order[i]))

    def initial_infection(self, first_infect):
        if first_infect is None:
            first_infect = randint(0, self.N_common-1)
        self.sim_network.nodes[first_infect].score = 1
        self.sim_network.nodes[first_infect].reshare_rate = 1
        self.sim_network.nodes[first_infect].recover_rate = 0
        return first_infect

    def simulate(self, max_time, recovered_debunking=False, SIR=False, first_infect=None):
        self.sim_network = copy.deepcopy(self.network)
        first_infect = self.initial_infection(first_infect)
        self.sim_network.infected_node = first_infect
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
                self.sim_network.nodes[node_id].update()
            else:
                self.sim_network.nodes[node_id].update_sir()

            score = self.sim_network.nodes[node_id].score

            if score == 1 or (score == -1 and recovered_debunking):
                reshare_rate = self.sim_network.nodes[node_id].reshare_rate
                for edge in self.sim_network.nodes[node_id].adj:
                    p = uniform(0, 1)
                    dest = edge.dest
                    weight = edge.weight
                    if p < reshare_rate and dest != first_infect:
                        self.propagate(dest, score, weight, SIR=SIR)

            # se un nodo è un bot, allora si collega più spesso
            if self.sim_network.nodes[node_id].type == NodeType.Bot:
                self.events_queue.put((time + expovariate(1/4), node_id))
            else:
                self.events_queue.put((time + expovariate(1/16), node_id))

        return hist_status

    def propagate(self, dest, score, weight, SIR):
        if SIR:
            self.sim_network.nodes[dest].message_queue.append(score)
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
