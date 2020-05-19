from network import Network, Node, NodeType
import math
from queue import PriorityQueue
from random import expovariate, shuffle, uniform, randint
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm


class Simulator:
    def __init__(self, N_common, N_influencers, N_bots, N_interests, random_const, random_phy_const, engagement_news,
                 int_avg, int_var, recover_avg, recover_var, vuln_avg, vuln_var, reshare_avg, reshare_var,
                 weighted=True):

        self.N_common = N_common
        self.N_influencers = N_influencers
        self.N_bots = N_bots
        self.random_const = random_const
        self.random_phy_const = random_phy_const
        self.engagement_news = engagement_news
        self.network = Network(N_common=N_common, N_influencers=N_influencers, N_bots=N_bots, N_interests=N_interests,
                               random_const=random_const, random_phy_const=random_phy_const, int_avg=int_avg,
                               int_var=int_var,
                               recover_avg=recover_avg, recover_var=recover_var, vuln_avg=vuln_avg, vuln_var=vuln_var,
                               reshare_avg=reshare_avg, reshare_var=reshare_var, weighted=weighted)

        self.network.generate_common(self.random_const, self.random_phy_const)
        self.N = len(self.network.nodes)
        self.sim_network = None

    def add_influencers(self, n):
        self.network.generate_influencers(self.random_const, self.random_phy_const, n)
        self.N = len(self.network.nodes)

    def add_bots(self, n):
        self.network.generate_bots(n)
        self.N = len(self.network.nodes)

    def first_population_queue(self, first_infect):
        order = []
        self.events_queue.put((0, first_infect))
        for i in range(self.N):
            if i != first_infect:
                order.append(i)
        shuffle(order)
        for i in range(self.N - 1):
            self.events_queue.put((expovariate(1 / 16), order[i]))

    def initial_infection(self, first_infect):
        if first_infect is None:
            first_infect = randint(0, self.N_common - 1)
        self.sim_network.nodes[first_infect].score = 1
        self.sim_network.nodes[first_infect].reshare_rate = 1
        self.sim_network.nodes[first_infect].recover_rate = 0
        return first_infect

    def simulate(self, max_time, recovered_debunking=False, SIR=False, first_infect=None):
        self.events_queue = PriorityQueue()
        self.sim_network = copy.deepcopy(self.network)
        first_infect = self.initial_infection(first_infect)
        self.sim_network.infected_node = first_infect
        self.first_population_queue(first_infect)

        # Add simulation checkpoint
        for i in range(0, max_time, 20):
            self.events_queue.put((i, -1))

        s = []
        i = []
        r = []

        time = 0
        while time < max_time:
            t, node_id = self.events_queue.get()

            time = t

            if node_id == -1:  # checkpoint
                S = 0
                I = 0
                R = 0
                for k in self.sim_network.nodes.keys():
                    if self.sim_network.nodes[k].type == NodeType.Common:
                        if self.sim_network.nodes[k].score == 1:
                            I += 1
                        elif self.sim_network.nodes[k].score == 0:
                            S += 1
                        else:
                            R += 1
                s.append(S)
                i.append(I)
                r.append(R)
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
                self.events_queue.put((time + expovariate(1 / 4), node_id))
            else:
                self.events_queue.put((time + expovariate(1 / 16), node_id))

        return s, i, r

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


def draw_simulation_network_roles(network):
    G = nx.DiGraph()

    pos = {}
    color_map = []
    for n, node in network.nodes.items():
        G.add_node(n)
        pos[n] = [node.position_x, node.position_y]

        if node.type == NodeType.Common:
            color_map.append("skyblue")
        elif node.type == NodeType.Influencer:
            color_map.append("gold")
        elif node.type == NodeType.Bot:
            color_map.append("red")

    edge_labels = []
    for a, node in network.nodes.items():
        for b in node.adj:
            G.add_edge(a, b.dest)
            edge_labels.append(b)

    f = plt.figure()
    nx.draw(G, ax=f.add_subplot(111), pos=pos, with_labels=True, font_size=4, node_size=60, node_color=color_map,
            edge_color="grey", width=0.5, arrowsize=5)
    f.savefig("graph.pdf")


def draw_degree_distribution(network):
    import collections
    degree_sequence = []
    for n, node in network.nodes.items():
        degree_sequence.append(len(node.adj))

    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    fig.savefig("degree.pdf")


from multiprocessing import Pool, Manager

res_queue = Manager().Queue()


def process_fn(_):
    s = copy.deepcopy(simulator)
    res_queue.put(s.simulate(max_time, recovered_debunking, SIR, first_infect=None))
    return True


def run_simulations(file_name):
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_fn, range(N)), total=N):
            pass

    print("Finished simulating", file_name)

    S = []
    I = []
    R = []

    for i in tqdm(range(N)):
        s, i, r = res_queue.get(timeout=1)
        for k in range(len(s)):
            if len(S) < len(s):
                S.append(0)
                I.append(0)
                R.append(0)
            S[k] += s[k]
            I[k] += i[k]
            R[k] += r[k]

    for v in range(len(S)):
        S[v] /= N
        I[v] /= N
        R[v] /= N

    print("Finished norm data", file_name)

    m = max(I)
    xm = np.argmax(I) * 20
    M = []
    for i in range(len(S)):
        M.append(m)

    times = [20 * i for i in range(len(S))]

    plt.plot(times, S, label="S")
    plt.plot(times, I, label="I")
    plt.plot(times, R, label="R")
    plt.plot(times, M, color="r", label="Max I")
    plt.axvline(x=xm, color='r', linestyle='dashed')
    plt.legend()
    plt.savefig(file_name + '.pdf')
    plt.clf()

    print("Saved file", file_name)


if __name__ == "__main__":
    const = 10

    n_common = 200 * const
    n_influencer = 3 * const
    n_bots = 3 * const
    n_interests = 5
    engagement_news = 1.0
    vulnerability_avg = 0.5
    vulnerability_var = 0.2
    reshare_avg = 0.5
    reshare_var = 0.2
    recover_avg = 0.2
    recover_var = 0.2
    interests_avg = 0
    interests_var = 0.4
    random_const = (0.1 + 0.02 * const) / const
    random_phy_const = (0.1 + 0.02 * const) / const
    SIR = True
    weighted = False
    recovered_debunking = True
    max_time = 5000

    N = 300

    simulator = Simulator(N_common=n_common, N_influencers=n_influencer, N_interests=n_interests,
                          N_bots=n_bots, engagement_news=engagement_news,
                          random_const=random_const, random_phy_const=random_phy_const,
                          recover_avg=recover_avg, recover_var=recover_var,
                          vuln_avg=vulnerability_avg, vuln_var=vulnerability_var,
                          reshare_avg=reshare_avg, reshare_var=reshare_var,
                          int_avg=interests_avg, int_var=interests_var, weighted=weighted)

    run_simulations('common')

    simulator.add_influencers(n_influencer)
    run_simulations('influencers')

    simulator.add_bots(n_bots)
    run_simulations('bots')

    draw_simulation_network_roles(simulator.network)
    draw_degree_distribution(simulator.network)
