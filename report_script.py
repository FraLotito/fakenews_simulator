from src.simulator import Simulator
from src.network import NodeType
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, Manager
import pathlib
from math import exp
from functools import partial
import pickle


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

    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    f = plt.figure()
    nx.draw(G, ax=f.add_subplot(111), pos=pos, with_labels=True, font_size=4, node_size=60, node_color=color_map,
            edge_color="grey", width=0.5, arrowsize=5)
    f.savefig("results/graph.pdf")


def draw_degree_distribution(network):
    import collections
    degree_sequence = []
    for n, node in network.nodes.items():
        degree_sequence.append(len(node.adj))

    fig, ax = plt.subplots()
    """
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    plt.bar(deg, cnt, width=0.80, color='b')
    """
    plt.hist(degree_sequence, bins=100)

    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    #ax.set_xticks([d + 0.4 for d in deg])
    #ax.set_xticklabels(deg)
    fig.savefig("results/degree.pdf")


manager = Manager()
res_queue = manager.Queue()


def process_fn(_, first_infect=None):
    s = copy.deepcopy(simulator)
    results = s.simulate(max_time, recovered_debunking, SIR, first_infect=first_infect, return_nets=False, weighted=weighted)
    res_queue.put(results)
    return True


def run_simulations(file_name):
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_fn, range(N)), total=N):
            pass

    S = []
    I = []
    R = []

    for _ in range(N):
        (s, i, r), inf_time = res_queue.get(timeout=1)
        for k in range(len(s)):
            if len(S) == 0:
                S = [0] * len(s)
                I = [0] * len(i)
                R = [0] * len(r)

            S[k] += s[k]
            I[k] += i[k]
            R[k] += r[k]

    for v in range(len(S)):
        S[v] /= N
        I[v] /= N
        R[v] /= N

    m = max(I)
    xm = np.argmax(I) * 20
    M = []
    for i in range(len(S)):
        M.append(m)

    times = [20 * i for i in range(len(S))]

    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    plt.plot(times, S, label="S")
    plt.plot(times, I, label="I")
    plt.plot(times, R, label="R")
    plt.plot(times, M, color="r", label="Max I")
    plt.axvline(x=xm, color='r', linestyle='dashed')
    plt.legend()

    label_text = 'N° simulations: {} - ' \
                 'Weighted: {} - ' \
                 'Eng: {}\n' \
                 'N° common: {} - ' \
                 'N° influencers: {} - ' \
                 'N° bots: {}'.format(N, weighted, engagament_val, simulator.network.count_node_type(NodeType.Common),
                                      simulator.network.count_node_type(NodeType.Influencer),
                                      simulator.network.count_node_type(NodeType.Bot))
    plt.title(label_text)
    plt.savefig('results/' + file_name + '.pdf')
    plt.clf()

    print("Finished simulating", file_name)


def infection_simulation(file_name):
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial(process_fn, first_infect=0), range(N_inf)), total=N_inf):
            pass

    infection_time = []

    for _ in range(N_inf):
        (s, i, r), inf_time = res_queue.get(timeout=1)
        for k in range(len(inf_time)):
            if len(infection_time) == 0:
                infection_time = [0] * len(inf_time)
            infection_time[k] += inf_time[k]

    for v in range(len(infection_time)):
        infection_time[v] /= N_inf

    G = nx.DiGraph()

    pos = {}
    nodes_lbl = {}
    for n, node in simulator.network.nodes.items():
        G.add_node(n)
        pos[n] = [node.position_x, node.position_y]
        for e in node.adj:
            nodes_lbl[e.dest] = nodes_lbl.get(e.dest, 0) + 1

    f, ax = plt.subplots()
    cmap = plt.get_cmap('Reds_r')
    nx.draw(G, ax=ax, pos=pos, with_labels=True, font_size=4, node_size=20, node_color=infection_time,
                cmap=cmap, labels=nodes_lbl)
    f.set_facecolor("gray")
    ax.axis('off')

    label_text = 'N° simulations: {} - ' \
                 'Weighted: {} - ' \
                 'Eng: {}\n' \
                 'N° common: {} - ' \
                 'N° influencers: {} - ' \
                 'N° bots: {}'.format(N_inf, weighted, engagament_val, simulator.network.count_node_type(NodeType.Common),
                                      simulator.network.count_node_type(NodeType.Influencer),
                                      simulator.network.count_node_type(NodeType.Bot))
    plt.title(label_text)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_time)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, max_time, 10),
                 boundaries=np.arange(0, max_time + 0.1, .1))

    f.savefig("results/graph_" + file_name + ".pdf", facecolor=f.get_facecolor())

    print("Finished simulating infection", file_name)


def save_net(filename):
    with open('results/' + filename, 'wb') as handle:
        pickle.dump(simulator.network, handle)


def calc_engagement(t, initial_val=1.0):
    return initial_val * exp(-1 / (max_time / 2) * t)


if __name__ == "__main__":
    const = 10

    n_common = 200 * const
    n_influencer = 3 * const
    n_bots = 3 * const
    n_interests = 5
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
    engagament_val = 1.0
    engagement_news = calc_engagement

    N = 800      # number of simulation for the SIR model
    N_inf = 100  # number of infection simulations

    simulator = Simulator(N_common=n_common, N_influencers=n_influencer, N_interests=n_interests,
                          N_bots=n_bots, engagement_news=engagement_news,
                          random_const=random_const, random_phy_const=random_phy_const,
                          recover_avg=recover_avg, recover_var=recover_var,
                          vuln_avg=vulnerability_avg, vuln_var=vulnerability_var,
                          reshare_avg=reshare_avg, reshare_var=reshare_var,
                          int_avg=interests_avg, int_var=interests_var)

    save_net('common')
    run_simulations('common')
    infection_simulation('common')

    steps = 3

    num = int(n_influencer / steps)
    for i in range(steps):
        simulator.add_influencers(num)
        name = 'influencers_' + str(num*(i+1))
        save_net(name)
        run_simulations(name)
        infection_simulation(name)

    num = int(n_bots / steps)
    for i in range(steps):
        simulator.add_bots(num)
        name = 'bots_' + str(num*(i+1))
        save_net(name)
        run_simulations(name)
        infection_simulation(name)

    weighted = True
    run_simulations('weighted_bots')
    infection_simulation('weighted_bots')

    engagament_val = 0.5
    simulator.engagement_news = partial(calc_engagement, initial_val=engagament_val)
    run_simulations('engagement_0.5')
    infection_simulation('engagement_0.5')

    engagament_val = 0.2
    simulator.engagement_news = partial(calc_engagement, initial_val=engagament_val)
    run_simulations('engagement_0.2')
    infection_simulation('engagement_0.2')

    draw_simulation_network_roles(simulator.network)
    draw_degree_distribution(simulator.network)
