from src.simulator import Simulator
from src.network import NodeType, norm_sample
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
import json


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


def draw_out_degree_distribution(network):
    degree_sequence = []
    for n, node in network.nodes.items():
        degree_sequence.append(len(node.adj))

    fig, ax = plt.subplots()
    plt.hist(degree_sequence, bins=100)

    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    plt.title("Out-degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    fig.savefig("results/out_degree.pdf")
    plt.clf()


def draw_in_degree_distribution(network):
    degree_sequence = [0] * len(network.nodes)
    for n, node in network.nodes.items():
        for edge in node.adj:
            degree_sequence[edge.dest] += 1

    fig, ax = plt.subplots()
    plt.hist(degree_sequence, bins=50)

    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    plt.title("In-degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    fig.savefig("results/in_degree.pdf")
    plt.clf()


manager = Manager()
res_queue = manager.Queue()


def process_fn(i, first_infect=None, with_temp_dyn=True):
    if first_infect is None:
        simulator = simulators[i % N_networks]
    else:
        simulator = simulators[0]
    s = copy.deepcopy(simulator)
    results = s.simulate(max_time, recovered_debunking, SIR, first_infect=first_infect, return_nets=False,
                         weighted=weighted, with_temp_dyn=with_temp_dyn)
    res_queue.put(results)
    return True


def run_simulations(file_name, first_infect=None, with_temp_dyn=True):
    N = N_sim * N_networks
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial(process_fn, first_infect=first_infect,
                                                  with_temp_dyn=with_temp_dyn), range(N)), total=N):
            pass

    S = []
    I = []
    R = []

    CDF_I = []

    for _ in range(N):
        (s, i, r, cdf_i), inf_time, rec_time = res_queue.get(timeout=1)
        for k in range(len(s)):
            if len(S) == 0:
                S = [0] * len(s)
                I = [0] * len(i)
                R = [0] * len(r)
                CDF_I = [0] * len(cdf_i)

            S[k] += s[k]
            I[k] += i[k]
            R[k] += r[k]
            CDF_I[k] += cdf_i[k]

    for v in range(len(S)):
        S[v] /= N
        I[v] /= N
        R[v] /= N
        CDF_I[v] /= N

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

    different_rate_text = ''
    if new_vuln_avg is not None:
        different_rate_text = ' - Vuln rate mean: {}'.format(new_vuln_avg)
    if new_recover_avg is not None:
        different_rate_text = ' - Recover rate mean: {}'.format(new_recover_avg)

    label_text = 'N° simulations: {} - ' \
                 'Weighted: {} - ' \
                 'Eng: {}{}\n' \
                 'N° common: {} - ' \
                 'N° influencers: {} - ' \
                 'N° bots: {} - ' \
                 'N° antibots: {}'.format(N, weighted, engagament_val, different_rate_text,
                                      simulators[0].network.count_node_type(NodeType.Common),
                                      simulators[0].network.count_node_type(NodeType.Influencer),
                                      simulators[0].network.count_node_type(NodeType.Bot),
                                      simulators[0].network.count_node_type(NodeType.Antibot))
    plt.title(label_text)
    plt.ylabel("Number of nodes")
    plt.xlabel("Simulation time")
    plt.savefig('results/' + file_name + '.pdf')
    plt.clf()

    data = {
        'S': S,
        'I': I,
        'R': R
    }
    with open('results/json/' + file_name + '.json', 'w') as outfile:
        json.dump(data, outfile)

    print("Finished simulating", file_name)
    return CDF_I[:-1], R


def plot_cdf_simulations(CDF_results):
    times = [20 * i for i in range(len(CDF_results[0]['results'][0]))]

    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    lbl = "_weighted" if weighted else ""
    lbl += "_with_tmp_dyn" if 'with_temp_dyn' in CDF_results[0] else ""
    with open('results/json/cdfs' + lbl + '.json', 'w') as outfile:
        json.dump(CDF_results, outfile)

    for res in CDF_results:
        lbl_td = "tmp dyn:{}".format(res['with_temp_dyn']) if 'with_temp_dyn' in res else ""
        plt.plot(times, res['results'][0], label="infected bots:{} antibots:{} {}".format(
            res['bots'], res['antibots'], lbl_td
        ))
        plt.plot(times, res['results'][1], label="recovered bots:{} antibots:{} {}".format(
            res['bots'], res['antibots'], lbl_td
        ))
    plt.legend()

    different_rate_text = ''
    if new_vuln_avg is not None:
        different_rate_text = ' - Vuln rate mean: {}'.format(new_vuln_avg)
    if new_recover_avg is not None:
        different_rate_text = ' - Recover rate mean: {}'.format(new_recover_avg)

    N = N_sim * N_networks
    label_text = 'CDF - N° sims: {} - ' \
                 'Weighted: {} - ' \
                 'Eng: {}{}'.format(N, weighted, engagament_val, different_rate_text)
    plt.title(label_text)
    plt.ylabel("Number of nodes")
    plt.xlabel("Simulation time")
    plt.savefig('results/CDFs' + lbl + '.pdf')
    plt.clf()

    print("Finished simulating", "CDFs")


def infection_simulation(file_name, first_infect=0, with_temp_dyn=True):
    simulator = simulators[0]
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial(process_fn, first_infect=first_infect,
                                                  with_temp_dyn=with_temp_dyn), range(N_inf)), total=N_inf):
            pass

    infection_time = []
    recovery_time = []

    for _ in range(N_inf):
        (s, i, r, _), inf_time, rec_time = res_queue.get(timeout=1)
        for k in range(len(inf_time)):
            if len(infection_time) == 0:
                infection_time = [0] * len(inf_time)
            infection_time[k] += inf_time[k]
        for k in range(len(rec_time)):
            if len(recovery_time) == 0:
                recovery_time = [0] * len(rec_time)
            recovery_time[k] += rec_time[k]

    for v in range(len(infection_time)):
        infection_time[v] /= N_inf
    for v in range(len(recovery_time)):
        recovery_time[v] /= N_inf

    def save_plot(data, title, cmap):
        with open('results/json/' + title + '.json', 'w') as outfile:
            json.dump(data, outfile)

        G = nx.DiGraph()

        pos = {}
        nodes_lbl = {}
        for n, node in simulator.network.nodes.items():
            G.add_node(n)
            pos[n] = [node.position_x, node.position_y]
            for e in node.adj:
                nodes_lbl[e.dest] = nodes_lbl.get(e.dest, 0) + 1

        f, ax = plt.subplots()
        nx.draw(G, ax=ax, pos=pos, with_labels=True, font_size=4, node_size=20, node_color=data,
                cmap=cmap, vmin=0, vmax=max_time, labels=nodes_lbl)
        ax.axis('off')

        different_rate_text = ''
        if new_vuln_avg is not None:
            different_rate_text = ' - Vuln rate mean: {}'.format(new_vuln_avg)
        if new_recover_avg is not None:
            different_rate_text = ' - Recover rate mean: {}'.format(new_recover_avg)

        label_text = 'N° sims: {} - ' \
                     'Weighted: {} - ' \
                     'Eng: {}{}\n' \
                     'N° common: {} - ' \
                     'N° inf: {} - ' \
                     'N° bots: {} - ' \
                     'N° antibots: {}'.format(N_inf, weighted, engagament_val, different_rate_text,
                                          simulator.network.count_node_type(NodeType.Common),
                                          simulator.network.count_node_type(NodeType.Influencer),
                                          simulator.network.count_node_type(NodeType.Bot),
                                          simulator.network.count_node_type(NodeType.Antibot))
        plt.title(label_text)
        norm = mpl.colors.Normalize(vmin=0, vmax=max_time)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(0, max_time, 10),
                            boundaries=np.arange(0, max_time + 0.1, .1))
        cbar.set_label('Average ' + title + ' time', rotation=270, labelpad=20)

        f.savefig("results/graph_" + title + '_' + file_name + ".pdf", facecolor=f.get_facecolor())
        plt.clf()

    save_plot(infection_time, 'infection', plt.get_cmap('plasma'))
    save_plot(recovery_time, 'recovery', plt.get_cmap('viridis'))

    print("Finished simulating infection", file_name)


def calc_avg_degrees():
    in_degree = {}  # keys are degree
    out_degree = {}
    for sim in simulators:
        in_degree_sequence = [0] * len(sim.network.nodes)
        for n, node in sim.network.nodes.items():
            out = len(node.adj)
            for edge in node.adj:
                in_degree_sequence[edge.dest] += 1
            out_degree[out] = out_degree.get(out, 0) + 1
        for inc in in_degree_sequence:
            in_degree[inc] = in_degree.get(inc, 0) + 1

    for inc in in_degree:
        in_degree[inc] /= len(simulators)
    for out in out_degree:
        out_degree[out] /= len(simulators)
    return in_degree, out_degree


def save_net(filename):
    if save_nets:
        with open('results/' + filename, 'wb') as handle:
            pickle.dump(simulators, handle)


def calc_engagement(t, initial_val=1.0):
    return initial_val * exp(-1 / (max_time / 2) * t)


def create_network(_):
    res_queue.put(Simulator(N_common=n_common, N_influencers=n_influencer, N_interests=n_interests,
                            N_bots=n_bots, engagement_news=None, N_antibots=n_antibots,
                            random_const=random_const, random_phy_const=random_phy_const,
                            recover_avg=recover_avg, recover_var=recover_var,
                            vuln_avg=vulnerability_avg, vuln_var=vulnerability_var,
                            reshare_avg=reshare_avg, reshare_var=reshare_var,
                            int_avg=interests_avg, int_var=interests_var))
    return True


if __name__ == "__main__":
    const = 10

    n_common = 200 * const
    n_influencer = 3 * const
    n_bots = 3 * const
    n_antibots = 3 * const
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
    engagement_news = partial(calc_engagement, initial_val=engagament_val)

    new_vuln_avg = None
    new_recover_avg = None

    save_nets = True

    N_sim = 8  # number of simulation for the SIR model
    N_networks = 50  # number of different networks to try
    N_inf = 100  # number of infection simulations

    simulators = []
    CDF_results = []
    degrees = []

    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(create_network, range(N_networks)), total=N_networks):
            pass

    for _ in range(N_networks):
        sim = res_queue.get(timeout=1)
        sim.engagement_news = partial(calc_engagement, initial_val=engagament_val)
        simulators.append(sim)

    save_net('common')
    CDF_r = run_simulations('common')
    """
    CDF_results.append({
        'influencers': 0,
        'bots': 0,
        'antibots': 0,
        'results': CDF_r
    })
    """
    infection_simulation('common')
    in_degree, out_degree = calc_avg_degrees()
    degrees.append({
        'influencers': 0,
        'bots': 0,
        'antibots': 0,
        'in_degree': in_degree,
        'out_degree': out_degree
    })

    steps = 3

    num = int(n_influencer / steps)
    for i in range(steps):
        for sim in simulators:
            sim.add_influencers(num)
        name = 'influencers_' + str(num * (i + 1))
        save_net(name)
        CDF_r = run_simulations(name)
        """
        CDF_results.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': 0,
            'antibots': 0,
            'results': CDF_r
        })
        """
        infection_simulation(name)
        in_degree, out_degree = calc_avg_degrees()
        degrees.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': 0,
            'antibots': 0,
            'in_degree': in_degree,
            'out_degree': out_degree
        })

    # Temporal dynamics results
    CDF_results_temp = []

    saved_10_bots_sims = []
    num = int(n_bots / steps)
    for i in range(steps):
        for sim in simulators:
            sim.add_bots(num)
        if i == 0:  # save simulators for later
            saved_10_bots_sims = copy.deepcopy(simulators)
        name = 'bots_' + str(num * (i + 1))
        save_net(name)
        CDF_r = run_simulations(name)
        """
        CDF_results.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': 0,
            'results': CDF_r
        })
        """
        infection_simulation(name)
        in_degree, out_degree = calc_avg_degrees()
        degrees.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': 0,
            'in_degree': in_degree,
            'out_degree': out_degree
        })

        CDF_results_temp.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': 0,
            'with_temp_dyn': True,
            'results': CDF_r
        })
        name += '_without_temp_dyn'
        CDF_r = run_simulations(name, with_temp_dyn=False)
        CDF_results_temp.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': 0,
            'with_temp_dyn': False,
            'results': CDF_r
        })
        break


    # run simulations with some antibots
    simulators = copy.deepcopy(saved_10_bots_sims)

    num = int(n_antibots / steps)
    for i in range(steps):
        for sim in simulators:
            sim.add_antibots(num)
        name = 'antibot_' + str(num * (i + 1))
        save_net(name)
        CDF_r = run_simulations(name)
        CDF_results.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': simulators[0].network.count_node_type(NodeType.Antibot),
            'results': CDF_r
        })
        infection_simulation(name)
        in_degree, out_degree = calc_avg_degrees()
        degrees.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': simulators[0].network.count_node_type(NodeType.Antibot),
            'in_degree': in_degree,
            'out_degree': out_degree
        })
        if i == 0:
            CDF_results_temp.append({
                'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
                'bots': simulators[0].network.count_node_type(NodeType.Bot),
                'antibots': simulators[0].network.count_node_type(NodeType.Antibot),
                'with_temp_dyn': True,
                'results': CDF_r
            })
            name += '_without_temp_dyn'
            CDF_r = run_simulations(name, with_temp_dyn=False)
            CDF_results_temp.append({
                'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
                'bots': simulators[0].network.count_node_type(NodeType.Bot),
                'antibots': simulators[0].network.count_node_type(NodeType.Antibot),
                'with_temp_dyn': False,
                'results': CDF_r
            })
            infection_simulation(name, with_temp_dyn=False)
    plot_cdf_simulations(CDF_results)
    plot_cdf_simulations(CDF_results_temp)

    # save in and out degrees to json
    with open('results/json/degrees.json', 'w') as outfile:
        json.dump(degrees, outfile)


    weighted = True
    run_simulations('weighted_bots')
    infection_simulation('weighted_bots')

    # run weighted simulations with some antibots
    simulators = copy.deepcopy(saved_10_bots_sims)
    CDF_results = []

    num = int(n_antibots / steps)
    for i in range(steps):
        for sim in simulators:
            sim.add_antibots(num)
        name = 'antibot_weighted_' + str(num * (i + 1))
        save_net(name)
        CDF_r = run_simulations(name)
        CDF_results.append({
            'influencers': simulators[0].network.count_node_type(NodeType.Influencer),
            'bots': simulators[0].network.count_node_type(NodeType.Bot),
            'antibots': simulators[0].network.count_node_type(NodeType.Antibot),
            'results': CDF_r
        })
        infection_simulation(name)
    plot_cdf_simulations(CDF_results)


    engagament_val = 0.5
    for sim in simulators:
        sim.engagement_news = partial(calc_engagement, initial_val=engagament_val)
    run_simulations('engagement_0.5')
    infection_simulation('engagement_0.5')

    engagament_val = 0.2
    for sim in simulators:
        sim.engagement_news = partial(calc_engagement, initial_val=engagament_val)
    run_simulations('engagement_0.2')
    infection_simulation('engagement_0.2')


    """
    # run an infection from the influencers with the max out-degree
    with open('report_results/influencers_10', 'rb') as handle:
        simulator.network = pickle.load(handle)
        simulator.N = len(simulator.network.nodes)
        first_influencer = list(
            filter(lambda x: simulator.network.nodes[x].type == NodeType.Influencer, simulator.network.nodes))
        first_influencer = max(first_influencer, key=lambda x: len(simulator.network.nodes[x].adj))
        run_simulations('from_influencer_max_adj', first_infect=first_influencer)
        infection_simulation('from_influencer_max_adj', first_infect=first_influencer)

    """
    # run simulations with a high vulnerability rate
    simulators = copy.deepcopy(saved_10_bots_sims)

    new_vuln_avg = 0.7
    for sim in simulators:
        for i, node in sim.network.nodes.items():
            node.vulnerability = norm_sample(avg=new_vuln_avg, var=vulnerability_var, clip1=0, clip2=1)

    run_simulations('vuln_rate_bots_10')
    infection_simulation('vuln_rate_bots_10')
    new_vuln_avg = None

    # run simulations with a high recovery rate
    simulators = copy.deepcopy(saved_10_bots_sims)

    new_recover_avg = 0.7
    for sim in simulators:
        for i, node in sim.network.nodes.items():
            node.recover_rate = norm_sample(avg=new_recover_avg, var=recover_var, clip1=0, clip2=1)

    run_simulations('recover_rate_bots_10')
    infection_simulation('recover_rate_bots_10')
    new_recover_avg = None

