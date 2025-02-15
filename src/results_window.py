from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import networkx as nx


class ResultsWindow(QtWidgets.QMainWindow):
    def __init__(self, sim_results, n_sim_results, SIR=False, *args, **kwargs):
        super(ResultsWindow, self).__init__(*args, **kwargs)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        self.setWindowTitle('Simulation results')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QGridLayout(self._main)

        self.SIR = SIR

        if sim_results is not None:  # single simulation
            self.plot_sim_results(sim_results)
        else:
            self.plot_n_sim_results(n_sim_results)

    def plot_sim_results(self, sim_results):
        simulation_time = [net[0] for net in sim_results]

        lbl = QLabel('SIR plot')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.layout.addWidget(lbl, 0, 0)
        score_figure = plt.figure()
        score_canvas = FigureCanvas(score_figure)
        self.layout.addWidget(score_canvas, 1, 0)
        infected = [net[1].count_score_equal(0.4, 1) for net in sim_results]
        neutral = [net[1].count_score_equal(-0.4, 0.4) for net in sim_results]
        recovered = [net[1].count_score_equal(-1.1, -0.4) for net in sim_results]
        plt.plot(simulation_time, infected, label="Infected")
        plt.plot(simulation_time, neutral, label="Neutral")
        plt.plot(simulation_time, recovered, label="Recovered")
        plt.legend()

        res_layout = QtWidgets.QGridLayout()
        self.layout.addLayout(res_layout, 1, 1)

        lbl = QLabel('Average weight per node:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        res_layout.addWidget(lbl, 0, 0)
        res_layout.addWidget(QLabel("{:.3f}".format(sim_results[0][1].average_weight())), 0, 1)

        lbl = QLabel('Worst node score:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        res_layout.addWidget(lbl, 1, 0)
        worst_node_idx = max(sim_results[0][1].nodes, key=lambda n: sim_results[0][1].nodes[n].score)
        worst_node = sim_results[0][1].nodes[worst_node_idx]
        res_layout.addWidget(QLabel("{:.3f}".format(worst_node.score)), 1, 1)

        lbl = QLabel('Worst node degree:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        res_layout.addWidget(lbl, 2, 0)
        res_layout.addWidget(QLabel("{}".format(len(worst_node.adj))), 2, 1)

        lbl = QLabel('Distribution of the node scores at the start:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.layout.addWidget(lbl, 2, 0)
        scores_start_figure = plt.figure()
        scores_start_canvas = FigureCanvas(scores_start_figure)
        self.layout.addWidget(scores_start_canvas, 3, 0)
        scores_start = list(map(lambda n: sim_results[0][1].nodes[n].score, sim_results[0][1].nodes))
        plt.hist(scores_start)
        plt.xlabel("Node scores")

        lbl = QLabel('Distribution of the node scores at the end:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.layout.addWidget(lbl, 2, 1)
        scores_end_figure = plt.figure()
        scores_end_canvas = FigureCanvas(scores_end_figure)
        self.layout.addWidget(scores_end_canvas, 3, 1)
        scores_end = list(map(lambda n: sim_results[-1][1].nodes[n].score, sim_results[-1][1].nodes))
        plt.hist(scores_end)
        plt.xlabel("Node scores")

    def plot_n_sim_results(self, n_sim_results):
        simulation_time = [net[0] for net in n_sim_results[0]]

        n = len(n_sim_results)

        def mean(vals):
            return sum(vals) / len(vals)

        lbl = QLabel('SIR plot with {} simulations'.format(n))
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.layout.addWidget(lbl, 0, 0)
        score_figure = plt.figure()
        score_canvas = FigureCanvas(score_figure)
        self.layout.addWidget(score_canvas, 1, 0)
        infected = [[sim[i][1].count_score_equal(0.4, 1) for sim in n_sim_results] for i in range(len(simulation_time))]
        infected = [mean(score) for score in infected]
        neutral = [[sim[i][1].count_score_equal(-0.4, 0.4) for sim in n_sim_results] for i in range(len(simulation_time))]
        neutral = [mean(score) for score in neutral]
        recovered = [[sim[i][1].count_score_equal(-1.1, -0.4) for sim in n_sim_results] for i in range(len(simulation_time))]
        recovered = [mean(score) for score in recovered]
        plt.plot(simulation_time, infected, label="Infected")
        plt.plot(simulation_time, neutral, label="Neutral")
        plt.plot(simulation_time, recovered, label="Recovered")
        plt.legend()

        res_layout = QtWidgets.QGridLayout()
        self.layout.addLayout(res_layout, 1, 1)

        lbl = QLabel('Average weight per node:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        res_layout.addWidget(lbl, 0, 0)
        res_layout.addWidget(QLabel("{:.3f}".format(n_sim_results[0][0][1].average_weight())), 0, 1)

        lbl = QLabel('Worst node score:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        res_layout.addWidget(lbl, 1, 0)
        worst_node_idx = max(n_sim_results[0][0][1].nodes, key=lambda n: n_sim_results[0][0][1].nodes[n].score)
        worst_node = n_sim_results[0][0][1].nodes[worst_node_idx]
        res_layout.addWidget(QLabel("{:.3f}".format(worst_node.score)), 1, 1)

        lbl = QLabel('Worst node degree:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        res_layout.addWidget(lbl, 2, 0)
        res_layout.addWidget(QLabel("{}".format(len(worst_node.adj))), 2, 1)

        lbl = QLabel('Distribution of the node scores at the start:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.layout.addWidget(lbl, 2, 0)
        scores_start_figure = plt.figure()
        scores_start_canvas = FigureCanvas(scores_start_figure)
        self.layout.addWidget(scores_start_canvas, 3, 0)
        scores_start = list(map(lambda n: n_sim_results[0][0][1].nodes[n].score, n_sim_results[0][0][1].nodes))
        plt.hist(scores_start)
        plt.xlabel("Node scores")

        lbl = QLabel('Average distribution of the node scores at the end of {} simulations:'.format(n))
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.layout.addWidget(lbl, 2, 1)
        scores_end_figure = plt.figure()
        scores_end_canvas = FigureCanvas(scores_end_figure)
        self.layout.addWidget(scores_end_canvas, 3, 1)
        scores_end = [
            mean(list(map(lambda n: sim[-1][1].nodes[n].score, sim[-1][1].nodes))) for sim in n_sim_results
        ]
        plt.hist(scores_end)
        plt.xlabel("Node scores")

        score_per_node_figure = plt.figure()
        score_per_node_canvas = FigureCanvas(score_per_node_figure)
        self.layout.addWidget(score_per_node_canvas, 1, 2)

        G = nx.DiGraph()

        pos = {}
        color_map = []
        net = n_sim_results[0][0][1]
        for n, node in net.nodes.items():
            G.add_node(n)
            pos[n] = [node.position_x, node.position_y]

            not_infected_val = simulation_time[-1] * 3
            mean_inf_time = mean([sim[-1][1].nodes[n].get_infection_time(not_infected_val) for sim in n_sim_results])
            color_map.append(-mean_inf_time)

        print(color_map)
        print("min", min(color_map))
        print("max", max(color_map))
        print("mean", mean(color_map))

        for a, node in net.nodes.items():
            for b in node.adj:
                G.add_edge(a, b.dest)

        plt.figure(score_per_node_figure.number)
        nx.draw(G, pos=pos, with_labels=True, font_size=8, node_size=150, edge_color="grey", node_color=color_map,
                cmap=plt.cm.Reds)
        score_per_node_canvas.draw_idle()
