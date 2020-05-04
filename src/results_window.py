from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


class ResultsWindow(QtWidgets.QMainWindow):
    def __init__(self, sim_results, *args, **kwargs):
        super(ResultsWindow, self).__init__(*args, **kwargs)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        self.setWindowTitle('Simulation results')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QGridLayout(self._main)

        simulation_time = [net[0] for net in sim_results]

        lbl = QLabel('Average node score')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        layout.addWidget(lbl, 0, 0)
        score_figure = plt.figure()
        score_canvas = FigureCanvas(score_figure)
        layout.addWidget(score_canvas, 1, 0)
        scores = [net[1].average_score() for net in sim_results]
        plt.plot(simulation_time, scores)
        plt.ylim(-1, 1)
        plt.xlabel("Simulation time")
        plt.ylabel("Average node score")

        res_layout = QtWidgets.QGridLayout()
        layout.addLayout(res_layout, 1, 1)

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
        layout.addWidget(lbl, 2, 0)
        scores_start_figure = plt.figure()
        scores_start_canvas = FigureCanvas(scores_start_figure)
        layout.addWidget(scores_start_canvas, 3, 0)
        scores_start = list(map(lambda n: sim_results[0][1].nodes[n].score, sim_results[0][1].nodes))
        plt.hist(scores_start)
        plt.xlabel("Node scores")

        lbl = QLabel('Distribution of the node scores at the end:')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        layout.addWidget(lbl, 2, 1)
        scores_end_figure = plt.figure()
        scores_end_canvas = FigureCanvas(scores_end_figure)
        layout.addWidget(scores_end_canvas, 3, 1)
        scores_end = list(map(lambda n: sim_results[-1][1].nodes[n].score, sim_results[-1][1].nodes))
        plt.hist(scores_end)
        plt.xlabel("Node scores")
