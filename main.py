from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import networkx as nx
import sys
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from src.simulator import Simulator


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fake news simulator')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QHBoxLayout(self._main)

        # Add plt canvas
        self.figure = plt.figure()
        self.network_canvas = FigureCanvas(self.figure)
        layout.addWidget(self.network_canvas)
        self.addToolBar(NavigationToolbar(self.network_canvas, self))
        self.network_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add buttons controls
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addLayout(btn_layout)
        graph_btn = QtWidgets.QPushButton("Create graph")
        graph_btn.clicked.connect(self.draw_network)
        btn_layout.addWidget(graph_btn)

        self.simulator = None

    def draw_network(self):
        self.figure.clf()

        self.simulator = Simulator(20, 4, 0.15, 0.1)

        G = nx.Graph()

        pos = {}
        for n, node in self.simulator.network.nodes.items():
            G.add_node(n)
            pos[n] = [node.position_x, node.position_y]

        for a, node in self.simulator.network.nodes.items():
            for b in node.adj:
                G.add_edge(a, b.dest, weight=b.weight)

        nx.draw(G, pos=pos, with_labels=True)
        self.network_canvas.draw_idle()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
