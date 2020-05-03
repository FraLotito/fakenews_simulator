from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import networkx as nx
import mplcursors
import sys
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from src.network import NodeType
from src.simulator import Simulator


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fake news simulator')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        # Add parameters grid
        self.init_parameters()
        layout.addWidget(self.create_parameters_grid())

        # Main layout: plot and buttons
        main_layout = QtWidgets.QHBoxLayout(self._main)
        layout.addLayout(main_layout)

        # Add plot canvas
        self.figure = plt.figure()
        self.network_canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.network_canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.network_canvas, self))
        self.network_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add buttons controls
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setAlignment(QtCore.Qt.AlignTop)
        main_layout.addLayout(btn_layout)
        graph_btn = QtWidgets.QPushButton("Create graph")
        graph_btn.clicked.connect(self.draw_network)
        btn_layout.addWidget(graph_btn)

        # Add node color legend
        btn_layout.addWidget(self.create_node_legend())

        self.simulator = None

    def init_parameters(self):
        self.n_common = 100
        self.n_influencer = 5

    def create_parameters_grid(self):
        param_groupbox = QtWidgets.QGroupBox("Simulation parameters:")

        par_layout = QtWidgets.QGridLayout()

        par_layout.addWidget(QLabel('Number of common:'), 0, 0)
        self.n_common_ui = QLineEdit()
        self.n_common_ui.setValidator(QtGui.QIntValidator())
        self.n_common_ui.setAlignment(QtCore.Qt.AlignRight)
        self.n_common_ui.setText(str(self.n_common))
        par_layout.addWidget(self.n_common_ui, 0, 1)

        par_layout.addWidget(QLabel('Number of influencers:'), 0, 2)
        self.n_influencer_ui = QLineEdit()
        self.n_influencer_ui.setValidator(QtGui.QIntValidator())
        self.n_influencer_ui.setAlignment(QtCore.Qt.AlignRight)
        self.n_influencer_ui.setText(str(self.n_influencer))
        par_layout.addWidget(self.n_influencer_ui, 0, 3)

        param_groupbox.setLayout(par_layout)
        return param_groupbox

    def create_node_legend(self):
        groupbox = QtWidgets.QGroupBox("Node legend:")

        layout = QtWidgets.QVBoxLayout()

        for type, color in self.node_color.items():
            lbl = QLabel(str(type).split('.')[1])
            sample_palette = QtGui.QPalette()
            color_qt = QtGui.QColor()
            color_qt.setNamedColor(color)
            sample_palette.setColor(QtGui.QPalette.Window, color_qt)

            lbl.setAutoFillBackground(True)
            lbl.setPalette(sample_palette)
            layout.addWidget(lbl)

        groupbox.setLayout(layout)
        return groupbox

    node_color = {
        NodeType.Common: "skyblue",
        NodeType.Influencer: "lightgreen",
        NodeType.Conspirator: "tomato",
        NodeType.Debunker: "gold",
        NodeType.Media: "plum"
    }

    def draw_network(self):
        self.figure.clf()

        self.n_common = int(self.n_common_ui.text())
        self.n_influencer = int(self.n_influencer_ui.text())
        self.simulator = Simulator(N_common=self.n_common, N_influencers=self.n_influencer, N_interests=4,
                                   random_const=0.1, random_phy_const=0.1)

        G = nx.DiGraph()

        pos = {}
        color_map = []
        for n, node in self.simulator.network.nodes.items():
            G.add_node(n, node=node)
            pos[n] = [node.position_x, node.position_y]
            color_map.append(self.node_color[node.type])

        for a, node in self.simulator.network.nodes.items():
            for b in node.adj:
                G.add_edge(a, b.dest, edge=b)

        nx.draw(G, pos=pos, with_labels=True, font_size=8, node_size=150, node_color=color_map, edge_color="grey")

        def annotate(sel):
            if not isinstance(sel.target.index, tuple):
                node = G.nodes[sel.target.index]['node']
                node_type = str(node.type).split('.')[1]
                return sel.annotation.set_text("Node: {}\nType: {}".format(sel.target.index, node_type))
            else:
                edge = list(G.edges.data())[sel.target.index[0]]
                return sel.annotation.set_text("{} - {}\nWeight: {:.3f}".format(edge[0], edge[1], edge[2]['edge'].weight))

        mplcursors.cursor(hover=True).connect("add", annotate)

        self.network_canvas.draw_idle()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
