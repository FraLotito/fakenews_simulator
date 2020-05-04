from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from mpldatacursor import datacursor
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from .network import NodeType
from .simulator import Simulator
from .results_window import ResultsWindow


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fake news simulator')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        # Add parameters grids
        self.init_parameters()
        layout.addWidget(self.create_net_parameters_grid())
        layout.addWidget(self.create_sim_parameters_grid())

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
        graph_btn = QtWidgets.QPushButton("Create network")
        graph_btn.clicked.connect(self.create_network)
        btn_layout.addWidget(graph_btn)
        load_btn = QtWidgets.QPushButton("Load network")
        load_btn.clicked.connect(self.load_network)
        btn_layout.addWidget(load_btn)
        save_btn = QtWidgets.QPushButton("Save network")
        save_btn.clicked.connect(self.save_network)
        btn_layout.addWidget(save_btn)

        btn_layout.addWidget(QHLine())

        run_btn = QtWidgets.QPushButton("Run one simulation")
        run_btn.clicked.connect(self.run_simulation)
        btn_layout.addWidget(run_btn)
        self.progress_bar = QtWidgets.QProgressBar()
        btn_layout.addWidget(self.progress_bar)
        results_btn = QtWidgets.QPushButton("Show results")
        results_btn.clicked.connect(self.show_results_window)
        btn_layout.addWidget(results_btn)

        # Add node color legend
        btn_layout.addWidget(self.create_node_legend())

        self.simulator = None

    def create_network(self):
        self.n_common = int(self.n_common_ui.text())
        self.n_influencer = int(self.n_influencer_ui.text())
        self.n_interests = int(self.n_interests_ui.text())
        self.engagement_news = float(self.engagement_news_ui.text())
        self.simulator = Simulator(N_common=self.n_common, N_influencers=self.n_influencer, N_interests=self.n_interests,
                                   random_const=0.1, random_phy_const=0.1, engagement_news=self.engagement_news)

        self.draw_network()

    def save_network(self):
        with open('network.pickle', 'wb') as handle:
            pickle.dump(self.simulator.network, handle)
        QtWidgets.QMessageBox.about(self, "Info", "Network saved!")

    def load_network(self):
        self.simulator = Simulator(N_common=self.n_common, N_influencers=self.n_influencer, N_interests=self.n_interests,
                                   random_const=0.1, random_phy_const=0.1, engagement_news=self.engagement_news)

        with open('network.pickle', 'rb') as handle:
            self.simulator.network = pickle.load(handle)
        self.draw_network()

        self.n_common = self.simulator.network.N_common
        self.n_influencer = self.simulator.network.N_influencers
        self.n_interests = self.simulator.network.N_interests
        self.n_common_ui.setText(self.n_common)
        self.n_influencer_ui.setText(self.n_influencer)
        self.n_interests_ui.setText(self.n_interests)

        QtWidgets.QMessageBox.about(self, "Info", "Network loaded!")

    def init_parameters(self):
        self.n_common = 100
        self.n_influencer = 5
        self.n_interests = 5
        self.sim_time = 1000
        self.engagement_news = 1.0

    def create_net_parameters_grid(self):
        param_groupbox = QtWidgets.QGroupBox("Network parameters:")

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

        par_layout.addWidget(QLabel('Number of interests:'), 0, 4)
        self.n_interests_ui = QLineEdit()
        self.n_interests_ui.setValidator(QtGui.QIntValidator())
        self.n_interests_ui.setAlignment(QtCore.Qt.AlignRight)
        self.n_interests_ui.setText(str(self.n_interests))
        par_layout.addWidget(self.n_interests_ui, 0, 5)

        param_groupbox.setLayout(par_layout)
        return param_groupbox

    def create_sim_parameters_grid(self):
        param_groupbox = QtWidgets.QGroupBox("Simulation parameters:")

        par_layout = QtWidgets.QGridLayout()

        par_layout.addWidget(QLabel('Simulation time:'), 0, 0)
        self.sim_time_ui = QLineEdit()
        self.sim_time_ui.setValidator(QtGui.QIntValidator())
        self.sim_time_ui.setAlignment(QtCore.Qt.AlignRight)
        self.sim_time_ui.setText(str(self.sim_time))
        par_layout.addWidget(self.sim_time_ui, 0, 1)

        par_layout.addWidget(QLabel('News engagement [0, 1]:'), 0, 2)
        self.engagement_news_ui = QLineEdit()
        self.engagement_news_ui.setValidator(QtGui.QDoubleValidator())
        self.engagement_news_ui.setAlignment(QtCore.Qt.AlignRight)
        self.engagement_news_ui.setText(str(self.engagement_news))
        par_layout.addWidget(self.engagement_news_ui, 0, 3)

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

        G = nx.DiGraph()

        pos = {}
        color_map = []
        for n, node in self.simulator.network.nodes.items():
            G.add_node(n)
            pos[n] = [node.position_x, node.position_y]
            color_map.append(self.node_color[node.type])

        edge_labels = []
        for a, node in self.simulator.network.nodes.items():
            for b in node.adj:
                G.add_edge(a, b.dest)
                edge_labels.append(b)

        nx.draw(G, pos=pos, with_labels=True, font_size=8, node_size=150, node_color=color_map, edge_color="grey")

        edges_artists = self.figure.get_axes()[0].patches

        def annotate_edges(event, ind, **kargs):
            if ind is None:
                idx = edges_artists.index(event.artist)
                edge = edge_labels[idx]
                return "{} - {}\nWeight: {:.3f}".format(edge.start, edge.dest, edge.weight)
            else:
                idx = ind[0]
                node = self.simulator.network.nodes[idx]
                node_type = str(node.type).split('.')[1]
                return "Node: {}\nType: {}\nScore: {:.3f}".format(idx, node_type, node.score)

        datacursor(draggable=True, formatter=annotate_edges)

        self.network_canvas.draw_idle()

    def draw_simulation_network(self, network):
        QtWidgets.QApplication.processEvents()
        self.figure.clf()

        G = nx.DiGraph()

        pos = {}
        color_map = []
        for n, node in network.nodes.items():
            G.add_node(n)
            pos[n] = [node.position_x, node.position_y]
            if 0.5 > node.score > -0.5:
                color_map.append("skyblue")
            elif node.score > 0.5:
                color_map.append("tomato")
            else:
                color_map.append("gold")

        edge_labels = []
        for a, node in network.nodes.items():
            for b in node.adj:
                G.add_edge(a, b.dest)
                edge_labels.append(b)

        nx.draw(G, pos=pos, with_labels=True, font_size=8, node_size=150, node_color=color_map, edge_color="grey")

        edges_artists = self.figure.get_axes()[0].patches

        def annotate_edges(event, ind, **kargs):
            if ind is None:
                idx = edges_artists.index(event.artist)
                edge = edge_labels[idx]
                return "{} - {}\nWeight: {:.3f}".format(edge.start, edge.dest, edge.weight)
            else:
                idx = ind[0]
                node = self.simulator.network.nodes[idx]
                node_type = str(node.type).split('.')[1]
                return "Node: {}\nType: {}\nScore: {:.3f}".format(idx, node_type, node.score)

        datacursor(draggable=True, formatter=annotate_edges)

        self.network_canvas.draw()

    def run_simulation(self):
        self.sim_time = int(self.sim_time_ui.text())
        self.sim_results = self.simulator.simulate(self.sim_time)
        self.progress_bar.setValue(0)
        for i, net in enumerate(self.sim_results):
            self.draw_simulation_network(net[1])
            self.progress_bar.setValue(int((i + 1) / len(self.sim_results) * 100))

        self.show_results_window()

    def show_results_window(self):
        res = ResultsWindow(self.sim_results, self)
        res.show()


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
