from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import networkx as nx
import sys
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


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

        # Add buttons controls
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addLayout(btn_layout)
        graph_btn = QtWidgets.QPushButton("Create graph")
        graph_btn.clicked.connect(self.draw_network)
        btn_layout.addWidget(graph_btn)

    def draw_network(self):
        self.figure.clf()
        z = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]

        G = nx.configuration_model(z)
        degree_sequence = [d for n, d in G.degree()]
        hist = {}
        for d in degree_sequence:
            if d in hist:
                hist[d] += 1
            else:
                hist[d] = 1

        nx.draw(G)
        self.network_canvas.draw_idle()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
