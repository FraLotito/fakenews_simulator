from PyQt5.QtWidgets import *
import sys

from src.application_window import ApplicationWindow


if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
