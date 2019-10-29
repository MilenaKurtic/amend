import sys

from PyQt5 import QtWidgets
from ui    import Ui

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    app.exec_()
