import math
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *


class App(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.plane = uic.loadUi('calc_ui.ui')
        self.matrix_expression()
        self.transposing()
        self.determinant()
        self.obuslov()
        self.slau()
        self.plane.show()

    def otrisovka(self):
        pass

    def matrix_expression(self):
        pass

    def transposing(self):
        pass

    def determinant(self):
        input_text = 81
        one_side = int(math.sqrt(input_text))
        for i in range(0, input_text):
            line = QLineEdit(f'input{i}', self.plane.tab3)
            line.resize(70, 30)
            line.setText('')
            line.setPlaceholderText('5')
            line.move(20 + (i % one_side) * 70, 100 + 30 * (i // one_side))

    def obuslov(self):
        pass

    def slau(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
