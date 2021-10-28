import math
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *


class App(QMainWindow):
    d = []
    def __init__(self):
        QMainWindow.__init__(self)
        self.plane = uic.loadUi('calc_ui.ui')
        self.otr()
        self.determinant()
        self.plane.show()

    def otr(self):
        for i in range(0, 81):
            line = QLineEdit(f'input{i}', self.plane.tab3)
            line.resize(70, 30)
            line.setText('')
            line.setPlaceholderText('5')
            line.move(20 + (i % 9) * 70, 100 + 30 * (i // 9))
            line.hide()
            self.d.append(line)

    def otrisovka(self, n):
        input_text = n
        one_side = int(math.sqrt(input_text))
        for i in range(0, n):
            self.d[i].show()
        for i in range(n, 81):
            self.d[i].hide()

    def true_input(self):
        if self.plane.lineEdit.displayText() == '':
            print('bbc')
        else:
            self.otrisovka(int(self.plane.lineEdit.displayText()))

    def determinant(self):
        self.plane.pushButton.clicked.connect(self.true_input)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
