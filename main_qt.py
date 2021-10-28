import math
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from determinant import compute_det


class App(QMainWindow):
    d = []

    def __init__(self):
        QMainWindow.__init__(self)
        self.plane = uic.loadUi('calc_ui.ui')
        self.determinant()
        self.plane.show()

    def otrisovka(self, n):
        if self.d:
            for i in self.d:
                i.hide()
        ds = []
        input_text = n
        one_side = int(math.sqrt(input_text))
        for i in range(0, n):
            line = QLineEdit(f'input{i}', self.plane.tab3)
            line.resize(70, 30)
            line.setText('')
            line.setPlaceholderText('5')
            line.move(20 + (i % one_side) * 70, 100 + 30 * (i // one_side))
            line.show()
            ds.append(line)
        self.d.extend(ds)

    def true_input(self):
        if self.plane.lineEdit.displayText() == '':
            print('bbc')
        else:
            self.otrisovka(int(self.plane.lineEdit.displayText()))

    def compute_determ(self):
        matrix = []
        a = int(len(self.d)**0.5)
        for i in range(a):
            matrix.append([])
            lines = self.d[i*a:i*a+a]
            for j in range(a):
                matrix[i].append(int(lines[j].displayText()))
        ans = compute_det(matrix=matrix)
        self.plane.label_2.setText(f'Ответ: {ans}')

    def determinant(self):
        self.plane.pushButton.clicked.connect(self.true_input)
        self.plane.pushButton_2.clicked.connect(self.compute_determ)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
