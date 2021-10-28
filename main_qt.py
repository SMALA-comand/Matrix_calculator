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
        self.transp()
        self.plane.show()

    def otrisovka(self, n):
        if self.d:
            for i in self.d:
                i.hide()
        ds = []
        input_text = n**2
        one_side = n
        for i in range(0, input_text):
            line = QLineEdit(f'input{i}', self.plane.tab3)
            line.resize(70, 30)
            line.setText('')
            line.setPlaceholderText('5')
            line.move(20 + (i % one_side) * 70, 140 + 30 * (i // one_side))
            line.show()
            ds.append(line)
        self.d.extend(ds)

    def true_input(self):
        self.plane.label_3.setText('Наличие ошибок:')
        text = self.plane.lineEdit.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2] == text[0]:
            number = self.plane.lineEdit.displayText()[0]
            number = int(number)
            self.otrisovka(number)
        elif len(text) == 1 and (2 <= int(text) <= 9):
            self.otrisovka(int(self.plane.lineEdit.displayText()))
        else:
            self.plane.label_3.setText('Наличие ошибок: ошибка ввода')

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

    def otrisovka_trans(self):
        pass

    def true_input_trans(self):
        pass

    def compute_trans(self):
        pass

    def transp(self):
        self.plane.pushButton_3.clicked.connect(self.true_input_trans)
        self.plane.pushButton_4.clicked.connect(self.compute_trans)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
