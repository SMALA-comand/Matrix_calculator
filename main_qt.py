import math
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from determinant import compute_det
from transpose_matrix import transposing


class App(QMainWindow):
    d = []
    t = []
    o = []
    s = []

    def __init__(self):
        QMainWindow.__init__(self)
        self.plane = uic.loadUi('calc_ui.ui')
        self.determinant()
        self.transp()
        self.plane.show()

    def otrisovka(self, n1, n2, mode):
        """
        :param n1: колво строк
        :param n2: колво столбцов
        :return:
        """
        if self.d and mode == 'd':
            for i in self.d:
                i.hide()
            self.d.clear()
        if self.t and mode == 't':
            self.t.pop()
            self.t.pop()
            for i in self.t:
                i.hide()
            self.t.clear()
        ds = []
        ts = []
        input_text = n1*n2
        for i in range(0, input_text):
            if mode == 'd':
                line = QLineEdit(f'input{i}', self.plane.tab3)
                ds.append(line)
            elif mode == 't':
                line = QLineEdit(f'input{i}{i}', self.plane.tab2)
                ts.append(line)
            line.resize(70, 30)
            line.setText('')
            line.setPlaceholderText('5')
            line.move(20 + (i % n2) * 70, 140 + 30 * (i // n2))
            line.show()
        self.d.extend(ds)
        self.t.extend(ts)
        self.t.extend([n1, n2])

    def true_input(self):
        self.plane.label_3.setText('Наличие ошибок:')
        text = self.plane.lineEdit.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2] == text[0]:
            number = int(text[0])
            self.otrisovka(number, number, mode='d')
        elif len(text) == 1 and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='d')
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

    def true_input_trans(self):
        self.plane.label_6.setText('Наличие ошибок:')
        text = self.plane.lineEdit_2.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2].isdigit():
            number1 = int(text[0])
            number2 = int(text[2])
            self.otrisovka(number1, number2, mode='t')
        elif len(text) == 1 and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='t')
        else:
            self.plane.label_6.setText('Наличие ошибок: ошибка ввода')

    def compute_trans(self):
        matrix = []
        n2 = self.t[-1]
        n1 = self.t[-2]
        for i in range(n1):
            matrix.append([])
            lines = self.t[i * n2:i * n2 + n2]
            for j in range(n2):
                matrix[i].append(int(lines[j].displayText()))
        ans = transposing(mat=matrix)
        for i in range(n1):
            lines = self.t[i * n2:i * n2 + n2]
            for j in range(n2):
                lines[j].setText(str(ans[i][j]))

    def transp(self):
        self.plane.pushButton_3.clicked.connect(self.true_input_trans)
        self.plane.pushButton_4.clicked.connect(self.compute_trans)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
