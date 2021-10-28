import math
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QApplication
from determinant import compute_det
from transpose_matrix import transposing
from conditionality_matrix import conditionality
from Jacobi_SofALE import solve_jacobi


class App(QMainWindow):
    d = []
    t = []
    c = []
    s = []
    ans_for_s = []

    def __init__(self):
        QMainWindow.__init__(self)
        self.plane = uic.loadUi('calc_ui.ui')
        self.determinant()
        self.transp()
        self.cond()
        self.slau()
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
        if self.c and mode == 'c':
            self.c.pop()
            self.c.pop()
            for i in self.c:
                i.hide()
            self.c.clear()
        if self.s and mode == 's':
            self.s.pop()
            self.s.pop()
            for i in self.s:
                i.hide()
            self.s.clear()
            for i in self.ans_for_s:
                i.hide()
            self.ans_for_s.clear()

        ds = []
        ts = []
        cs = []
        ss = []
        input_text = n1 * n2
        for i in range(0, input_text):
            if mode == 'd':
                line = QLineEdit(f'input{i}', self.plane.tab3)
                ds.append(line)
            elif mode == 't':
                line = QLineEdit(f'input{i}{i}', self.plane.tab2)
                ts.append(line)
            elif mode == 'c':
                line = QLineEdit(f'input{i}{i}{i}', self.plane.tab4)
                cs.append(line)
            elif mode == 's':
                line = QLineEdit(f'input{i}{i}{i}{i}', self.plane.tab5)
                ss.append(line)
            line.resize(70, 30)
            line.setText('')
            line.setPlaceholderText(f'{i}')
            line.move(20 + (i % n2) * 70, 140 + 30 * (i // n2))
            line.show()
        self.d.extend(ds)
        self.t.extend(ts)
        self.t.extend([n1, n2])
        self.c.extend(cs)
        self.c.extend([n1, n2])
        self.s.extend(ss)
        self.s.extend([n1, n2])

    def true_input(self):
        self.plane.label_3.setText('Наличие ошибок:')
        self.plane.label_3.setStyleSheet('background: green;')
        text = self.plane.lineEdit.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2] == text[0]:
            number = int(text[0])
            self.otrisovka(number, number, mode='d')
        elif len(text) == 1 and text.isdigit() and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='d')
        else:
            self.plane.label_3.setText('Наличие ошибок: ошибка ввода')
            self.plane.label_3.setStyleSheet('background: red;')

    def compute_determ(self):
        matrix = []
        a = int(len(self.d) ** 0.5)
        for i in range(a):
            matrix.append([])
            lines = self.d[i * a:i * a + a]
            for j in range(a):
                matrix[i].append(int(lines[j].displayText()))
        ans = compute_det(matrix=matrix)
        self.plane.label_2.setText(f'Ответ: {ans}')

    def determinant(self):
        self.plane.pushButton.clicked.connect(self.true_input)
        self.plane.pushButton_2.clicked.connect(self.compute_determ)

    def true_input_trans(self):
        self.plane.label_6.setText('Наличие ошибок:')
        self.plane.label_6.setStyleSheet('background: green;')
        text = self.plane.lineEdit_2.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2].isdigit():
            number1 = int(text[0])
            number2 = int(text[2])
            self.otrisovka(number1, number2, mode='t')
        elif len(text) == 1 and text.isdigit() and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='t')
        else:
            self.plane.label_6.setText('Наличие ошибок: ошибка ввода')
            self.plane.label_6.setStyleSheet('background: red;')

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

    def true_input_cond(self):
        self.plane.label_7.setText('Наличие ошибок:')
        self.plane.label_7.setStyleSheet('background: green;')
        text = self.plane.lineEdit_3.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2].isdigit():
            number1 = int(text[0])
            number2 = int(text[2])
            self.otrisovka(number1, number2, mode='c')
        elif len(text) == 1 and text.isdigit() and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='c')
        else:
            self.plane.label_7.setText('Наличие ошибок: ошибка ввода')
            self.plane.label_7.setStyleSheet('background: red;')

    def compute_cond(self):
        matrix = []
        n2 = self.c[-1]
        n1 = self.c[-2]
        for i in range(n1):
            matrix.append([])
            lines = self.c[i * n2:i * n2 + n2]
            for j in range(n2):
                matrix[i].append(int(lines[j].displayText()))
        ans = conditionality(matrix)
        if type(ans) == 'str' and 'S' in ans:
            self.plane.label_4.setText(f'Ответ: {ans}')
        else:
            self.plane.label_4.setText(f'Ответ: {round(ans, 3)}')

    def cond(self):
        self.plane.pushButton_5.clicked.connect(self.true_input_cond)
        self.plane.pushButton_6.clicked.connect(self.compute_cond)

    def true_input_slau(self):
        self.plane.label_10.setText('Наличие ошибок:')
        self.plane.label_10.setStyleSheet('background: green;')
        text = self.plane.lineEdit_4.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2].isdigit():
            number1 = int(text[0])
            number2 = int(text[2])
            self.otrisovka(number1, number2, mode='s')
        elif len(text) == 1 and text.isdigit() and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='s')
        else:
            self.plane.label_10.setText('Наличие ошибок: ошибка ввода')
            self.plane.label_10.setStyleSheet('background: red;')

    def compute_slau(self):
        matrix = []
        n2 = self.s[-1]
        n1 = self.s[-2]
        for i in range(n1):
            matrix.append([])
            lines = self.s[i * n2:i * n2 + n2]
            for j in range(n2):
                matrix[i].append(int(lines[j].displayText()))
        print(matrix)
        ans = solve_jacobi(matrix)
        print(ans[1])
        if type(ans) == 'str':
            self.plane.label_11.setText(f'Ответ: {ans}')
        else:
            print('dxcv')
            row = len(matrix)
            for i in range(row):
                line = QLineEdit(f'input0{i}', self.plane.tab5)
                line.resize(70, 30)
                line.setText(str(round(ans[i], 3)))
                line.move(20 + (row + 1) * 70, 140 + 30 * i)
                line.show()
                self.ans_for_s.append(line)

    def slau(self):
        self.plane.pushButton_7.clicked.connect(self.true_input_slau)
        self.plane.pushButton_8.clicked.connect(self.compute_slau)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()

