import sys
import csv
import random
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QApplication, QFileDialog
from determinant import compute_det
from transpose_matrix import transposing
from conditionality_matrix import conditionality
from Jacobi_SofALE import solve_jacobi
from input_matrix import *
from copy import deepcopy
from input_for_qt import get_list


class App(QMainWindow):
    d = []
    t = []
    c = []
    s = []
    ans_for_s = []
    e = []
    listt = []
    main_list = []
    letterss = []
    main_text = ''
    flag = False

    def __init__(self):
        QMainWindow.__init__(self)
        self.plane = uic.loadUi('calc_ui.ui')
        self.determinant()
        self.transp()
        self.cond()
        self.slau()
        self.matrix_exp()
        self.plane.show()

    def otrisovka(self, n1, n2, mode, let=None):
        """
        :param mode: для чего происходит отрисовка. d - детерминант, t - транспонирование, с - обусловленность, s - СЛАУ
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
        if self.e and mode == 'e':
            self.e.pop()
            self.e.pop()
            self.e.pop()
            for i in self.e:
                i.hide()
            self.e.clear()

        ds = []
        ts = []
        cs = []
        ss = []
        es = []
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
            elif mode == 'e':
                line = QLineEdit(f'input{i}{i}{i}{i}{i}', self.plane.tab1)
                es.append(line)
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
        self.e.extend(es)
        self.e.extend([n1, n2, let])

    # -------------------------

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
        if not self.s:
            self.plane.label_3.setText('Наличие ошибок: нет ячеек')
            self.plane.label_3.setStyleSheet('background: red;')
        else:
            matrix = []
            a = int(len(self.d) ** 0.5)
            for i in range(a):
                matrix.append([])
                lines = self.d[i * a:i * a + a]
                for j in range(a):
                    matrix[i].append(float(lines[j].displayText()))
            ans = compute_det(matrix=matrix)
            self.plane.label_2.setText(f'Ответ: {round(ans, 3)}')

    def random_det(self):
        if self.d:
            n = len(self.d)
            for i in range(n):
                rand_number = round(random.uniform(-1000, 1000), 3)
                self.d[i].setText(f'{rand_number}')
        else:
            self.plane.label_3.setText('Наличие ошибок: нет ячеек')
            self.plane.label_3.setStyleSheet('background: red;')

    def csv_determinant(self):
        file_name, _ = QFileDialog.getOpenFileName(self.plane, '', 'Choose file', 'CSV files (*.csv)')
        if len(file_name) > 0 and file_name != 'Choose file':
            self.plane.lineEdit_8.setText(file_name)

    def use_csv_det(self):
        if self.d:
            if self.plane.lineEdit_8.displayText() != '':
                n = int(len(self.d) ** 0.5)
                with open(self.plane.lineEdit_8.displayText()) as f:
                    reader = csv.reader(f)
                    matrix = []
                    for line in reader:
                        l = line[0].split(';')
                        matrix.append(l)
                    for i in range(0, n):
                        for j in range(0, n):
                            self.d[i*n+j].setText(matrix[i][j])
            else:
                self.plane.label_3.setText('Наличие ошибок: не указан путь')
                self.plane.label_3.setStyleSheet('background: red;')
        else:
            self.plane.label_3.setText('Наличие ошибок: нет ячеек')
            self.plane.label_3.setStyleSheet('background: red;')

    def determinant(self):
        self.plane.pushButton.clicked.connect(self.true_input)
        self.plane.pushButton_2.clicked.connect(self.compute_determ)
        self.plane.pushButton_10.clicked.connect(self.random_det)
        self.plane.pushButton_16.clicked.connect(self.csv_determinant)
        self.plane.pushButton_15.clicked.connect(self.use_csv_det)

    # -------------------------

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
        if not self.s:
            self.plane.label_6.setText('Наличие ошибок: нет ячеек')
            self.plane.label_6.setStyleSheet('background: red;')
        else:
            matrix = []
            n2 = self.t[-1]
            n1 = self.t[-2]
            for i in range(n1):
                matrix.append([])
                lines = self.t[i * n2:i * n2 + n2]
                for j in range(n2):
                    matrix[i].append(lines[j].displayText())
            ans = transposing(mat=matrix)
            self.otrisovka(n2, n1, mode='t')
            for i in range(n2):
                lines = self.t[i * n1:i * n1 + n1]
                for j in range(n1):
                    lines[j].setText(str(ans[i][j]))

    def random_trans(self):
        if self.t:
            n = len(self.t)
            for i in range(0, n-2):
                rand_number = round(random.uniform(-1000, 1000), 3)
                self.t[i].setText(f'{rand_number}')
        else:
            self.plane.label_6.setText('Наличие ошибок: нет ячеек')
            self.plane.label_6.setStyleSheet('background: red;')

    def csv_trans(self):
        file_name, _ = QFileDialog.getOpenFileName(self.plane, '', 'Choose file', 'CSV files (*.csv)')
        if len(file_name) > 0 and file_name != 'Choose file':
            self.plane.lineEdit_7.setText(file_name)

    def use_csv_trans(self):
        if self.t:
            if self.plane.lineEdit_7.displayText() != '':
                n1 = self.t[-2]
                n2 = self.t[-1]
                with open(self.plane.lineEdit_7.displayText()) as f:
                    reader = csv.reader(f)
                    matrix = []
                    for line in reader:
                        l = line[0].split(';')
                        matrix.append(l)

                    for i in range(0, n1):
                        for j in range(0, n2):
                            self.t[i*n2+j].setText(matrix[i][j])

            else:
                self.plane.label_6.setText('Наличие ошибок: не указан путь')
                self.plane.label_6.setStyleSheet('background: red;')
        else:
            self.plane.label_6.setText('Наличие ошибок: нет ячеек')
            self.plane.label_6.setStyleSheet('background: red;')

    def transp(self):
        self.plane.pushButton_3.clicked.connect(self.true_input_trans)
        self.plane.pushButton_4.clicked.connect(self.compute_trans)
        self.plane.pushButton_9.clicked.connect(self.random_trans)
        self.plane.pushButton_13.clicked.connect(self.csv_trans)
        self.plane.pushButton_14.clicked.connect(self.use_csv_trans)

    # -------------------------

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
        if not self.s:
            self.plane.label_7.setText('Наличие ошибок: нет ячеек')
            self.plane.label_7.setStyleSheet('background: red;')
        else:
            matrix = []
            n2 = self.c[-1]
            n1 = self.c[-2]
            for i in range(n1):
                matrix.append([])
                lines = self.c[i * n2:i * n2 + n2]
                for j in range(n2):
                    matrix[i].append(float(lines[j].displayText()))
            ans = conditionality(matrix)
            if type(ans) == 'str' and 'S' in ans:
                self.plane.label_4.setText(f'Ответ: {ans}')
            else:
                self.plane.label_4.setText(f'Ответ: {round(ans, 3)}')

    def random_cond(self):
        if self.c:
            n = len(self.c)
            for i in range(0, n-2):
                rand_number = round(random.uniform(-1000, 1000), 3)
                self.c[i].setText(f'{rand_number}')
        else:
            self.plane.label_7.setText('Наличие ошибок: нет ячеек')
            self.plane.label_7.setStyleSheet('background: red;')

    def csv_cond(self):
        file_name, _ = QFileDialog.getOpenFileName(self.plane, '', 'Choose file', 'CSV files (*.csv)')
        if len(file_name) > 0 and file_name != 'Choose file':
            self.plane.lineEdit_9.setText(file_name)

    def use_csv_cond(self):
        if self.c:
            if self.plane.lineEdit_9.displayText() != '':
                n1 = self.c[-2]
                n2 = self.c[-1]
                with open(self.plane.lineEdit_9.displayText()) as f:
                    reader = csv.reader(f)
                    matrix = []
                    for line in reader:
                        l = line[0].split(';')
                        matrix.append(l)
                    for i in range(0, n1):
                        for j in range(0, n2):
                            self.c[i*n2+j].setText(matrix[i][j])
            else:
                self.plane.label_7.setText('Наличие ошибок: не указан путь')
                self.plane.label_7.setStyleSheet('background: red;')
        else:
            self.plane.label_7.setText('Наличие ошибок: нет ячеек')
            self.plane.label_7.setStyleSheet('background: red;')

    def cond(self):
        self.plane.pushButton_5.clicked.connect(self.true_input_cond)
        self.plane.pushButton_6.clicked.connect(self.compute_cond)
        self.plane.pushButton_11.clicked.connect(self.random_cond)
        self.plane.pushButton_17.clicked.connect(self.csv_cond)
        self.plane.pushButton_18.clicked.connect(self.use_csv_cond)

    # -------------------------

    def true_input_slau(self):
        self.plane.label_10.setText('Наличие ошибок:')
        self.plane.label_10.setStyleSheet('background: green;')
        text = self.plane.lineEdit_4.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2].isdigit() and 2 <= int(text[0]) < int(text[2]) <= 8:
            number1 = int(text[0])
            number2 = int(text[2])
            self.otrisovka(number1, number2, mode='s')
        else:
            self.plane.label_10.setText('Наличие ошибок: ошибка ввода')
            self.plane.label_10.setStyleSheet('background: red;')

    def compute_slau(self):
        if not self.s:
            self.plane.label_10.setText('Наличие ошибок: нет ячеек')
            self.plane.label_10.setStyleSheet('background: red;')
        else:
            matrix = []
            n2 = self.s[-1]
            n1 = self.s[-2]
            for i in range(n1):
                matrix.append([])
                lines = self.s[i * n2:i * n2 + n2]
                for j in range(n2):
                    matrix[i].append(float(lines[j].displayText()))
            ans = solve_jacobi(matrix)
            if type(ans) == 'str':
                self.plane.label_11.setText(f'Ответ: {ans}')
            else:
                row = len(matrix)
                for i in range(row):
                    line = QLineEdit(f'input0{i}', self.plane.tab5)
                    line.resize(70, 30)
                    line.setText(str(round(ans[i], 3)))
                    line.move(20 + (row + 1) * 70, 140 + 30 * i)
                    line.show()
                    self.ans_for_s.append(line)

    def random_slau(self):
        if self.s:
            n = len(self.s)
            for i in range(0, n - 2):
                rand_number = round(random.uniform(-1000, 1000), 3)
                self.s[i].setText(f'{rand_number}')
        else:
            self.plane.label_10.setText('Наличие ошибок: нет ячеек')
            self.plane.label_10.setStyleSheet('background: red;')

    def csv_slau(self):
        file_name, _ = QFileDialog.getOpenFileName(self.plane, '', 'Choose file', 'CSV files (*.csv)')
        if len(file_name) > 0 and file_name != 'Choose file':
            self.plane.lineEdit_10.setText(file_name)

    def use_csv_slau(self):
        if self.s:
            if self.plane.lineEdit_10.displayText() != '':
                n1 = self.s[-2]
                n2 = self.s[-1]
                with open(self.plane.lineEdit_10.displayText()) as f:
                    reader = csv.reader(f)
                    matrix = []
                    for line in reader:
                        l = line[0].split(';')
                        matrix.append(l)
                    for i in range(0, n1):
                        for j in range(0, n2):
                            self.s[i*n2+j].setText(matrix[i][j])
            else:
                self.plane.label_10.setText('Наличие ошибок: не указан путь')
                self.plane.label_10.setStyleSheet('background: red;')
        else:
            self.plane.label_10.setText('Наличие ошибок: нет ячеек')
            self.plane.label_10.setStyleSheet('background: red;')

    def slau(self):
        self.plane.pushButton_7.clicked.connect(self.true_input_slau)
        self.plane.pushButton_8.clicked.connect(self.compute_slau)
        self.plane.pushButton_12.clicked.connect(self.random_slau)
        self.plane.pushButton_20.clicked.connect(self.csv_slau)
        self.plane.pushButton_19.clicked.connect(self.use_csv_slau)

    # -------------------------

    def true_input_exp(self):
        self.plane.label_14.setText('Наличие ошибок:')
        self.plane.label_14.setStyleSheet('background: green;')
        text = self.plane.lineEdit_5.displayText()
        if text != '':
            list_of_mat, new_text = get_list(text)
            self.listt = list_of_mat
            self.main_list = deepcopy(list_of_mat)
            self.main_text = new_text
            if list_of_mat:
                print('первый вызов', self.listt)
                self.nnext(self.listt.pop())
            else:
                self.plane.label_14.setText('Наличие ошибок: ошибка в выражении')
                self.plane.label_14.setStyleSheet('background: red;')
        else:
            self.plane.label_14.setText('Наличие ошибок: нет выражения')
            self.plane.label_14.setStyleSheet('background: red;')

    def nnext(self, n):
        self.plane.label_13.setText(f'Размер матрицы "{n}"')
        self.plane.pushButton_21.clicked.connect(self.true_input_matrix_for_exp)
        self.plane.pushButton_23.clicked.connect(self.get_inter)

    def true_input_matrix_for_exp(self):
        self.plane.label_14.setText('Наличие ошибок:')
        self.plane.label_14.setStyleSheet('background: green;')
        text = self.plane.lineEdit_6.displayText()
        if len(text) == 3 and text[1] in ('x', 'х') and text[0].isdigit() and text[2].isdigit():
            number1 = int(text[0])
            number2 = int(text[2])
            self.otrisovka(number1, number2, mode='e', let=self.plane.label_13.text()[-2])
        elif len(text) == 1 and text.isdigit() and (2 <= int(text) <= 9):
            self.otrisovka(int(text), int(text), mode='e', let=self.plane.label_13.text()[-2])
        else:
            self.plane.label_14.setText('Наличие ошибок: ошибка ввода')
            self.plane.label_14.setStyleSheet('background: red;')

    def get_inter(self):
        if not self.e:
            self.plane.label_14.setText('Наличие ошибок: нет ячеек')
            self.plane.label_14.setStyleSheet('background: red;')
        else:
            if not self.flag:
                matrix = []
                let = self.e[-1]
                n2 = self.e[-2]
                n1 = self.e[-3]
                for i in range(n1):
                    matrix.append([])
                    lines = self.e[i * n2:i * n2 + n2]
                    for j in range(n2):
                        matrix[i].append(float(lines[j].displayText()))
                exec(f'{let} = Matrix({n1}, {n2}, {matrix})')
                exec(f'self.letterss.append({let})')
                print(self.letterss)
                exec(f'print("Матрица {let}", {let}.matrix)')
                if self.listt:
                    print('Второй вызов', self.listt)
                    self.nnext(self.listt.pop())
                else:
                    if not self.flag:
                        diction = {}
                        count = 0
                        for i in self.main_list[::-1]:
                            diction[i] = self.letterss[count]
                            count += 1
                        print(self.letterss)
                        new_t = ''
                        for i in self.main_text:
                            if i.isalpha():
                                new_t += f'diction["{i}"]'
                            else:
                                new_t += i
                        print(new_t)
                        print(diction)
                        print('ewsdfbd')
                        ans = eval(new_t).matrix
                        print(ans)
                        self.flag = True
                        self.otrisovka(len(ans), len(ans[0]), mode='e')
                        for i in range(len(ans)):
                            lines = self.e[i*len(ans[0]): i*len(ans[0]) + len(ans[0])]
                            for j in range(len(ans[0])):
                                lines[j].setText(str(ans[i][j]))
                    else:
                        pass

    def matrix_exp(self):
        self.plane.pushButton_22.clicked.connect(self.true_input_exp)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
