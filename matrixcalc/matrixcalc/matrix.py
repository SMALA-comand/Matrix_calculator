'''
``matrixcalc.matrix``
Здесь представлено решение задания №2 "Матричный калькулятор" и задания №3 "СЛАУ" по дисциплине "Численные методы"

Функционал калькулятора:

    1. Транспонирование матриц
    2. Определитель матрицы
    3. Матричные выражения
    4. Нахождение обусловленности матриц
    5. Решение СЛАУ

Калькулятор - matrix_calc()

'''

import numpy as np
from numpy import linalg as LA
import random as ra
import csv
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import sympy
from sympy.solvers.solveset import linsolve



#'1_transpose_matrix.py'
def get_col(mat, col):
    """
    :param mat: матрица, из которой нужно взять столбец
    :param col: номер столбца, который нужно вычленить, отсчет от нуля
    :return: столбец в виде массива
    """
    res = []
    for row in range(len(mat)):
        res.append(mat[row][col])
    return res


def transposing(mat):
    """
    :param mat: матрица произвольного размера, которую транспонируем
    :return: транспонированная матрица
    """
    row, col = len(mat), len(mat[0])
    if row == col:
        n = row
        for rows in range(0, n):
            for columns in range(rows, n):
                mat[rows][columns], mat[columns][rows] = mat[columns][rows], mat[rows][columns]
    else:
        ans = []
        for columns in range(0, col):
            ans.append(get_col(mat=mat, col=columns))
        mat = ans
    return mat


#if __name__ == '__main__':
#    print(transposing([[1, 2, 3],
#                       [4, 5, 6],
#                       [7, 8, 9]]))
#
# Сравнение скорости работы нашего транспонирования с numpy
'''
import numpy as np
from matrix_generator import matrix_generator
import time
import matplotlib.pyplot as plt

time_our = []
time_np = []
size_m = [] # 1*1 , 2*2, 3*3, 4*4, 5*5,  и т.д.
def comparison(matrix): 
    a = np.array(matrix)
    return a.transpose()

for i in range(1, 501,10):
    m = matrix_generator(i,i)
    time_start  = time.time()
    comparison(m)  # Траспонирование numpy
    elapsed_time = time.time() - time_start
    time_np.append(elapsed_time)

    time_start  = time.time()
    transposing(m)   # Наше транспонирование 
    elapsed_time = time.time() - time_start
    time_our.append(elapsed_time)

    size_m.append(i)

fig, ax = plt.subplots()
ax.plot(size_m, time_our, 'co-',label = 'Наше транмпонирование')
ax.plot(size_m, time_np, 'r--',label = 'Numpy')
ax.set(xlabel = 'Размер матрицы x*x', ylabel = "Время в секундах")
ax.legend(loc = 'upper left')
plt.title('Скорость выполнения функции',loc = 'center', pad = 10 )
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show() '''


#'2_matrix_generator.py'



def matrix_generator(n=0, m=0):
    if n == m == 0:
        print('Введите количество строк (n): ')
        n = int(input())
        print('Введите количество строк (m): ')
        m = int(input())

    matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(ra.random() * (10 ** ra.randint(1, 6)))
        matrix.append(row)
    with open("rand_matrix.csv", mode='w', encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=',', lineterminator='\r')
        file_writer.writerows(matrix)
    return matrix


#'3_determinant.py'
dict = {}


def matr_to_string(matrix):
    s = ""
    for row in matrix:
        for i in range(0, len(matrix)):
            s += str(row[i])
    return s


def get_matrix_minor(mat, column=0, row=0):
    """
    :param row: строка, от которой избавляемся. По дефолту равна 0
    :param mat: матрица, для которой нужно посчитать минор
    :param column: столбец, от которого нужно избавиться, индексация с нуля
    :return: минор матрицы
    """
    # print(mat, "**")
    mat1 = deepcopy(mat)
    mat1.pop(row)
    for el in range(0, len(mat1)):
        mat1[el].pop(column)
    # print(mat1, "!!")
    return mat1


def find_best_var(matrix):
    # поиск по строке
    count_0_row = [0, sum([abs(matrix[0][i]) for i in range(0, len(matrix[0]))]),
                   0]  # Счётчик оптимизации с наибольшим чилом нулей и наименьшем модулем коэфициентов (Только для строки!)
    k = 0  # Счётчик строки
    for stroka in matrix:
        if count_0_row[0] < stroka.count(0):
            count_0_row[0] = stroka.count(0)
            count_0_row[2] = k

        elif count_0_row[0] == stroka.count(0):
            cur_sum = sum([abs(stroka[i]) for i in range(0, len(matrix))])
            if cur_sum <= count_0_row[1]:
                count_0_row[1] = cur_sum
                count_0_row[2] = k
        k += 1

    # поиск по столбцу
    count_0_col = [0, sum([abs(matrix[i][0]) for i in range(0, len(matrix[0]))]),
                   0]  # Счётчик оптимизации с наибольшим чилом нулей и наименьшем модулем коэфициентов (Только для строки!)
    for i in range(1, len(matrix)):
        stolb = [matrix[j][i] for j in range(0, len(matrix))]
        if count_0_col[0] < stolb.count(0):
            count_0_col[0] = stolb.count(0)
            count_0_col[2] = i

        elif count_0_col[0] == stolb.count(0):
            cur_sum = sum([abs(stolb[i]) for i in range(0, len(matrix))])
            if cur_sum <= count_0_col[1]:
                count_0_col[1] = cur_sum
                count_0_col[2] = i

    if count_0_row[0] > count_0_col[0]:
        return count_0_row[2], 0
    elif count_0_row[0] < count_0_col[0]:
        return count_0_col[2], 1
    elif count_0_row[1] > count_0_col[1]:
        return count_0_col[2], 1
    return count_0_row[2], 0


def compute_det(matrix) -> int:
    """
    :param matrix: квадратная матрица
    :return: определитель матрицы
    """
    if len(matrix) == 1:
        return matrix[0][0]
    # if len(matrix) == 2 and len(matrix[0]) == 2:
    #    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    # elif len(matrix) == 3 and len(matrix[0]) == 3:
    #    return matrix[0][0] * matrix[1][1] * matrix [2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2]
    else:
        s = matr_to_string(matrix)
        if dict.get(s) != None:
            return dict[s]
        count = 0
        plan = []

        best_num, dim = find_best_var(matrix)
        if dim == 0:
            for item in matrix[best_num]:
                plan.append(
                    (-1) ** (count + best_num) * item * compute_det(get_matrix_minor(matrix, count, row=best_num)))
                count += 1
                # print(plan, best_num, "row", count, matrix)
        else:
            for item in range(len(matrix)):
                plan.append((-1) ** (best_num + item) * matrix[item][best_num] * compute_det(
                    get_matrix_minor(matrix, column=best_num, row=item)))
                # print(plan, item, "col", best_num, matrix)
        a = sum(plan)
        dict[matr_to_string(matrix)] = a
        return a


#if __name__ == '__main__':
    #print(compute_det(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))
    # print(get_matrix_minor(mat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], column=0))


def numpy_linalg_det(matrix):
    return numpy.linalg.det(matrix)


# numpy_linalg_det([[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]])
'''
# Сравнение скорости работы нашего нахождения детерминанта с numpy
import time
from matrix_generator import matrix_generator
import matplotlib.pyplot as plt

time_our = []
time_np = []
size_m = []  # 1*1 , 2*2, 3*3, 4*4, 5*5,  и т.д.

for i in range(1, 8):
    m = matrix_generator(i, i)
    time_start = time.time()
    numpy_linalg_det(m)  # Нахождение детерминанта numpy
    elapsed_time = time.time() - time_start
    time_np.append(elapsed_time)

    time_start = time.time()
    compute_det(m)  # Наше нахождение детерминанта
    elapsed_time = time.time() - time_start
    time_our.append(elapsed_time)

    size_m.append(i)

fig, ax = plt.subplots()
ax.plot(size_m, time_our, 'co-', label='Наше нахождение детерминанта')
ax.plot(size_m, time_np, 'r--', label='Numpy')
ax.set(xlabel='Размер матрицы x*x', ylabel="Время в секундах")
ax.legend(loc='upper left')
plt.title('Скорость выполнения функции', loc='center', pad=10)
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show()'''

#'4_matrix_examples.py'
def give_example():
    dict = {}
    dict[(1, 1)] = [[1]]
    dict[(1, 2)] = [[2, 4]]
    dict[(2, 2)] = [[1, 2], [4, 3]]
    dict[(2, 3)] = [[1, 4, 6], [2, 3, 5]]
    dict[(3, 3)] = [[3, 6, 9], [5, 4, 2], [1, 8, 7]]
    dict[(2, 4)] = [[1, 2, 5, 7], [3, 5, 1, 9]]
    dict[(3, 4)] = [[1, 2, 6, 8], [4, 1, 7, 5], [0, 2, 4, 1]]
    dict[(4, 4)] = [[1, 4, 6, 9], [2, 5, 3, 7], [4, 9, 1, 5], [1, 3, 8, 0]]
    dict[(3, 5)] = [[1, 1, 1, 1, 1], [5, 4, 3, 2, 1], [7, 8, 2, 1, 5]]
    dict[(4, 5)] = [[5, 5, 5, 5, 5], [8, 1, 5, 4, 1], [1, 2, 3, 4, 5], [0, 1, 0, 1, 4]]
    dict[(5, 5)] = [[1, 1, 2, 3, 5], [7, 0, 3, 6, 9], [3, 2, 5, 8, 5], [1, 1, 1, 1, 1], [2, 0, 7, 1, 8]]
    for i in dict.keys:
        if (m, n) not in dict.keys:
            dict[(m, n)] = transposing(dict[(n, m)])

    print(dict)
    print('Введите необходимое количество строк(n):')
    n = int(input())
    print('Введите необходимое количество строк(m):')
    m = int(input())

    return(dict[(n, m)])



#'5_matrix_norms.py'
# Тут будут представлены 3 способа нахождения нормы матрицы
# 1 способ  - ∞-норма матрицы – это максимальная сумма модулей
# элементов каждой из строк матрицы (бесконечная или построчная норма)
# 2 способ - 1-норма – это максимальная сумма модулей элементов
# каждого из столбцов матрицы
# 3 способ - 2-норма (евклидова норма) – длина вектора в n-мерном пространстве
# (корень квадратный из суммы квадратов всех элементов матрицы)

def infinity_norm(matrix):
    """
        :param matrix: матрица, для которой мы будем считать норму
    """
    s = [] # Список с суммами модулей элементов построчно
    for stroka in matrix:
        summa = 0
        for el in stroka:
            summa += abs(el)
        s.append(summa)
    return max(s)


def first_norm(matrix):
    """
        :param matrix: матрица, для которой мы будем считать норму
    """
    s = []  # Список с суммами модулей элементов по столбцам
    for i in range(0, len(matrix)):
        stolb = [matrix[j][i] for j in range(0, len(matrix))]
        summa = 0
        for el in stolb:
            summa += abs(el)
        s.append(summa)
    return max(s)


def second_norm(matrix):
    """
        :param matrix: матрица, для которой мы будем считать норму
    """
    summa = 0
    for stroka in matrix:
        for el in stroka:
            summa += (el)**2
    return (summa)**(0.5)


#if __name__ == '__main__':
#    print(infinity_norm(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))


#"6_conditionality_matrix.py"
def conditionality(matrix):
    """
        :param matrix: матрица, для которой мы будем считать обусловленность
    """
    # Считаем обратную матрицу с помощью numpy.linalg.inv()
    matrix_reverse = np.linalg.inv(matrix)
    # Находим обусловленность
    cond = first_norm(matrix) * first_norm(matrix_reverse)
    return cond


#if __name__ == '__main__':
#   print(conditionality(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))



#'7_Gauss_SofALE.py'
def find_max_row(m, col):
    """Заменим строку [col] на одну из нижележащих строк с наибольшим по модулю первым элементом.

    :param m: исходная матрица
    :param col: индекс столбца/строки, из которого будет запущен базовый поиск
    """
    max_element = m[col][col]
    max_row = col
    for i in range(col + 1, len(m)):
        if abs(m[i][col]) > abs(max_element):
            max_element = m[i][col]
            max_row = i
    if max_row != col:
        m[col], m[max_row] = m[max_row], m[col]


# Решаем Гауссом
def solve_gauss(m):
    """
    :param m: Исходная матрица
    """
    n = len(m)
    # Прямой ход
    for k in range(n - 1):
        find_max_row(m, k)
        for i in range(k + 1, n):
            if m[k][k] != 0:
                div = m[i][k] / m[k][k]
                m[i][-1] -= div * m[k][-1]
                for j in range(k, n):
                    m[i][j] -= div * m[k][j]

    # Проверяем, имеет ли система конечное число решений
    if np.sum(m[:][:]) == 0:
        return m[:][1]
    if m[-2][:-1] == m[-1][:-1] or (m[-1][-1] != 0 and np.sum(m[-1][:-1]) == 0):
        x = 'Система не имеет решений ¯\_(ツ)_/¯ '
        return x
    elif is_singular(m):
        x = 'Система имеет бесконечное количество решений '
        return x

    # Обратный ход
    x = [0 for i in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = (m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)])) / m[k][k]

    return x


def is_singular(m):
    """
    :param m: матрица
    """
    for i in range(len(m)):
        if not m[i][i]:
            return True
    return False


# Прямой алгоритм Гаусса Жордана, с вычислениями правильных дробей
def solve_gauss_fractions(m):
    from fractions import Fraction
    """
    :param m: Исходная матрица
    """
    n = len(m)
    # Прямой ход
    for k in range(n - 1):
        find_max_row(m, k)
        for i in range(k + 1, n):
            if m[k][k] != 0:
                div = Fraction(Fraction(m[i][k]).limit_denominator(10 ** 9),
                               Fraction(m[k][k]).limit_denominator(10 ** 9))
                m[i][-1] -= div * m[k][-1]
                for j in range(k, n):
                    m[i][j] -= div * m[k][j]

    # Проверяем, имеет ли система конечное число решений
    if np.sum(m[:][:]) == 0:
        return m[:][1]
    if m[-2][:-1] == m[-1][:-1] or (m[-1][-1] != 0 and np.sum(m[-1][:-1]) == 0):
        x = 'Система не имеет решений ¯\_(ツ)_/¯ '
        return x
    elif is_singular(m):
        x = 'Система имеет бесконечное количество решений '
        return x

    # Обратный ход
    x = [0 for i in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = Fraction(
            Fraction((m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)]))).limit_denominator(10 ** 9),
            Fraction(m[k][k]).limit_denominator(10 ** 9))

    return x


def inverse_matrix(matrix_origin):
    """
    :param matrix_origin: Исходная матрица
    """
    matrix_origin = np.array(matrix_origin)
    m = np.hstack((matrix_origin,
                   np.matrix(np.diag([1.0 for i in range(matrix_origin.shape[0])]))))
    # Прямой ход
    for k in range(len(m)):
        swap_row = pick_nonzero_row(m, k)  # Меняем местами k-строку с одной из нижележащих, если m[k, k] = 0
        if swap_row != k:
            m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        # Делаем диагональный элемент равным 1
        if m[k, k] != 1:
            m[k, :] *= 1 / m[k, k]
        for row in range(k + 1, len(m)):
            m[row, :] -= m[k, :] * m[row, k]

    # Обратный ход
    for k in range(len(m) - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                #  Делаем все вышележащие элементы равными нулю в прежней матрице идентичности
                m[row, :] -= m[k, :] * m[row, k]

    return np.hsplit(m, len(m) // 2)[1]


def pick_nonzero_row(m, k):
    """
    :param m: Исходная матрица
    :param k: k-строка матрицы
    """

    while k < m.shape[0] and not m[k, k]:
        k += 1
    return k


#if __name__ == '__main__':
    # print(solve_gauss_fractions(m=[[2.6, -1.7, 2.5, 3.7], [1.5, 6.2, -2.9, 3.2], [2.8, -1.7, 3.8, 2.8]]))
    # print(solve_gauss_fractions(m = [[1,2,3,6],[2,3,1,6],[3,1,2,6]]))
    # print(solve_gauss_fractions(m=[[0, 0, 0, 1], [0, 0, 0,0], [0, 0, 0, 0]]))

#'8_Jacobi_SofALE.py'
# Метод Якоби

def solve_jacobi(m, acc=10 ** (-7)):
    """
    :param m: Исходная матрица
    :param acc: Точность вычислений у метода Якоби
    """
    matrix = m
    m = np.array(matrix)
    x_right = m[:, -1]
    m = m[:, :-1]
    count = 0  # Кол-во интераций (если больше тысячи, то переходим к Гауссу)
    x_new = [1.0 for i in range(len(m[0]))]
    x_prev = [0.0 for i in range(len(m[0]))]
    delta = 1

    # Проверка диагонального преобладания
    for i in range(len(m)):
        summa = 0
        for j in range(len(m[i])):
            if i != j:
                summa += abs(m[i][j])
        if abs(m[i][i]) <= summa:
            print('Не соблюдается диагональное преобладание')
            print('Пробуем решить Якоби: ...')
            break

    while count < 1001:
        if count == 1000:
            print('Больше тысячи интераций, переходим к Гауссу')
            return solve_gauss(matrix)
        x_new = []
        count += 1

        for i in range(len(m)):
            dif = 0
            k = 0
            for el in m[i]:
                if el != m[i][i]:
                    dif -= el * x_prev[k]
                k += 1
            x_new.append((x_right[i] + dif) / m[i][i])

        flag = is_need_to_complete(x_prev, x_new, acc)
        if flag:
            break
        x_prev = x_new

    return x_prev


def is_need_to_complete(x_prev, x_new, acc):
    """
        :param x_prev: Предыдущие решение СЛАУ
        :param x_new: Новое решение СЛАУ (последнее)
        :param acc: Точность вычислений у метода Якоби
    """
    sum_up = 0
    sum_low = 0
    for k in range(len(x_prev)):
        sum_up += (x_new[k] - x_prev[k]) ** 2
        sum_low += (x_new[k]) ** 2
    return ((sum_up / sum_low) ** 0.5) < acc


#if __name__ == '__main__':
    # print(solve_jacobi(m=[[2.6, -1.7, 2.5, 3.7], [1.5, 6.2, -2.9, 3.2], [2.8, -1.7, 3.8, 2.8]]))


#'9_input_matrix.py'

class Int(int):
    def __init__(self, number):
        int.__init__(self)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return multi_num(mat=other.matrix, number=self)


class Float(float):
    def __init__(self, number):
        float.__init__(self)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return multi_num(mat=other.matrix, number=self)


class Matrix:
    def __init__(self, height, width, matrix):
        self.height = height
        self.width = width
        self.matrix = matrix

    def __mul__(self, other):
        assert (isinstance(other, Matrix) or isinstance(other, int) or isinstance(other, float) or isinstance(other, complex)), 'Не тот тип'
        if isinstance(other, int):
            return multi_num(mat=self.matrix, number=other)
        if isinstance(other, float):
            return multi_num(mat=self.matrix, number=other)
        if isinstance(other, complex):
            return multi_num(mat=self.matrix, number=other)
        if isinstance(other, Matrix):
            if self.width != other.height:
                print('Не совпадают размеры матриц при умножении их друг на друга'.upper())
                assert self.width == other.height, 'Разное кол-во столбцов левой матрица и кол-ва строк правой'
            else:
                return multi_mat(mat1=self.matrix, mat2=other.matrix)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            print('Вы складываете матрицу не с матрицей!'.upper())
            assert isinstance(other, Matrix), 'Вы используете не матрицу'
        if not (self.height == other.height and self.width == other.width):
            print('Матрицы при сложении имеют разные размеры!'.upper())
            assert (self.height == other.height and self.width == other.width), 'Разные длины/высоты'
        else:
            return add_mat(mat1=self.matrix, mat2=other.matrix)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            print('Вы складываете матрицу не с матрицей!'.upper())
            assert isinstance(other, Matrix), 'Вы используете не матрицу'
        if not (self.height == other.height and self.width == other.width):
            print('Матрицы при сложении имеют разные размеры!'.upper())
            assert (self.height == other.height and self.width == other.width), 'Разные длины/высоты'
        else:
            return sub_mat(mat1=self.matrix, mat2=other.matrix)

    def __truediv__(self, other):
        print('Нельзя делить матрицы друг на друга или на число!'.upper())
        assert 1 == 2

    def __pow__(self, power, modulo=None):
        print('Нельзя возводить матрицу в степень'.upper())
        assert 1 == 2

    def __floordiv__(self, other):
        print('Нельзя делить матрицы друг на друга или на число!'.upper())
        assert 1 == 2

    def __mod__(self, other):
        print('Нельзя делить матрицы друг на друга или на число!'.upper())
        assert 1 == 2

    @property
    def trans(self):
        return transposing(mat=self.matrix)

    @property
    def det(self):
        return compute_det(self.matrix)


def multi_num(mat, number) -> Matrix:
    """
    :param mat: матрица, которую собираемся умножать на число
    :param number: число, на которое нужно умножить матрицу
    :return: результ поэлементного умножения на число
    """
    for row in range(len(mat)):
        mat[row] = list(map(lambda x: x * number, mat[row]))
    return Matrix(len(mat), len(mat[0]), mat)


def add_mat(mat1, mat2) -> Matrix:
    """
    :param mat1: первая матрица
    :param mat2: вторая матрица
    :return: результат поэлементного сложения левой и правой матриц
    """
    for row in range(len(mat1)):
        try:
            mat1[row] = list(map(lambda x, y: x + y, mat1[row], mat2[row]))
        except TypeError:
            print('В сложении матриц есть недопустимые операции!'.upper())
            assert 1 == 2
    return Matrix(len(mat1), len(mat1[0]), mat1)


def sub_mat(mat1, mat2) -> Matrix:
    """
    :param mat1: первая матрица
    :param mat2: вторая матрица
    :return: результат поэлементного вычитания второй матрицы из первой
    """
    for row in range(len(mat1)):
        try:
            mat1[row] = list(map(lambda x, y: x - y, mat1[row], mat2[row]))
        except TypeError:
            print('В вычитании матриц есть недопустимые операции!'.upper())
            assert 1 == 2
    return Matrix(len(mat1), len(mat1[0]), mat1)


def get_column(mat, col: int) -> list:
    """
    :param mat: матрица, из которой нужно взять столбец
    :param col: номер столбца, который нужно вычленить, отсчет от нуля
    :return: столбец в виде массива
    """
    res = []
    for row in range(len(mat)):
        res.append(mat[row][col])
    return res


def multi_mat(mat1, mat2) -> Matrix:
    """
    :param mat1: левая матрица
    :param mat2: правая матрица
    :return: результат перемножения левой матрицы на правую
    """
    res = []
    for row in range(0, len(mat1)):
        zaglushka = []
        for col in range(0, len(mat2[0])):
            try:
                multi_row_col = list(map(lambda x, y: x * y, mat1[row], get_column(mat=mat2, col=col)))
            except TypeError:
                print('При умножении матрицы на матрицу есть недопустимое перемножение элементов!'.upper())
                assert 1 == 2
            try:
                numb = sum(multi_row_col)
            except TypeError:
                print('При сложении элементов в умножении матриц есть недопустимые операции!'.upper())
                assert 1 == 2
            zaglushka.append(numb)
        res.append(zaglushka)
    return Matrix(len(res), len(res[0]), res)


def input_expression(t=1):
    if t == 1:
        flag_for_exp = None
        while flag_for_exp is None:
            string = input('Введите матричное выражение: ')

            # здесь будет вычленение всех комплексных чисел
            # ...
            # ...
            letters_mod = ''.join(['A', 'P', 'O', 'X', 'K', 'F', 'S', 'H', 'Z', 'W', 'D',
                                   'L', 'V', 'G', 'C', 'N', 'M', 'T', 'Q', 'U', 'B', 'Y', 'E', 'R'])
            letters_mod = letters_mod.lower()
            for i in string:
                if i == 'i':
                    string = string.replace(i, 'j')
                elif i in letters_mod:
                    string = string.replace(i, i.upper())

            letters = frozenset({'I', 'A', 'P', 'O', 'X', 'K', 'J', 'F', 'S', 'H', 'Z', 'W', 'D',
                                 'L', 'V', 'G', 'C', 'N', 'M', 'T', 'Q', 'U', 'B', 'Y', 'E', 'R'})
            our_letters = []
            for i in string:
                if i in letters:
                    our_letters.append(i)

            # сразу проверим правильность введенного выражения
            for i in set(our_letters):
                exec(f'{i} = 1')
            try:
                eval(string)
            except Exception:
                print('Синтаксическая ошибка'.upper())
                continue
            # дальше заходим, если с выражением всё норм
            # теперь заменяем все 5 на Int(5)

            letters_for_replace = '+-*=_^!@#$%&()/'
            string_new = string
            for let in string:
                if let in letters_for_replace or let in letters:
                    string_new = string_new.replace(let, '')

            string_new = string_new.split(' ')
            for i in set(string_new):
                if i.isdigit():
                    string = string.replace(i, f'Int({i})')
                else:
                    try:
                        float(i)
                    except ValueError:
                        continue
                    else:
                        string = string.replace(i, f'Float({i})')
            print(string)

            # самое главное - ввод матриц
            flag_for_matrix = None
            while flag_for_matrix is None:
                for i in set(our_letters):
                    print(f'''
            Каким образом Вы хотите ввести матрицу {i}?
            1 - вручную
            2 - сгенерировать случайным образом
            3 - взять имеющуюся матрицу''')
                    typ = None
                    while typ is None:
                        try:
                            typ = int(input())
                        except ValueError:
                            print('Введите число в правильном формате')
                            continue
                        if typ in (1, 2, 3):
                            break

                    if typ == 1:
                        rows = int(input(f'Введите количество строк матрицы {i}: '))
                        columns = int(input(f'Введите количество столбцов матрицы {i}: '))
                        matrix = []
                        for r in range(rows):
                            row = []
                            for c in range(columns):
                                el = input(f'Введите элемент {r + 1, c + 1} матрицы {i}: ')
                                if 'i' in el:
                                    el = el.replace('i', 'j')
                                    el = complex(el)
                                elif 'j' in el:
                                    el = complex(el)
                                elif el.isdigit():
                                    el = int(el)
                                else:
                                    try:
                                        float(el)
                                    except ValueError:
                                        pass
                                    else:
                                        el = float(el)
                                row.append(el)
                            matrix.append(row)
                        exec(f'{i} = Matrix({rows}, {columns}, {matrix})')

                    elif typ == 2:
                        rows = int(input(f'Введите количество строк матрицы {i}: '))
                        columns = int(input(f'Введите количество столбцов матрицы {i}: '))
                        matrix = matrix_generator(rows, columns)
                        exec(f'{i} = Matrix({rows}, {columns}, {matrix})')

                    elif typ == 3:
                        # Пока разрабатывается
                        pass

                try:
                    eval(string)
                except Exception:
                    print('ВВЕДИТЕ МАТРИЦЫ ЗАНОВО!')
                    continue
                else:
                    flag_for_matrix = True

            flag_for_exp = True

        return eval(string).matrix

    elif t == 2:
        # транспонирование
        typ = None
        while typ is None:
            try:
                n = int(input('Введите кол-во строк матрицы: '))
                m = int(input('Введите кол-во столбцов матрицы: '))
            except ValueError:
                print('Введите корректные данные')
                continue
            typ = True

        matrix = []
        for i in range(n):
            row = []
            for j in range(m):
                el = input(f'Введите элемент ({i}, {j}):')
                if 'i' in el:
                    el = el.replace('i', 'j')
                    el = complex(el)
                elif 'j' in el:
                    el = complex(el)
                elif el.isdigit():
                    el = int(el)
                else:
                    try:
                        float(el)
                    except ValueError:
                        el = str(el)
                    else:
                        el = float(el)
                row.append(el)
            matrix.append(row)
        matrix = Matrix(n, m, matrix)
        return matrix.trans

    elif t == 3:
        # детерминант
        typ = None
        while typ is None:
            try:
                n = int(input('Введите кол-во строк матрицы: '))
            except ValueError:
                print('Введите корректные данные')
                continue
            typ = True

        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                flag = False
                while not flag:
                    el = input(f'Введите элемент ({i}, {j}): ')
                    if 'i' in el:
                        el = el.replace('i', 'j')
                        el = complex(el)
                    elif 'j' in el:
                        s_t = el.split(' ')

                        prom = el[:el.find('j') + 1]
                        if prom != el:
                            el = s_t[2] + s_t[1] + s_t[0]
                        el = complex(el)
                    elif el.isdigit():
                        el = int(el)
                    else:
                        try:
                            float(el)
                        except ValueError:
                            continue
                        else:
                            el = float(el)
                    flag = True
                row.append(el)
            matrix.append(row)

        matrix = Matrix(n, n, matrix)
        return matrix.det

    elif t == 4:
        # обусловленность матрицы
        typ = None
        while typ is None:
            try:
                n = int(input('Введите кол-во строк матрицы: '))
            except ValueError:
                print('Введите корректные данные')
                continue
            typ = True

        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                flag = False
                while not flag:
                    el = input(f'Введите элемент ({i}, {j}): ')
                    if 'i' in el:
                        el = el.replace('i', 'j')
                        el = complex(el)
                    elif 'j' in el:
                        s_t = el.split(' ')

                        prom = el[:el.find('j') + 1]
                        if prom != el:
                            el = s_t[2] + s_t[1] + s_t[0]
                        el = complex(el)
                    elif el.isdigit():
                        el = int(el)
                    else:
                        try:
                            float(el)
                        except ValueError:
                            continue
                        else:
                            el = float(el)
                    flag = True
                row.append(el)
            matrix.append(row)

        return conditionality(np.array(matrix))

    elif t == 5:
        # решение СЛАУ
        typ = None
        while typ is None:
            try:
                n = int(input('Введите кол-во строк матрицы: '))
            except ValueError:
                print('Введите корректные данные')
                continue
            typ = True
        print('Введите данные с учётом столбца коэфициентов')
        matrix = []
        for i in range(n):
            row = []
            for j in range(n + 1):
                flag = False
                while not flag:
                    el = input(f'Введите элемент ({i}, {j}): ')
                    if 'i' in el:
                        el = el.replace('i', 'j')
                        el = complex(el)
                    elif 'j' in el:
                        s_t = el.split(' ')

                        prom = el[:el.find('j') + 1]
                        if prom != el:
                            el = s_t[2] + s_t[1] + s_t[0]
                        el = complex(el)
                    elif el.isdigit():
                        el = int(el)
                    else:
                        try:
                            float(el)
                        except ValueError:
                            continue
                        else:
                            el = float(el)
                    flag = True
                row.append(el)
            matrix.append(row)

        matrix = np.array(matrix)
        num = conditionality(matrix[:,:-1])
        if num < 100:
            result = solve_jacobi(matrix)
        elif 100 <= num < 1000:
            result = solve_gauss(matrix)
        else:
            result = solve_gauss_fractions(matrix)
        print('A= ', matrix, '\n', 'A^(-1) = ', np.linalg.inv(matrix[:, :-1]), '\n', 'X= ', result, '\n', 'Обусловленность = ', num)


#'10_work_start.py'
# Легенда данной программы
def matrix_calc():
    '''The main function. At startup, a message appears with options for the available operations.
     The following operations are currently supported:
     1) Processing a matrix expression
     2) Calculation of the transposed matrix
     3) Calculation the determinant of the matrix
     4) Calculation the conditionality of the matrix
     5) SLAE solution
    '''
    print('''Добро пожаловать в инструмент по работе с матрицами!
    На данный момент поддерживаются следующие операции:
    1)Обработка матричного выражения
    2)Расчёт транспонированной матрицы
    3)Вычислить определитель матрицы
    4)Вычислить обусловленность матрицы
    5)Решение СЛАУ''')

    typ = None
    while typ is None:
        try:
            typ = int(input('Какой функционал требуется: '))
        except ValueError:
            print('Введите число в правильном формате')
            continue
        if typ in (1, 2, 3):
            break

    if typ == 1:
        print('''Введите выражение, с использованием матриц.
    В качестве обозначения можно использовать любую заглавную латинскую букву.
    Поддерживаются операции сложения, вычитания, умножения матриц, а также умножение матрицы на число''')
        print(input_expression(t=1))

    elif typ == 2:
        print('''Введите матрицу.
    Программа вернёт вам транспонированную матрицу''')
        ans = input_expression(t=2)
        for i in ans:
            print(i)

    elif typ == 3:
        print('''Введите матрицу.
    Программа вернёт вам определитель этой матрицы''')
        print(input_expression(t=3))

    elif typ == 4:
        print('''Введите матрицу.
    Программа вычислит обусловленность этой матрицы''')
        print(input_expression(t=4))

    elif typ == 5:
        print('''Введите СЛАУ в виде матрицы.
    Программа вернёт решение этой системы''')
        input_expression(t=5)


#'11_graphics_conditionality.py'
'matrix = [[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]] для примера'


def graphics_conditionality(matrix, accuracy=3):
    '''
    plotting the conditionality graph

    :param matrix: source matrix
    :param accuracy: decimal point precision (min=0, max=14)
    '''
    # Матрица для которой я буду строить график - new_matrix
    new_matrix = []
    for stroka in matrix:
        s = []
        for el in stroka:
            el += float('0.' + str(ra.randint(12345678912345, 88888888888888)))
            s.append(el)
        new_matrix.append(s)

    # Строим график (Максимальная точность - 14, минимальная точность 0)
    accuracy_x = []
    conditionality_y = []
    for i in range(0, accuracy + 1):
        matrix = []
        for stroka in new_matrix:
            s = []
            for el in stroka:
                s.append(round(el, i))
            matrix.append(s)
        conditionality_y.append(conditionality(matrix))
        accuracy_x.append(i)

    fig, ax = plt.subplots()
    ax.plot(accuracy_x, conditionality_y, 'co-')
    ax.set(xlabel='Точность знаков после запятой', ylabel="Обусловленность матрицы")
    plt.title('Обусловленность от точности знаков после запятой', loc='center', pad=10)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    plt.show()

#'12_compare_SofALE.py'
def compare_SofALE():
    time_our = []
    time_np = []
    size_m = []  # 2*3, 3*4, 4*5, 5*6,  и т.д.

    def comparison(matrix):
        M = Matrix(matrix)
        system = A, b = M[:, :-1], M[:, -1]
        alfa = []
        alfa1 = []
        for i in range(1, len(matrix) + 1):
            alfa.append('x' + str(i))
        for el in alfa:
            el = symbols(str(el))
            alfa1.append(el)
        return linsolve(system, alfa1)

    for i in range(2, 10):
        m = matrix_generator(i, i + 1)
        time_start = time.time()
        b = comparison(m)  # Sympy
        elapsed_time = time.time() - time_start
        time_np.append(elapsed_time)

        time_start = time.time()
        a = solve_jacobi(m)  # Наше решение СЛАУ
        elapsed_time = time.time() - time_start
        time_our.append(elapsed_time)

        size_m.append(i)

    fig, ax = plt.subplots()
    ax.plot(size_m, time_our, 'co-', label='Наше решение СЛАУ')
    ax.plot(size_m, time_np, 'r--', label='Sympy')
    ax.set(xlabel='Размер матрицы n*(n+1)', ylabel="Время в секундах")
    ax.legend(loc='upper left')
    plt.title('Скорость выполнения функции', loc='center', pad=10)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    plt.show()

#'12_compare_det.py'
def compare_det():
    time_our = []
    time_np = []
    size_m = []  # 1*1 , 2*2, 3*3, 4*4, 5*5,  и т.д.

    def comparison(matrix):
        a = np.array(matrix)
        return LA.det(a)

    for i in range(1, 10):
        m = matrix_generator(i, i)
        time_start = time.time()
        b = comparison(m)  # Опр numpy
        elapsed_time = time.time() - time_start
        time_np.append(elapsed_time)

        time_start = time.time()
        a = compute_det(m)  # Наше транспонирование
        elapsed_time = time.time() - time_start
        time_our.append(elapsed_time)

        size_m.append(i)

    fig, ax = plt.subplots()
    ax.plot(size_m, time_our, 'co-', label='Наше транспонирование')
    ax.plot(size_m, time_np, 'r--', label='Numpy')
    ax.set(xlabel='Размер матрицы x*x', ylabel="Время в секундах")
    ax.legend(loc='upper left')
    plt.title('Скорость выполнения функции', loc='center', pad=10)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    plt.show()
