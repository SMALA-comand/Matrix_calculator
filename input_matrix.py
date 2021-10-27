from transpose_matrix import transposing
from matrix_generator import matrix_generator
from determinant import compute_det
#from matrix_examples import *
from conditionality_matrix import *
from Gauss_SofALE import *
from Jacobi_SofALE import *
import numpy as np


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


