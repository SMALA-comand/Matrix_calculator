from transpose_matrix import transposing
from matrix_generator import matrix_generator
from determinant import compute_det
from matrix_examples import *


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
        assert (isinstance(other, Matrix) or isinstance(other, int) or isinstance(other, float)), 'Не тот тип'
        if isinstance(other, int):
            return multi_num(mat=self.matrix, number=other)
        if isinstance(other, float):
            return multi_num(mat=self.matrix, number=other)
        if isinstance(other, Matrix):
            assert self.width == other.height, 'Разное кол-во столбцов левой матрица и кол-ва строк правой'
            return multi_mat(mat1=self.matrix, mat2=other.matrix)

    def __add__(self, other):
        assert isinstance(other, Matrix), 'Вы используете не матрицу'
        assert (self.height == other.height and self.width == other.width), 'Разные длины/высоты'
        return add_mat(mat1=self.matrix, mat2=other.matrix)

    def __sub__(self, other):
        assert isinstance(other, Matrix), 'Вы используете не матрицу'
        assert (self.height == other.height and self.width == other.width), 'Разные длины/высоты'
        return sub_mat(mat1=self.matrix, mat2=other.matrix)

    def __truediv__(self, other):
        print('Нельзя делить матрицы друг на друга или на число!')

    def __pow__(self, power, modulo=None):
        print('Нельзя возводить матрицу в степень')

    def __floordiv__(self, other):
        print('Нельзя делить матрицы друг на друга или на число!')

    def __mod__(self, other):
        print('Нельзя делить матрицы друг на друга или на число!')

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
        mat1[row] = list(map(lambda x, y: x + y, mat1[row], mat2[row]))
    return Matrix(len(mat1), len(mat1[0]), mat1)


def sub_mat(mat1, mat2) -> Matrix:
    """
    :param mat1: первая матрица
    :param mat2: вторая матрица
    :return: результат поэлементного вычитания второй матрицы из первой
    """
    for row in range(len(mat1)):
        mat1[row] = list(map(lambda x, y: x - y, mat1[row], mat2[row]))
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
            multi_row_col = list(map(lambda x, y: x * y, mat1[row], get_column(mat=mat2, col=col)))
            numb = sum(multi_row_col)
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
            string = string.upper()
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
            except:
                print('Синтаксическая ошибка')
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
                                element = input(f'Введите элемент {r + 1, c + 1}: ')
                                try:
                                    float(element)
                                except ValueError:
                                    row.append(element)
                                else:
                                    row.append(float(element))
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
                    print(Exception)
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

        print('Если хотите ввести числовую матрицу, то введите 1. Комплексную - 2. Любую другую - 3.')
        typ = None
        while typ is None:
            try:
                ans = int(input('Введите ответ: '))
            except ValueError:
                print('Введите корректные данные')
                continue
            if ans in (1, 2, 3):
                break

        matrix = []
        if ans == 1:
            for i in range(n):
                row = []
                for j in range(m):
                    typ = None
                    while typ is None:
                        try:
                            el = float(input(f'Введите элемент {i+1, j+1}: '))
                        except ValueError:
                            print('Введите корректные данные')
                            continue
                        typ = True
                    row.append(el)
                matrix.append(row)
            matrix = Matrix(n, m, matrix)
            return matrix.trans

        elif ans == 2:
            for i in range(n):
                row = []
                for j in range(m):
                    typ = None
                    while typ is None:
                        try:
                            el = complex(input(f'Введите элемент {i+1, j+1}: '))
                        except ValueError:
                            print('Введите корректные данные')
                            continue
                        typ = True
                    row.append(el)
                matrix.append(row)
            matrix = Matrix(n, m, matrix)
            return matrix.trans

        elif ans == 3:
            for i in range(n):
                row = []
                for j in range(m):
                    el = input(f'Введите элемент {i+1, j+1}: ')
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
                typ = None
                while typ is None:
                    try:
                        el = float(input(f'Введите элемент {i+1, j+1}: '))
                    except ValueError:
                        print('Введите корректные данные')
                        continue
                    typ = True
                row.append(el)
            matrix.append(row)
        matrix = Matrix(n, n, matrix)
        return matrix.det
