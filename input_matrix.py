from transpose_matrix import transposing


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

    @property
    def trans(self):
        return transposing(mat=self.matrix)


def multi_num(mat, number):
    """
    :param mat: матрица, которую собираемся умножать на число
    :param number: число, на которое нужно умножить матрицу
    :return: результ поэлементного умножения на число
    """
    for row in range(len(mat)):
        mat[row] = list(map(lambda x: x * number, mat[row]))
    return mat


def add_mat(mat1, mat2):
    """
    :param mat1: первая матрица
    :param mat2: вторая матрица
    :return: результат поэлементного сложения левой и правой матриц
    """
    for row in range(len(mat1)):
        mat1[row] = list(map(lambda x, y: x + y, mat1[row], mat2[row]))
    return mat1


def sub_mat(mat1, mat2):
    """
    :param mat1: первая матрица
    :param mat2: вторая матрица
    :return: результат поэлементного вычитания второй матрицы из первой
    """
    for row in range(len(mat1)):
        mat1[row] = list(map(lambda x, y: x - y, mat1[row], mat2[row]))
    return mat1


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


def multi_mat(mat1, mat2):
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
    return res


def input_expression():
    # TODO(Mark): обработать все возможные ошибки
    string = input()
    string = string.upper()
    letters = frozenset({'I', 'A', 'P', 'O', 'X', 'K', 'J', 'F', 'S', 'H', 'Z', 'W', 'D',
                         'L', 'V', 'G', 'C', 'N', 'M', 'T', 'Q', 'U', 'B', 'Y', 'E', 'R'})
    our_letters = []
    for i in string:
        if i in letters:
            our_letters.append(i)

    letters_for_replace = '+-*=_^!@#$%&()/'
    string_new = string
    for let in string:
        if let in letters_for_replace or let in letters:
            string_new = string_new.replace(let, '')

    string_new = string_new.split(' ')
    for i in string_new:
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

    for i in our_letters:
        rows = int(input(f'Введите количество строк матрицы {i}: '))
        columns = int(input(f'Введите количество столбцов матрицы {i}: '))
        matrix = []
        for r in range(rows):
            row = []
            for c in range(columns):
                element = input(f'Введите элемент {r + 1, c + 1}: ')
                row.append(int(element))
            matrix.append(row)

        exec(f'{i} = Matrix({rows}, {columns}, {matrix})')
    return eval(string)


print(input_expression())
