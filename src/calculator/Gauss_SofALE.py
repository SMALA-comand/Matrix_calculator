import numpy as np


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


if __name__ == '__main__':
    print(solve_gauss_fractions(m=[[2.6, -1.7, 2.5, 3.7], [1.5, 6.2, -2.9, 3.2], [2.8, -1.7, 3.8, 2.8]]))
    # print(solve_gauss_fractions(m = [[1,2,3,6],[2,3,1,6],[3,1,2,6]]))
    # print(solve_gauss_fractions(m=[[0, 0, 0, 1], [0, 0, 0,0], [0, 0, 0, 0]]))
