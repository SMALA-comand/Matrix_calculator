import  numpy as np
from fractions import Fraction

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
    if m[-2][:-1] == m[-1][:-1] or (m[-1][-1] !=0 and np.sum(m[-1][:-1]) == 0):
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
                div = Fraction(Fraction(m[i][k]).limit_denominator(10**9), Fraction(m[k][k]).limit_denominator(10**9))
                m[i][-1] -= div * m[k][-1]
                for j in range(k, n):
                    m[i][j] -= div * m[k][j]

    # Проверяем, имеет ли система конечное число решений
    if np.sum(m[:][:]) == 0:
        return m[:][1]
    if m[-2][:-1] == m[-1][:-1] or (m[-1][-1] !=0 and np.sum(m[-1][:-1]) == 0):
        x = 'Система не имеет решений ¯\_(ツ)_/¯ '
        return x
    elif is_singular(m):
        x = 'Система имеет бесконечное количество решений '
        return x

    # Обратный ход
    x = [0 for i in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = Fraction(Fraction((m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)]))).limit_denominator(10**9),Fraction(m[k][k]).limit_denominator(10**9))

    return x

# Обратная матрица методом Гаусса Жордана
def inverse_matrix_gauss(matrix_origin):
    """
        :param matrix_origin: Исходная матрица
    """
    # Склеиваем 2 матрицы: слева - первоначальная, справа - единичная
    matrix_origin = np.array(matrix_origin)
    n = matrix_origin.shape[0]
    m = np.hstack((matrix_origin, np.eye(n)))

    # Прямой ход
    for nrow, row in enumerate(m):  # nrow равен номеру строки, row содержит саму строку матрицы
        divider = row[nrow]  # диагональный элемент
        row /= divider
        # Теперь вычитаем приведённую строку из всех нижележащих строк
        for lower_row in m[nrow + 1:]:
            factor = lower_row[nrow]  # элемент строки в колонке nrow
            lower_row -= factor * row  # Зануляем все оставшиеся элементы в колонке

    # Обратный ход:
    for k in range(n - 1, 0, -1):
        for row_ in range(k - 1, -1, -1):
            if m[row_, k]:
                # Все элементы выше главной диагонали делаем равными нулю
                m[row_, :] -= m[k, :] * m[row_, k]
    return m[:, n:].copy()

# Обратная матрица методом Гаусса Жордана + fraction
def inverse_matrix_gauss_fraction(matrix_origin):
    from fractions import Fraction
    """
        :param matrix_origin: Исходная матрица
    """
    # Склеиваем 2 матрицы: слева - первоначальная, справа - единичная
    matrix_origin = np.array(matrix_origin)
    n = matrix_origin.shape[0]
    m = np.hstack((matrix_origin, np.eye(n)))

    # Прямой ход
    for nrow, row in enumerate(m):  # nrow равен номеру строки, row содержит саму строку матрицы
        divider = row[nrow]  # диагональный элемент
        row = Fraction(Fraction(row).limit_denominator(10 ** 9), Fraction(divider).limit_denominator(10 ** 9))
        # Теперь вычитаем приведённую строку из всех нижележащих строк
        for lower_row in m[nrow + 1:]:
            factor = lower_row[nrow]  # элемент строки в колонке nrow
            lower_row -= factor * row  # Зануляем все оставшиеся элементы в колонке

    # Обратный ход:
    for k in range(n - 1, 0, -1):
        for row_ in range(k - 1, -1, -1):
            if m[row_, k]:
                # Все элементы выше главной диагонали делаем равными нулю
                m[row_, :] -= m[k, :] * m[row_, k]
    return m[:, n:].copy()



if __name__ == '__main__':
    print(inverse_matrix_gauss([[2.6,-1.7,2.5],[1.5,6.2,-2.9],[2.8,-1.7,3.8]]))
    #print(solve_gauss_fractions(m = [[1,2,3,6],[2,3,1,6],[3,1,2,6]]))
    #print(solve_gauss_fractions(m=[[0, 0, 0, 1], [0, 0, 0,0], [0, 0, 0, 0]]))