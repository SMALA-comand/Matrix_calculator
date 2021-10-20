import  numpy as np

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
            div = m[i][k] / m[k][k]
            m[i][-1] -= div * m[k][-1]
            for j in range(k, n):
                m[i][j] -= div * m[k][j]

    # Проверяем, имеет ли система конечное число решений
    if is_singular(m):
        print('Система имеет бесконечное количество решений ¯\_(ツ)_/¯ ')
        return

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


if __name__ == '__main__':
    print(solve_gauss(m = [[2.6,-1.7,2.5,3.7],[1.5,6.2,-2.9,3.2],[2.8,-1.7,3.8,2.8]]))
    #print(solve_gauss(m = [[1,2,3,6],[2,3,1,6],[3,1,2,6]]))