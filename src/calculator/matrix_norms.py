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


if __name__ == '__main__':
    print(infinity_norm(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))
