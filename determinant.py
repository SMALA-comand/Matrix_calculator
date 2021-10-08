from copy import deepcopy


def get_matrix_minor(mat, column, row=0):
    """
    :param row: строка, от которой избавляемся. По дефолту равна 0
    :param mat: матрица, для которой нужно посчитать минор
    :param column: столбец, от которого нужно избавиться, индексация с нуля
    :return: минор матрицы
    """
    mat1 = deepcopy(mat)
    mat1.pop(row)
    for el in range(0, len(mat1)):
        mat1[el].pop(column)
    return mat1


def compute_det(matrix) -> int:
    """
    :param matrix: квадратная матрица
    :return: определитель матрицы
    """
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        return matrix[0][0] * matrix[1][1] * matrix [2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2]
    else:
        count = 0
        plan = []
        count_0 = [0, sum(matrix[0]), 0]  # Счётчик оптимизации с наибольшим чилом нулей и наименьшем модулем коэфициентов (Только для строки!)
        k = 0   # Счётчик строки
        for stroka in matrix:
            if count_0[0] < stroka.count(0):
                count_0[0] = stroka.count(0)
                count_0[2] = k
            elif abs(sum(stroka)) <= count_0[1] and count_0[0] == stroka.count(0):
                    count_0[1] = abs(sum(stroka))
                    count_0[2] = k
            k += 1

        for item in matrix[count_0[2]]:
            plan.append((-1) ** count * item * compute_det(get_matrix_minor(matrix, count)))
            count += 1
        return sum(plan)


if __name__ == '__main__':
    print(compute_det(matrix=[[1, 2, 333], [42, 5, 6], [7, 8, 9]]))
    # print(get_matrix_minor(mat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], column=0))
