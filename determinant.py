def get_matrix_minor(mat, column, row=0):
    """
    :param row: строка, от которой избавляемся. По дефолту равна 0
    :param mat: матрица, для которой нужно посчитать минор
    :param column: столбец, от которого нужно избавиться, индексация с нуля
    :return: минор матрицы
    """
    mat.pop(row)
    for el in range(0, len(mat)):
        mat[el].pop(column)
    return mat


def compute_det(matrix) -> int:
    """
    :param matrix: квадратная матрица
    :return: определитель матрицы
    """
    if len(matrix) == 1:
        return matrix[0][0]
    else:
        # return sum([((-1)**j * matrix[0][j] * compute_det(get_matrix_minor(matrix, j))) for j in range(0, len(matrix))])
        count = 0
        plan = []
        for item in matrix[0]:
            plan.append((-1) ** count * item * compute_det(get_matrix_minor(matrix, count)))
            count += 1
        return sum(plan)


if __name__ == '__main__':
    print(compute_det(matrix=[[1, 2], [3, 4]]))
    # print(get_matrix_minor(mat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], column=0))

