from copy import deepcopy


def get_matrix_minor(mat, column, row=0):
    """
    :param row: строка, от которой избавляемся. По дефолту равна 0
    :param mat: матрица, для которой нужно посчитать минор
    :param column: столбец, от которого нужно избавиться, индексация с нуля
    :return: минор матрицы
    """
    print(mat, "**")
    mat1 = deepcopy(mat)
    mat1.pop(row)
    for el in range(0, len(mat1)):
        mat1[el].pop(column)
    print(mat1, "!!")
    return mat1


def compute_det(matrix) -> int:
    """
    :param matrix: квадратная матрица
    :return: определитель матрицы
    """
    if len(matrix) == 2:
        print("exit")
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        # return sum([((-1)**j * matrix[0][j] * compute_det(get_matrix_minor(matrix, j))) for j in range(0, len(matrix))])
        print(matrix)
        count = 0
        plan = []
        for item in matrix[0]:
            plan.append((-1) ** count * item * compute_det(get_matrix_minor(matrix, count)))
            count += 1
            print(plan, count, len(matrix))
        return sum(plan)


if __name__ == '__main__':
    print(compute_det(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))
    # print(get_matrix_minor(mat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], column=0))
