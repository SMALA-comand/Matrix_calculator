from copy import deepcopy

dict = {}

def matr_to_string(matrix):
    s = ""
    for row in matrix:
        for i in range(0, len(matrix)):
            s += str(row[i])
    return s
def get_matrix_minor(mat, column = 0, row= 0):
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

def find_best_var(matrix):
    #поиск по строке
    count_0_row = [0, sum([abs(matrix[0][i]) for i in range(0, len(matrix[0]))]), 0]  # Счётчик оптимизации с наибольшим чилом нулей и наименьшем модулем коэфициентов (Только для строки!)
    k = 0  # Счётчик строки
    for stroka in matrix:
        if count_0_row[0] < stroka.count(0):
            count_0_row[0] = stroka.count(0)
            count_0_row[2] = k

        elif count_0_row[0] == stroka.count(0):
            cur_sum = sum([abs(stroka[i]) for i in range(0, len(matrix))])
            if cur_sum <= count_0_row[1]:
                count_0_row[1] = cur_sum
                count_0_row[2] = k
        k += 1

    #поиск по столбцу
    count_0_col = [0, sum([abs(matrix[i][0]) for i in range(0, len(matrix[0]))]), 0]  # Счётчик оптимизации с наибольшим чилом нулей и наименьшем модулем коэфициентов (Только для строки!)
    for i in range(1, len(matrix)):
        stolb = [matrix[j][i] for j in range(0, len(matrix))]
        if count_0_col[0] < stolb.count(0):
            count_0_col[0] = stolb.count(0)
            count_0_col[2] = i

        elif count_0_col[0] == stolb.count(0):
            cur_sum = sum([abs(stolb[i]) for i in range(0, len(matrix))])
            if cur_sum <= count_0_col[1]:
                count_0_col[1] = cur_sum
                count_0_col[2] = i

    if count_0_row[0] > count_0_col[0]:
        return(count_0_row[2], 0)
    elif count_0_row[0] < count_0_col[0]:
        return(count_0_col[2], 1)
    elif count_0_row[1] > count_0_col[1]:
        return(count_0_col[2], 1)
    return (count_0_row[2], 0)

def compute_det(matrix) -> int:
    """
    :param matrix: квадратная матрица
    :return: определитель матрицы
    """
    if (len(matrix) == 1):
        return matrix[0][0]
    #У другой группы слышал нюанс, что выход осуществлять только при матрице размера 1x1. Думаю оставим всё как есть, а если попрост, быстро поменяем
    #if len(matrix) == 2 and len(matrix[0]) == 2:
    #    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    #elif len(matrix) == 3 and len(matrix[0]) == 3:
    #    return matrix[0][0] * matrix[1][1] * matrix [2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2]
    else:
        s = matr_to_string(matrix)
        if dict.get(s) != None:
            return dict[s]
        count = 0
        plan = []

        best_num, dim = find_best_var(matrix)
        if dim == 0:
            for item in matrix[best_num]:
                plan.append((-1) ** (count + best_num) * item * compute_det(get_matrix_minor(matrix, count,row = best_num)))
                count += 1
                #print(plan, best_num, "row", count, matrix)
        else:
            for item in range (len(matrix)):
                plan.append((-1) ** (best_num + item) * matrix[item][best_num] * compute_det(get_matrix_minor(matrix, column = best_num,row = item)))
                #print(plan, item, "col", best_num, matrix)
        a = sum(plan)
        dict[matr_to_string(matrix)] = a
        return a


if __name__ == '__main__':
    print(compute_det(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))
    # print(get_matrix_minor(mat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], column=0))

def numpy_linalg_det(matrix):
    import numpy
    return numpy.linalg.det(matrix)

#numpy_linalg_det([[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]])