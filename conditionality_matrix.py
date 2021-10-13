# Обусловленность матрицы
import numpy as np
from matrix_norms import infinity_norm
from matrix_norms import first_norm
from matrix_norms import second_norm

def conditionality(matrix):
    """
        :param matrix: матрица, для которой мы будем считать обусловленность
    """
    # Считаем обратную матрицу с помощью numpy.linalg.inv()
    matrix_reverse = np.linalg.inv(matrix)
    # Теперь находим обусловленность (я взял третий способ нахождения, нужно ли рассматривать все???)
    cond = second_norm(matrix) * second_norm(matrix_reverse)
    return cond

if __name__ == '__main__':
    print(conditionality(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))