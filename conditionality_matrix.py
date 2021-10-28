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
    try:
        matrix_reverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return 'Singular matrix'
    # Теперь находим обусловленность (я взял третий способ нахождения, нужно ли рассматривать все???)
    cond = first_norm(matrix) * first_norm(matrix_reverse)
    return cond


if __name__ == '__main__':
    print(conditionality(matrix=[[1, 2], [3, 4]]))
