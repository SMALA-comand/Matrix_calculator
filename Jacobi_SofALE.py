# Метод Якоби
from Gauss_SofALE import solve_gauss
import numpy as np

def solve_jacobi(m, acc=10**(-6)):
    """
    :param m: Исходная матрица
    :param acc: Точность вычислений у метода Якоби
    """
    matrix = m
    m = np.array(matrix)
    x_right = m[:, -1]
    m = m[:, :-1]
    count = 0     # Кол-во интераций (если больше тысячи, то переходим к Гауссу)
    x_new = [1.0 for i in range(len(m[0]))]
    x_prev = [0.0 for i in range(len(m[0]))]
    delta = max(abs(max(x_prev) - min(x_new )),abs(max(x_new ) - min(x_prev)))

    while delta > acc:
        x_new = []
        count += 1
        if count == 1001:
            print('Больше тысячи интераций, переходим к Гауссу')
            #solve_gauss(matrix)
            return solve_gauss(matrix)
        for i in range(len(m)):
            dif = 0
            k = 0
            for el in m[i]:
                if el != m[i][i]:
                    dif -= el*x_prev[k]
                k += 1
            x_new.append((x_right[i]-dif)/m[i][i])
        x_prev = x_new
        delta = max(abs(max(x_prev) - min(x_new)), abs(max(x_new) - min(x_prev)))


if __name__ == '__main__':
    print(solve_jacobi(m = [[2.6,-1.7,2.5,3.7],[1.5,6.2,-2.9,3.2],[2.8,-1.7,3.8,2.8]]))