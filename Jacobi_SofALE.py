# Метод Якоби
from Gauss_SofALE import solve_gauss
from Gauss_SofALE import solve_gauss_fractions
import numpy as np

def solve_jacobi(m, acc=10**(-7)):
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
    delta = 1

    # Проверка диагонального преобладания
    for i in range(len(m)):
        summa = 0
        for j in range(len(m[i])):
            if i != j:
                summa += abs(m[i][j])
        if abs(m[i][i]) <= summa:
            print('Не соблюдается диагональное преобладание')
            print('Пробуем решить Якоби: ...')
            break

    while count < 1001:
        if count == 1000:
            print('Больше тысячи интераций, переходим к Гауссу')
            return solve_gauss(matrix)
        x_new = []
        count += 1

        for i in range(len(m)):
            dif = 0
            k = 0
            for el in m[i]:
                if el != m[i][i]:
                    dif -= el*x_prev[k]
                k += 1
            x_new.append((x_right[i]+dif)/m[i][i])

        flag = is_need_to_complete(x_prev, x_new, acc)
        if flag:
            break
        x_prev = x_new

    return x_prev

def is_need_to_complete(x_prev, x_new, acc):
    """
        :param x_prev: Предыдущие решение СЛАУ
        :param x_new: Новое решение СЛАУ (последнее)
        :param acc: Точность вычислений у метода Якоби
    """
    sum_up = 0
    sum_low = 0
    for k in range(len(x_prev)):
        sum_up += (x_new[k] - x_prev[k]) ** 2
        sum_low += (x_new[k]) ** 2
    return ((sum_up / sum_low)**0.5) < acc

if __name__ == '__main__':
    print(solve_jacobi(m = [[2.6,-1.7,2.5,3.7],[1.5,6.2,-2.9,3.2],[2.8,-1.7,3.8,2.8]]))