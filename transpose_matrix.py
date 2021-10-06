def get_col(mat, col):
    """
    :param mat: матрица, из которой нужно взять столбец
    :param col: номер столбца, который нужно вычленить, отсчет от нуля
    :return: столбец в виде массива
    """
    res = []
    for row in range(len(mat)):
        res.append(mat[row][col])
    return res


def transposing(mat):
    """
    :param mat: матрица произвольного размера, которую транспонируем
    :return: транспонированная матрица
    """
    row, col = len(mat), len(mat[0])
    if row == col:
        n = row
        for rows in range(0, n):
            for columns in range(rows, n):
                mat[rows][columns], mat[columns][rows] = mat[columns][rows], mat[rows][columns]
    else:
        ans = []
        for columns in range(0, col):
            ans.append(get_col(mat=mat, col=columns))
        mat = ans
    return mat


if __name__ == '__main__':
    print(transposing([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]))

# Сравнение скорости работы нашего транспонирования с numpy
import numpy as np
from matrix_generator import matrix_generator
import time
import matplotlib.pyplot as plt

time_our = []
time_np = []
size_m = [] # 1*1 , 2*2, 3*3, 4*4, 5*5,  и т.д.
def comparison(matrix):
    a = np.array(matrix)
    return a.transpose()

for i in range(1, 1001,10):
    m = matrix_generator(i,i)
    time_start  = time.time()
    comparison(m)
    elapsed_time = time.time() - time_start
    time_np.append(elapsed_time)

    time_start  = time.time()
    transposing(m)
    elapsed_time = time.time() - time_start
    time_our.append(elapsed_time)
    
    size_m.append(i)

fig, ax = plt.subplots()
ax.plot(size_m, time_our, 'co-',label = 'Наше транмпонирование')
ax.plot(size_m, time_np, 'r--',label = 'Numpy')
ax.set(xlabel = 'Размер матрицы x*x', ylabel = "Время в секундах")
ax.legend(loc = 'upper left')
plt.title('Скорость выполнения функции',loc = 'center', pad = 10 )
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show()