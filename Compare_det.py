# Сравнение скорости работы нашего детерминанта с numpy
import numpy as np
from numpy import linalg as LA
from matrix_generator import matrix_generator
import time
import matplotlib.pyplot as plt
from determinant import compute_det

time_our = []
time_np = []
size_m = []  # 1*1 , 2*2, 3*3, 4*4, 5*5,  и т.д.


def comparison(matrix):
    a = np.array(matrix)
    return LA.det(a)


for i in range(1, 10):
    m = matrix_generator(i, i)
    time_start = time.time()
    b = comparison(m)  # Опр numpy
    elapsed_time = time.time() - time_start
    time_np.append(elapsed_time)

    time_start = time.time()
    a = compute_det(m)  # Наше транспонирование
    elapsed_time = time.time() - time_start
    time_our.append(elapsed_time)

    size_m.append(i)


fig, ax = plt.subplots()
ax.plot(size_m, time_our, 'co-', label='Наше транмпонирование')
ax.plot(size_m, time_np, 'r--', label='Numpy')
ax.set(xlabel='Размер матрицы x*x', ylabel="Время в секундах")
ax.legend(loc='upper left')
plt.title('Скорость выполнения функции', loc='center', pad=10)
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show()
