# pip install sympy - НАДО ввести в консоль и установить !!!

import numpy as np
from numpy import linalg as LA
from matrix_generator import matrix_generator
import time
import matplotlib.pyplot as plt
from determinant import compute_det
from Jacobi_SofALE import *
from sympy import *
from sympy.solvers.solveset import linsolve

time_our = []
time_np = []
size_m = []  # 2*3, 3*4, 4*5, 5*6,  и т.д.


def comparison(matrix):
    M = Matrix(matrix)
    system = A, b = M[:, :-1], M[:, -1]
    alfa = []
    alfa1 = []
    for i in range(1, len(matrix)+1):
        alfa.append('x'+str(i))
    for el in alfa:
        el = symbols(str(el))
        alfa1.append(el)
    return linsolve(system, alfa1)


for i in range(2, 10):
    m = matrix_generator(i, i+1)
    time_start = time.time()
    b = comparison(m)  # Sympy
    elapsed_time = time.time() - time_start
    time_np.append(elapsed_time)

    time_start = time.time()
    a = solve_jacobi(m)  # Наше решение СЛАУ
    elapsed_time = time.time() - time_start
    time_our.append(elapsed_time)

    size_m.append(i)


fig, ax = plt.subplots()
ax.plot(size_m, time_our, 'co-', label='Наше решение СЛАУ')
ax.plot(size_m, time_np, 'r--', label='Sympy')
ax.set(xlabel='Размер матрицы n*(n+1)', ylabel="Время в секундах")
ax.legend(loc='upper left')
plt.title('Скорость выполнения функции', loc='center', pad=10)
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show()
