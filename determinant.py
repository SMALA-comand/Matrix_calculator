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
  
    if len(matrix) == 2 and len(matrix[0]) == 2:
        print("exit")
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3 and len(matrix[0]) == 3:
        print('exit')
        return matrix[0][0] * matrix[1][1] * matrix [2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2] 
    else:
        # return sum([((-1)**j * matrix[0][j] * compute_det(get_matrix_minor(matrix, j))) for j in range(0, len(matrix))])
        print(matrix)
        count = 0
        plan = []
        count_0 = [0,sum(matrix[0]),0]  # Счётчик оптимизации с наибольшим чилом нулей и наименьшем модулем коэфициентов (Только для строки!)
        k = 0   #  Счётчик строки
        for stroka in matrix:
            if count_0[0] < stroka.count(0):
                count_0[0] = stroka.count(0)
                count_0[2] = k
            elif abs(sum(stroka)) <= count_0[1] and count_0[0] == stroka.count(0):
                    count_0[1] = abs(sum(stroka))
                    count_0[2] = k
            k +=1
        
        for item in matrix[count_0[2]]:
            plan.append((-1) ** count * item * compute_det(get_matrix_minor(matrix, count,row = count_0[2])))
            count += 1
            print(plan, count, len(matrix))
        return sum(plan)


if __name__ == '__main__':
    print(compute_det(matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]))
    # print(get_matrix_minor(mat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], column=0))

def numpy_linalg_det(matrix):
    import numpy
    return numpy.linalg.det(matrix)

numpy_linalg_det([[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]])


# Сравнение скорости работы нашего нахождения детерминанта с numpy
import time
from matrix_generator import matrix_generator
import matplotlib.pyplot as plt
time_our = []
time_np = []
size_m = [] # 1*1 , 2*2, 3*3, 4*4, 5*5,  и т.д.

for i in range(1, 8):
    m = matrix_generator(i,i)
    time_start  = time.time()
    numpy_linalg_det(m)  # Нахождение детерминанта numpy
    elapsed_time = time.time() - time_start
    time_np.append(elapsed_time)

    time_start  = time.time()
    compute_det(m)   # Наше нахождение детерминанта 
    elapsed_time = time.time() - time_start
    time_our.append(elapsed_time)
    
    size_m.append(i)

fig, ax = plt.subplots()
ax.plot(size_m, time_our, 'co-',label = 'Наше нахождение детерминанта')
ax.plot(size_m, time_np, 'r--',label = 'Numpy')
ax.set(xlabel = 'Размер матрицы x*x', ylabel = "Время в секундах")
ax.legend(loc = 'upper left')
plt.title('Скорость выполнения функции',loc = 'center', pad = 10 )
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show()