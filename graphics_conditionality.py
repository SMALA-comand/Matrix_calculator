from conditionality_matrix import conditionality
import random
import numpy as np
import matplotlib.pyplot as plt

# Матрица для которой я буду строить график - new_matrix
matrix=[[10, 2, 5, 60], [7, 9, 0, 78], [15, 22, 65, 111], [50, 100, 17, 33]]
new_matrix = []
for stroka in matrix:
    s = []
    for el in stroka:
        el += float('0.'+ str(random.randint(12345678912345,88888888888888)))
        s.append(el)
    new_matrix.append(s)

# Строим график (Максимальная точность - 14, минимальная точность 0)
accuracy_x = []
conditionality_y = []
for i in range(0,15):
    matrix = []
    for stroka in new_matrix:
        s = []
        for el in stroka:
            s.append(round(el,i))
        matrix.append(s)
    conditionality_y.append(conditionality(matrix))
    accuracy_x.append(i)


fig, ax = plt.subplots()
ax.plot(accuracy_x, conditionality_y, 'co-')
ax.set(xlabel = 'Точность знаков после запятой', ylabel = "Обусловленность матрицы")
plt.title('Обусловленность от точности знаков после запятой',loc = 'center', pad = 10 )
fig.set_figwidth(12)
fig.set_figheight(6)
plt.show()
