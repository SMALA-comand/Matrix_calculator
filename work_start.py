from input_matrix import *
#Легенда данной программы

print('''Добро пожаловать в инструмент по работе с матрицами!
На данный момент поддерживаются следующие операции:
1)Обработка матричного выражения
2)Рассчёт транспонированной матрицы
3)Вычислить определитель матрицы''')

typ = None
while typ == None:
    try:
        typ = int(input())
    except ValueError:
        print('Введите число в правильном формате')
        continue
    if typ in (1, 2, 3):
        break

if typ == 1:
    print('''Введите выражение, с использованием матриц.
    В качестве обозначения можно использовать любую заглавную латинскую букву.
    Поддерживаются операции сложения, вычитания, умножения матриц, а также умножение матрицы на число''')
    print(input_expression())
else:
    rows = int(input(f'Введите количество строк матрицы (n): '))
    columns = int(input(f'Введите количество столбцов матрицы (m): '))
    matrix = []
    for r in range(rows):
        row = []
        for c in range(columns):
            element = input(f'Введите элемент {r + 1, c + 1}: ')
            row.append(int(element))
        matrix.append(row)
    if typ == 2:
        print(transposing(matrix))
    if typ == 3:
        print(compute_det(matrix))