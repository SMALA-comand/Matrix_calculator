import random as ra
import csv
def matrix_generator(n = 0, m = 0):
    if n == m == 0:
        print('Введите количество строк (n):')
        n = int(input())
        print('Введите количество строк (m):')
        m = int(input())

    matrix = []
    for i in range (n):
        row = []
        for j in range (m):
            row.append(ra.random() * (10 ** ra.randint(1, 6)))
        matrix.append(row)
    with open("rand_matrix.csv", mode = 'w', encoding = 'utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter = ',', lineterminator = '\r')
        file_writer.writerows(matrix)
    return(matrix)

#<<<<<<< HEAD
#print(matrix_generator(3, 3))
#=======
#print(matrix_generator(3, 3))
#>>>>>>> origin/Artem_branch
