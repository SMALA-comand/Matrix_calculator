#Библиотека готовых матриц для тестирования

from transpose_matrix import*


def give_example():
    dict = {}
    dict[(1, 1)] = [[1]]
    dict[(1, 2)] = [[2, 4]]
    dict[(2, 2)] = [[1, 2], [4, 3]]
    dict[(2, 3)] = [[1, 4, 6], [2, 3, 5]]
    dict[(3, 3)] = [[3, 6, 9], [5, 4, 2], [1, 8, 7]]
    dict[(2, 4)] = [[1, 2, 5, 7], [3, 5, 1, 9]]
    dict[(3, 4)] = [[1, 2, 6, 8], [4, 1, 7, 5], [0, 2, 4, 1]]
    dict[(4, 4)] = [[1, 4, 6, 9], [2, 5, 3, 7], [4, 9, 1, 5], [1, 3, 8, 0]]
    dict[(3, 5)] = [[1, 1, 1, 1, 1], [5, 4, 3, 2, 1], [7, 8, 2, 1, 5]]
    dict[(4, 5)] = [[5, 5, 5, 5, 5], [8, 1, 5, 4, 1], [1, 2, 3, 4, 5], [0, 1, 0, 1, 4]]
    dict[(5, 5)] = [[1, 1, 2, 3, 5], [7, 0, 3, 6, 9], [3, 2, 5, 8, 5], [1, 1, 1, 1, 1], [2, 0, 7, 1, 8]]
    for i in dict.keys:
        if (m, n) not in dict.keys:
            dict[(m, n)] = transposing(dict[(n, m)])

    # print(dict)
    #print('Введите необходимое количество строк(n):')
    #n = int(input())
    #print('Введите необходимое количество строк(m):')
   # m = int(input())

    #return(dict[(n, m)])


