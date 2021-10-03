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

