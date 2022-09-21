import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from src.calculator.transpose_matrix import transposing


input_matrix = [
    [1, 2, 3, 10],
    [4, 5, 6, 11],
    [7, 8, 9, 12]
]

expected_matrix = [
    [1, 4, 7],
    [2, 5, 8],
    [3, 6, 9],
    [10, 11, 12]
]


def test_func():
    assert transposing(mat=input_matrix) == expected_matrix, "Транспонирование работает неверно"


if __name__ == "__main__":
    test_func()