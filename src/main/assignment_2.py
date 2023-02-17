import math
import numpy as np
from decimal import *


def problem1(x: float, degree: int):
    x_table = [3.6, 3.8, 3.9]
    fx_table = [1.675, 1.436, 1.318]
    
    matrix = np.zeros((3, 3))

    for counter, row in enumerate(matrix):
        row[0] = fx_table[counter]

    num_of_points = 3

    for i in range(1, num_of_points):
        for j in range(1, num_of_points):
            first_mult = float((x - x_table[i-j]) * matrix[i][j-1])
            second_mult = float((x - x_table[i]) * matrix[i-1][j-1])

            denominator: float = float(x_table[i] - x_table[i-j])

            coefficient: float = float((first_mult - second_mult) / denominator)
            matrix[i][j] = coefficient


    ans: float = float(matrix[degree, degree])
    print(ans)
    return ans

def problem2():
    print()

def problem3():
    print()

def problem4():
    print()

def problem5():
    print()


if __name__ == "__main__":
    ans1 = problem1(3.7, 2)
