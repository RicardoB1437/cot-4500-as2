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
    print()
    return ans

def problem2():
    x_table = [7.2, 7.4, 7.5, 7.6]
    fx_table = [23.5492, 25.3913, 26.8224, 27.4589]
    size: int = 4
    matrix: np.array = np.zeros((4,4))

    for index, row in enumerate(matrix):
        row[0] = float(fx_table[index])

    for i in range(1, 4):
        for j in range(1, 4):
            numerator: float = float(matrix[i][j-1] - matrix[i-1][j-1])

            denominator: float = float(x_table[i] - x_table[i-j])

            operation: float = float(numerator / denominator)

            #matrix[i][j] = '{0:.7g}'.format(operation)
            matrix[i][j] = float(operation)
        print(float(matrix[i][i]))

    print()
    return matrix

def problem3(matrix, x: float, x_table):
    reoccuring_x_span: float = 1
    reoccuring_px_result: float = matrix[0][0]

    # pn(x) = p(n-1) + matrix[n][n](x-)

    for index in range(1, 4):
        polynomial_coefficient: float = matrix[index][index]

        i: int = index
        #reoccuring_x_span *= float(x - x_table[index])
        x_span_mult: float = float(x - x_table[i-1])
        i = i-1

        while i>0:
            x_span_mult *= float(x-x_table[i-1])
            i = i-1

        mult_operation: float = float(polynomial_coefficient * x_span_mult)
        reoccuring_px_result = reoccuring_px_result + mult_operation
        

    ans: float = reoccuring_px_result
    print(ans)
    return ans

def problem4():
    print()

def problem5():
    print()


if __name__ == "__main__":
    x_table = [7.2, 7.4, 7.5, 7.6]
    ans1 = problem1(3.7, 2)
    ans2Matrix = problem2()
    ans3 = problem3(ans2Matrix, 7.3, x_table)
