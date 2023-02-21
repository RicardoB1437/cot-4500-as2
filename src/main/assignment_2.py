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

def prob4Helper(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            left: float = float(matrix[i][j-1])
            diagonal_left: float = float(matrix[i-1][j-1])
            numerator: float = float(left - diagonal_left)
            denominator: float = float(matrix[i][0] - matrix[i-j+1][0])

            operation: float = numerator / denominator
            matrix[i][j] = operation

    return matrix

def problem4():
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    x_table = [3.6, 3.8, 3.9]
    fx_table = [1.675, 1.436, 1.318]
    fx_prime_table = [-1.195, -1.188, -1.182]
    size: int = len(x_table)
    matrix: np.array = np.zeros((2*size,2*size))

    for x in range(0, 6):
        if x == 0 or x == 1:
            matrix[x][0] = x_table[0]
        if x == 2 or x == 3:
            matrix[x][0] = x_table[1]
        if x == 4 or x == 5:
            matrix[x][0] = x_table[2]

    for x in range(0, 6):
        if x == 0 or x == 1:
            matrix[x][1] = fx_table[0]
        if x == 2 or x == 3:
            matrix[x][1] = fx_table[1]
        if x == 4 or x == 5:
            matrix[x][1] = fx_table[2]

    for x in range(1, 6):
        if x == 1:
            matrix[x][2] = fx_prime_table[0]
        if x == 3:
            matrix[x][2] = fx_prime_table[1]
        if x == 5:
            matrix[x][2] = fx_prime_table[2]

    filled_matrix = prob4Helper(matrix)
    print()
    print(filled_matrix)


def problem5():
    # there will be n number of coords (0,1) (1,2) (2,4) (3,8)
    # there will be n-1 spines between them (x1-x0) (x2-x1) (x3-x2)
    # along the main diagonal its 1 TL 1 BR with 2( h(i-1) + h(i) ) and on the left of it is h(i-1) on the right is h(i)
    # everything else is 0
    x_table = [2, 5, 8, 10]
    fx_table = [3, 5, 7, 9]
    size = len(x_table)
    matrix = np.zeros((size, size))

    matrix[0][0] = 1
    matrix[size-1][size-1] = 1
    # loop from 1 to n-1
    for i in range(1, size-1):
        for j in range(0, 3):
            hLeft = x_table[i] - x_table[i-1]
            hRight = x_table[i+1] - x_table[i]
            temp = j + (i-1)
            if j == 0:
                matrix[i][temp] = hLeft
            if j == 1:
                matrix[i][temp] = 2 * (hLeft + hRight)
            if j == 2:
                matrix[i][temp] = hRight
    print()
    print(matrix)

    ##now find vector b
    #3/hi (ai+1 - ai) - 3/hi-1 (ai - ai-1)
    h = np.zeros(size-1)
    for i in range(0,size-1):
        h[i] = x_table[i+1] - x_table[i]

    b = np.zeros(size)
    for i in range(1,size-1):
        b[i] = (((3/h[i])*(fx_table[i+1] - fx_table[i])) - ((3/h[i-1])*(fx_table[i] - fx_table[i-1])))

    print()
    print(b)

    ##now find vector x
    #Ax = b
    A_np = np.array(matrix)
    b_np = np.array(b)
    x = np.linalg.solve(A_np, b_np)

    print()
    print(x)


if __name__ == "__main__":
    x_table = [7.2, 7.4, 7.5, 7.6]
    ans1 = problem1(3.7, 2)
    ans2Matrix = problem2()
    ans3 = problem3(ans2Matrix, 7.3, x_table)
    ans4 = problem4()
    ans5 = problem5()
