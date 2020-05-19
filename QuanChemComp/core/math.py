# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : math.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import numba

SQRT_PI = 1.77245385090551602729816748334  # sqrt(pi)
PI = 3.14159265358979323846264
PI5_2 = 17.49341832762486284626282  # pi^{5/2}


@numba.njit
def Factorial(n):
    """
    Factorial of integer. Also called $N!$
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 6
    elif n == 4:
        return 24
    elif n == 5:
        return 120
    elif n == 6:
        return 720
    elif n == 7:
        return 5040
    elif n == 8:
        return 40320
    elif n > 8:
        return n * Factorial(n - 1)


@numba.njit
def DoubleFactorial(n):
    """
    Double Factorial of integer. Also called $N!!$
    """

    if n == -1:
        return 1
    elif n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 3
    elif n == 4:
        return 8
    elif n == 5:
        return 15
    elif n == 6:
        return 48
    elif n == 7:
        return 105
    elif n == 8:
        return 384
    elif n == 9:
        return 945
    elif n == 10:
        return 3840
    elif n == 11:
        return 10395
    elif n == 12:
        return 46080
    elif n == 13:
        return 135135
    elif n == 14:
        return 645120
    elif n == 15:
        return 2027025
    elif n == 16:
        return 10321920
    elif n == 17:
        return 34459425
    elif n == 18:
        return 185794560
    elif n == 19:
        return 654729075
    elif n == 20:
        return 3715891200
    elif n == 21:
        return 13749310575
    elif n == 22:
        return 81749606400
    elif n == 23:
        return 316234143225
    elif n == 24:
        return 1961990553600
    elif n == 25:
        return 7905853580625
    elif n == 26:
        return 51011754393600
    elif n == 27:
        return 213458046676875
    elif n == 28:
        return 1428329123020800
    elif n == 29:
        return 6190283353629375
    elif n == 30:
        return 42849873690624000
    elif n == 31:
        return 191898783962510625
    elif n > 31:
        return n * DoubleFactorial(n - 2)


@numba.njit
def Comb(n: int, m: int):
    """
    Combination of m in n. 0<=m<=n.
    :param n:
    :param m:
    :return:
    """
    if not m <= n and m >= 0:
        pass
        # raise NotImplementedError("Combination of ({},{}) is not defined.".format(n, m))
    elif m <= n and m == 0:
        return 1
    elif m == n:
        return 1
    else:
        return Factorial(n) / (Factorial(m) * Factorial(n - m))


import math

import sys

# @jit(nopython=True)
# def incomp_gamma_function(m, w):
#     if w > 250:
#         print("WARNING: BOYS_FUNC maybe negetive!", sys.stderr)
#     eps = __EPS__OF__BOYSFUNCTION__ / np.exp(-w)
#     result = (DoubleFactorial(2 * m - 1)) / (DoubleFactorial(2 * m + 1))
#     delta = 1
#     i = 0
#     while delta > eps and i < __MAX__ITER__OF__BOYSFUNC__:
#         i = i + 1
#         delta = (DoubleFactorial(2 * m - 1) * (2 * w) ** i) / (DoubleFactorial(2 * m + 2 * i + 1))
#         result = result + delta
#     return result

if __name__ == "__main__":
    Factorial(int(35))
