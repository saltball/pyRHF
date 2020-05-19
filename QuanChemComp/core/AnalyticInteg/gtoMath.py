# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : gtoMath.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import math
import numpy as np
import numba
import sympy

from QuanChemComp.core.math import DoubleFactorial, SQRT_PI, PI

sqrt_pi = SQRT_PI  # sqrt(pi)
pi = PI


def _symbol_norm_GTO(la: int, ma: int, na: int, lb: int, mb: int, nb: int):
    a, b = sympy.symbols("a,b")
    Na = (2 * a / sympy.pi) ** (3 / 4) * ((4 * a) ** (la + ma + na) / (
            DoubleFactorial(2 * la - 1) * DoubleFactorial(2 * ma - 1) * DoubleFactorial(2 * na - 1))) ** (1 / 2)
    Nb = (2 * b / sympy.pi) ** (3 / 4) * ((4 * b) ** (lb + mb + nb) / (
            DoubleFactorial(2 * lb - 1) * DoubleFactorial(2 * mb - 1) * DoubleFactorial(2 * nb - 1))) ** (1 / 2)
    return (Na, Nb)

def _new_symbol_norm_GTO(a, la=0, ma=0, na=0):
    # return (2 * a / sympy.pi) ** (3 / 4) * ((4 * a) ** (la + ma + na) / (
    #         sympy.factorial2(2 * la - 1) * sympy.factorial2(2 * ma - 1) * sympy.factorial2(2 * na - 1))) ** (1 / 2)
    return sympy.symbols("N_Norm")

@numba.njit
def norm_GTO(a: float, la=0, ma=0, na=0):
    """
    Return the Normalized Parameter of GTO $\\left|aAlmn\\right>$
    :param a: Exponential of GTO
    :param la: Exponential of x
    :param ma: Exponential of y
    :param na: Exponential of z
    :return: Normalized Parameter $N=(2 * a / \\pi) ^ (3 / 4) * ((4 * a) ^ (la + ma + na) / ((2 * la - 1)!! * (2 * ma - 1)!! * (2 * na - 1)!!)) ^ (1 / 2)$
    """
    # 000
    if la == 0 and ma == 0 and na == 0:
        return 1.681792830507429 * pi ** (-0.75) * a ** 0.75
    # 100
    elif (la == 1 and ma == 0 and na == 0) or (la == 0 and ma == 1 and na == 0) or (la == 0 and ma == 0 and na == 1):
        return 3.36358566101486 * pi ** (-0.75) * a ** 1.25
    # 200
    elif (la == 2 and ma == 0 and na == 0) or (la == 0 and ma == 2 and na == 0) or (la == 0 and ma == 0 and na == 2):
        return 3.88393417365859 * pi ** (-0.75) * a ** 1.75
    elif (la == 1 and ma == 1 and na == 0) or (la == 1 and ma == 1 and na == 0) or (la == 0 and ma == 1 and na == 1):
        return 6.72717132202972 * pi ** (-0.75) * a ** 1.75
    # 300
    elif (la == 3 and ma == 0 and na == 0) or (la == 0 and ma == 3 and na == 0) or (la == 0 and ma == 0 and na == 3):
        return 3.47389633297403 * pi ** (-0.75) * a ** 0.75 * (a ** 3) ** 0.5
    elif (la == 2 and ma == 1 and na == 0) or (la == 2 and ma == 0 and na == 1) or (
            la == 0 and ma == 2 and na == 1) or (la == 1 and ma == 2 and na == 0) or (
            la == 1 and ma == 0 and na == 2) or (la == 0 and ma == 1 and na == 2):
        return 7.76786834731717 * pi ** (-0.75) * a ** 0.75 * (a ** 3) ** 0.5
    elif (la == 1 and ma == 1 and na == 1) or (la == 1 and ma == 1 and na == 1) or (la == 1 and ma == 1 and na == 1):
        return 13.4543426440594 * pi ** (-0.75) * a ** 0.75 * (a ** 3) ** 0.5

    # TODO(yanbohan98@gmail.com):To be Extended.

    else:
        # assert type(la) == int and type(ma) == int and type(na) == int\
            # , \
            # TypeError\
                #      (
                # "Got Wrong number in norm_GTO, here got a [{}] type l,a [{}] type m,a [{}] type n".format(type(la),
                #                                                                                           type(ma),
                #                                                                                           type(na)))
        return (2 * a / math.pi) ** (3 / 4) * ((4 * a) ** (la + ma + na) / (
                DoubleFactorial(2 * la - 1) * DoubleFactorial(2 * ma - 1) * DoubleFactorial(2 * na - 1))) ** (1 / 2)


@numba.njit
def K_GTO(a, b, dAB_2):
    """
    Return the often Used K value in GTO calculation.
    :param a:
    :param b:
    :param dAB:
    :return: K $K=\\exp{-ab/(a+b)|AB|^2}$
    """
    return np.exp(-a * b / (a + b) * dAB_2)


if __name__ == "__main__":
    print(_symbol_norm_GTO(2, 0, 0, 0, 0, 0))
