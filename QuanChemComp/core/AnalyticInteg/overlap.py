# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : overlap.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import sympy
import numba

from QuanChemComp.core.math import Factorial, DoubleFactorial

# 所有符号
xa, ya, za, xb, yb, zb, rA, rB, a, b = sympy.symbols('x_a, y_a, z_a, x_b, y_b, z_b, r_A, r_B, a, b', real=True)
AB2 = (xb - xa) ** 2 + (yb - ya) ** 2 + (zb - za) ** 2
# xp = b / (a + b) * xb + a / (a + b) * xa
# yp = b / (a + b) * yb + a / (a + b) * ya
# zp = b / (a + b) * zb + a / (a + b) * za
xp, yp, zp = sympy.symbols("x_p,y_p,z_p")
ABx = xb - xa
# ABx = sympy.symbols('AB_x', real=True)
ABy = yb - ya
ABz = zb - za

PAx = xa - xp
# PAx = b/(a+b)*ABx
PAy = ya - yp
PAz = za - zp
PBx = xb - xp
# PBx = a/(a+b)*ABx
PBy = yb - yp
PBz = zb - zp

PAx, PAy, PAz, PBx, PBy, PBz = sympy.symbols('PA_x, PA_y, PA_z, PB_x, PB_y, PB_z', real=True)
K = sympy.exp(-(a * b) / (a + b) * AB2)


def _S(la: int, ma: int, na: int, lb: int, mb: int, nb: int):
    """
    Get the overlap integration of
    GTO A(la,ma,na) \\left| x_{A}^{l_a}y_{A}^{m_a}z_{A}^{n_a} a r_A \\right>
    and
    GTO B(la,ma,na) \\left| x_{B}^{l_b}y_{B}^{m_b}z_{B}^{n_b} a r_B \\right>.
    :param la:
    :param ma:
    :param na:
    :param lb:
    :param mb:
    :param nb:
    :return:
    """
    # Na = (2 * a / sympy.pi) ** (3 / 4) * ((4 * a) ** (la + ma + na) / (
    #         DoubleFactorial(2 * la - 1) * DoubleFactorial(2 * ma - 1) * DoubleFactorial(2 * na - 1))) ** (1 / 2)
    # Nb = (2 * b / sympy.pi) ** (3 / 4) * ((4 * b) ** (lb + mb + nb) / (
    #         DoubleFactorial(2 * lb - 1) * DoubleFactorial(2 * mb - 1) * DoubleFactorial(2 * nb - 1))) ** (1 / 2)
    Ix = 0
    for j in range(la + lb + 1):
        if j % 2 == 0:
            Ix = Ix + (_f(j, la, lb, PAx, PBx) * (a + b) ** -(j / 2 + 1 / 2) * sympy.gamma(
                j / 2 + 1 / 2) / 1.7724538509055159) * sympy.sqrt(sympy.pi)  # numerical to print comfortable.
    Iy = 0
    for j in range(ma + mb + 1):
        if j % 2 == 0:
            Iy = Iy + (_f(j, ma, mb, PAy, PBy) * (a + b) ** -(j / 2 + 1 / 2) * sympy.gamma(
                j / 2 + 1 / 2) / 1.7724538509055159) * sympy.sqrt(sympy.pi)
    Iz = 0
    for j in range(na + nb + 1):
        if j % 2 == 0:
            Iz = Iz + (_f(j, na, nb, PAz, PBz) * (a + b) ** -(j / 2 + 1 / 2) * sympy.gamma(
                j / 2 + 1 / 2) / 1.7724538509055159) * sympy.sqrt(sympy.pi)
    return Ix, Iy, Iz


def _Sxyz(la=0, ma=0, na=0, lb=0, mb=0, nb=0, norm=False):
    Ix, Iy, Iz = _S(la, ma, na, lb, mb, nb)
    return Ix * Iy * Iz


@numba.njit
def _f(i, la, lb, PA, PB):
    fI = 0
    if la < 0 or lb < 0:
        return fI
    else:
        for j in range(0, la + 1):
            for k in range(0, lb + 1):
                if j + k == i:
                    fI = fI + (Factorial(la) / (Factorial(j) * Factorial(la - j)) * (PA) ** (
                            la - j)) * (
                                 Factorial(lb) / (Factorial(k) * Factorial(lb - k)) * (PB) ** (
                                 lb - k))
        return fI


def _SxyzPrint(la=2, ma=0, na=0, lb=3, mb=0, nb=0):
    """
    Print the and N Sxyz function Python Code.
    :param la:
    :param ma:
    :param na:
    :param lb:
    :param mb:
    :param nb:
    :return:
    """
    Na = (2 * a / sympy.pi) ** (3 / 4) * ((4 * a) ** (la + ma + na) / (
            DoubleFactorial(2 * la - 1) * DoubleFactorial(2 * ma - 1) * DoubleFactorial(2 * na - 1))) ** (1 / 2)
    Nb = (2 * b / sympy.pi) ** (3 / 4) * ((4 * b) ** (lb + mb + nb) / (
            DoubleFactorial(2 * lb - 1) * DoubleFactorial(2 * mb - 1) * DoubleFactorial(2 * nb - 1))) ** (1 / 2)
    print("Na={}".format(Na))
    print("Nb={}".format(Nb))
    Ix, Iy, Iz = _S(la, ma, na, lb, mb, nb)
    print("I={}".format(Ix))


from QuanChemComp.core.math import SQRT_PI, PI
from math import sqrt

sqrt_pi = SQRT_PI


@numba.njit
def _IDefold(a, b, PA_x, PB_x, la, lb):
    """
    Calculate the Integration Value I_x in overlap S=KIxIyIz.
    :param a: GTO A
    :param b: GTO B
    :param PA_x: PA_x / PA_y / PA_z
    :param PB_x: PB_x / PB_y / PB_z
    :param la: power of x / y / z in GTO A
    :param lb: power of x / y / z in GTO A
    :return: Int Value
    """
    p = a + b
    if la < 0 or lb < 0:
        return 0
    # 0
    # 00
    elif la == 0 and lb == 0:
        return SQRT_PI / sqrt(p)
    # 1
    # 01
    elif (la == 0 and lb == 1):
        return PB_x * SQRT_PI / sqrt(p)
    # 10
    elif (la == 1 and lb == 0):
        return PA_x * SQRT_PI / sqrt(p)
    # 2
    # 02
    elif (la == 0 and lb == 2):
        return SQRT_PI * (PB_x * PB_x / sqrt(p) + 1 / sqrt(p) / p / 2)
    # 20
    elif (la == 2 and lb == 0):
        return SQRT_PI * (PA_x * PA_x / sqrt(p) + 1 / sqrt(p) / p / 2)
    # 11
    elif (la == 1 and lb == 1):
        return SQRT_PI * (PA_x * PB_x / sqrt(p) + 1 / sqrt(p) / p / 2)
    # 3
    # 12
    elif (la == 1 and lb == 2):
        return SQRT_PI * (PB_x / sqrt(p) / p + PA_x * (1 / sqrt(p) / p / 2 + PB_x * PB_x / sqrt(p)))
    # 21
    elif (la == 2 and lb == 1):
        return SQRT_PI * (PA_x / sqrt(p) / p + PB_x * (1 / sqrt(p) / p / 2 + PA_x * PA_x / sqrt(p)))
    # 03
    elif (la == 0 and lb == 3):
        return SQRT_PI / 2 * PB_x * (3 / sqrt(p) / p + 2 * PB_x * PB_x / sqrt(p))
    # 30
    elif (la == 3 and lb == 0):
        return SQRT_PI / 2 * PA_x * (3 / sqrt(p) / p + 2 * PA_x * PA_x / sqrt(p))
    # 4
    elif (la == 1 and lb == 3):
        return SQRT_PI * (0.75 / sqrt(p) / p / p + PB_x * PB_x * PB_x * PA_x / sqrt(p) + 1.5 / sqrt(p) / p * PB_x * (
                    PB_x + PA_x))
    elif (la == 3 and lb == 1):
        return SQRT_PI * (0.75 / sqrt(p) / p / p + PA_x * PA_x * PA_x * PB_x / sqrt(p) + 1.5 / sqrt(p) / p * PA_x * (
                    PA_x + PB_x))
    elif (la == 2 and lb == 2):
        return SQRT_PI * (0.75 / sqrt(p) / p / p + PA_x * PA_x * PB_x * PB_x / sqrt(p) + 0.5 / sqrt(p) / p * (
                    (PA_x + PB_x) * (PA_x + PB_x) + 2 * PA_x * PB_x))
    elif (la == 4 and lb == 0):
        return SQRT_PI * (0.75 / sqrt(p) / p / p + 3 * PA_x * PA_x / sqrt(p) / p + PA_x * PA_x * PA_x * PA_x / sqrt(p))
    elif (la == 0 and lb == 4):
        return SQRT_PI * (0.75 / sqrt(p) / p / p + 3 * PB_x * PB_x / sqrt(p) / p + PB_x * PB_x * PB_x * PB_x / sqrt(p))
    # 5
    elif (la == 5 and lb == 0):
        return SQRT_PI * PA_x * (
                3.75 / sqrt(p) / p / p + 5 * PA_x * PA_x / sqrt(p) / p + PA_x * PA_x * PA_x * PA_x / sqrt(p))
    elif (la == 0 and lb == 5):
        return SQRT_PI * PB_x * (
                3.75 / sqrt(p) / p / p + 5 * PB_x * PB_x / sqrt(p) / p + PB_x * PB_x * PB_x * PB_x / sqrt(p))
    elif (la == 4 and lb == 1):
        return SQRT_PI * (3 / sqrt(p) / p / p * PA_x + 2 / sqrt(p) / p * PA_x * PA_x * PA_x + 0.75 / sqrt(
            p) / p / p * PB_x + 3 / sqrt(p) / p * PA_x * PA_x * PB_x + PA_x * PA_x * PA_x * PA_x * PB_x / sqrt(p))
    elif (la == 1 and lb == 4):
        return SQRT_PI * (3 / sqrt(p) / p / p * PB_x + 2 / sqrt(p) / p * PB_x * PB_x * PB_x + 0.75 / sqrt(
            p) / p / p * PA_x + 3 / sqrt(p) / p * PB_x * PB_x * PA_x + PB_x * PB_x * PB_x * PB_x * PA_x / sqrt(p))
    elif (la == 2 and lb == 3):
        return SQRT_PI * (1.5 / sqrt(p) / p / p * PA_x + 3 / sqrt(p) / p * PB_x * PB_x * PA_x + PB_x * (
                    2.25 / sqrt(p) / p / p + 1.5 / sqrt(p) / p * PA_x * PA_x) + PB_x * PB_x * PB_x * (
                                      0.5 / sqrt(p) / p + PA_x * PA_x / sqrt(p)))
    elif (la == 3 and lb == 2):
        return SQRT_PI * (1.5 / sqrt(p) / p / p * PB_x + 3 / sqrt(p) / p * PA_x * PA_x * PB_x + PA_x * (
                    2.25 / sqrt(p) / p / p + 1.5 / sqrt(p) / p * PB_x * PB_x) + PA_x * PA_x * PA_x * (
                                      0.5 / sqrt(p) / p + PB_x * PB_x / sqrt(p)))

    # 6: 60, 06, 51, 15, 42, 24, 33
    elif (la == 6 and lb == 0):
        return SQRT_PI * (PA_x * PA_x * PA_x * PA_x * PA_x * PA_x / sqrt(p) + 7.5 * PA_x * PA_x * PA_x * PA_x / sqrt(
            p) / p + 11.25 * PA_x * PA_x / sqrt(p) / p / p + 1.875 / sqrt(p) / p / p / p)
    elif (la == 0 and lb == 6):
        return SQRT_PI * (PB_x * PB_x * PB_x * PB_x * PB_x * PB_x / sqrt(p) + 7.5 * PB_x * PB_x * PB_x * PB_x / sqrt(
            p) / p + 11.25 * PB_x * PB_x / sqrt(p) / p / p + 1.875 / sqrt(p) / p / p / p)
    elif (la == 5 and lb == 1):
        return SQRT_PI * (PA_x * PA_x * PA_x * PA_x * PA_x * PB_x / sqrt(p) + 2.5 / sqrt(p) / p * PA_x * PA_x * PA_x * (
                PA_x + 2 * PB_x) + 1.875 * (1 / sqrt(p) / p / p / p + 2 / sqrt(p) / p / p * PA_x * (2 * PA_x + PB_x)))
    elif (la == 1 and lb == 5):
        return SQRT_PI * (PB_x * PB_x * PB_x * PB_x * PB_x * PA_x / sqrt(p) + 2.5 / sqrt(p) / p * PB_x * PB_x * PB_x * (
                PB_x + 2 * PA_x) + 1.875 * (1 / sqrt(p) / p / p / p + 2 / sqrt(p) / p / p * PB_x * (2 * PB_x + PA_x)))

    elif (la == 4 and lb == 2):
        return SQRT_PI * (1.875 / sqrt(p) / p / p / p + 6 / sqrt(p) / p / p * PA_x * PB_x + 4 / sqrt(
            p) / p * PA_x * PA_x * PA_x * PB_x + 0.75 / sqrt(p) / p / p * PB_x * PB_x + PA_x * PA_x * (
                                      4.5 / sqrt(p) / p / p + 3 / sqrt(
                                  p) / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * (
                                      0.5 / sqrt(p) / p + PB_x * PB_x / sqrt(p)))

    elif (la == 2 and lb == 4):
        return SQRT_PI * (1.875 / sqrt(p) / p / p / p + 6 / sqrt(p) / p / p * PB_x * PA_x + 4 / sqrt(
            p) / p * PB_x * PB_x * PB_x * PA_x + 0.75 / sqrt(p) / p / p * PA_x * PA_x + PB_x * PB_x * (
                                  4.5 / sqrt(p) / p / p + 3 / sqrt(p) / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * (
                                  0.5 / sqrt(p) / p + PA_x * PA_x / sqrt(p)))

    elif (la == 3 and lb == 3):
        return SQRT_PI * (1.875 / sqrt(p) / p / p / p + PA_x * PA_x * PA_x * PB_x * PB_x * PB_x / sqrt(p) + (
                2.25 / sqrt(p) / p / p + 1.5 / sqrt(p) / p * PA_x * PB_x) * (
                                      PA_x * PA_x + 3 * PA_x * PB_x + PB_x * PB_x))

    # 7: 70, 07, 61, 16, 52, 25, 43, 34

    elif (la == 7 and lb == 0):
        return SQRT_PI * (
                PA_x * PA_x * PA_x * PA_x * PA_x * PA_x * PA_x / sqrt(
            p) + 10.5 * PA_x * PA_x * PA_x * PA_x * PA_x / sqrt(
            p) / p + 26.25 * PA_x * PA_x * PA_x / sqrt(p) / p / p + 13.125 * PA_x / sqrt(p) / p / p / p)
    elif (la == 0 and lb == 7):
        return SQRT_PI * (
                PB_x * PB_x * PB_x * PB_x * PB_x * PB_x * PB_x / sqrt(
            p) + 10.5 * PB_x * PB_x * PB_x * PB_x * PB_x / sqrt(
            p) / p + 26.25 * PB_x * PB_x * PB_x / sqrt(p) / p / p + 13.125 * PB_x / sqrt(p) / p / p / p)
    elif (la == 6 and lb == 1):
        return SQRT_PI * (
                PA_x * PA_x * PA_x * PA_x * PA_x * PA_x * PB_x / sqrt(p) + 3 * PA_x * PA_x * PA_x * PA_x * PA_x / sqrt(
            p) / p + 7.5 * PA_x * PA_x * PA_x * PA_x * PB_x / sqrt(p) / p + 15 * PA_x * PA_x * PA_x / sqrt(
            p) / p / p + 11.25 * PA_x * PA_x * PB_x / sqrt(p) / p / p + 11.25 * PA_x / sqrt(
            p) / p / p / p + 1.875 * PB_x / sqrt(p) / p / p / p)
    elif (la == 1 and lb == 6):
        return SQRT_PI * (
                PB_x * PB_x * PB_x * PB_x * PB_x * PB_x * PA_x / sqrt(p) + 3 * PB_x * PB_x * PB_x * PB_x * PB_x / sqrt(
            p) / p + 7.5 * PB_x * PB_x * PB_x * PB_x * PA_x / sqrt(p) / p + 15 * PB_x * PB_x * PB_x / sqrt(
            p) / p / p + 11.25 * PB_x * PB_x * PA_x / sqrt(p) / p / p + 11.25 * PB_x / sqrt(
            p) / p / p / p + 1.875 * PA_x / sqrt(p) / p / p / p)
    elif (la == 5 and lb == 2):
        return SQRT_PI * (3.75 / sqrt(p) / p / p / p * PB_x + 15 / sqrt(p) / p / p * PA_x * PA_x * PB_x + 5 / sqrt(
            p) / p * PA_x * PA_x * PA_x * PA_x * PB_x + PA_x * (9.375 / sqrt(p) / p / p / p + 3.75 / sqrt(
            p) / p / p * PB_x * PB_x) + PA_x * PA_x * PA_x * (7.5 / sqrt(p) / p / p + 5 / sqrt(
            p) / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * PA_x * (0.5 / sqrt(p) / p + PB_x * PB_x / sqrt(p)))
    elif (la == 2 and lb == 5):
        return SQRT_PI * (3.75 / sqrt(p) / p / p / p * PA_x + 15 / sqrt(p) / p / p * PB_x * PB_x * PA_x + 5 / sqrt(
            p) / p * PB_x * PB_x * PB_x * PB_x * PA_x + PB_x * (9.375 / sqrt(p) / p / p / p + 3.75 / sqrt(
            p) / p / p * PA_x * PA_x) + PB_x * PB_x * PB_x * (7.5 / sqrt(p) / p / p + 5 / sqrt(
            p) / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * PB_x * (0.5 / sqrt(p) / p + PA_x * PA_x / sqrt(p)))
    elif (la == 4 and lb == 3):
        return SQRT_PI * (5.625 / sqrt(p) / p / p / p * PB_x + 0.75 / sqrt(p) / p / p * PB_x * PB_x * PB_x + PA_x * (
                7.5 / sqrt(p) / p / p / p + 9 / sqrt(p) / p / p * PB_x * PB_x) + PA_x * PA_x * PA_x * (
                                  3 / sqrt(p) / p / p + 6 / sqrt(p) / p * PB_x * PB_x) + PA_x * PA_x * PB_x * (
                                  13.5 / sqrt(p) / p / p + 3 / sqrt(
                              p) / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * PB_x * (
                                  1.5 / sqrt(p) / p + PB_x * PB_x / sqrt(p)))
    elif (la == 3 and lb == 4):
        return SQRT_PI * (5.625 / sqrt(p) / p / p / p * PA_x + 0.75 / sqrt(p) / p / p * PA_x * PA_x * PA_x + PB_x * (
                7.5 / sqrt(p) / p / p / p + 9 / sqrt(p) / p / p * PA_x * PA_x) + PB_x * PB_x * PB_x * (
                                  3 / sqrt(p) / p / p + 6 / sqrt(p) / p * PA_x * PA_x) + PB_x * PB_x * PA_x * (
                                  13.5 / sqrt(p) / p / p + 3 / sqrt(
                              p) / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * PA_x * (
                                  1.5 / sqrt(p) / p + PA_x * PA_x / sqrt(p)))

    # 8: 53, 35, 44
    elif (la == 5 and lb == 3):
        return SQRT_PI * (6.5625 / sqrt(p) / p / p / p / p + 5.625 / sqrt(p) / p / p / p * PB_x * PB_x + PA_x * PB_x * (
                28.125 / sqrt(p) / p / p / p + 3.75 / sqrt(p) / p / p * PB_x * PB_x) + PA_x * PA_x * (
                                  18.75 / sqrt(p) / p / p / p + 22.5 / sqrt(
                              p) / p / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * (
                                  3.75 / sqrt(p) / p / p + 7.5 / sqrt(
                              p) / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PB_x * (
                                  22.5 / sqrt(p) / p / p + 5 / sqrt(
                              p) / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * PA_x * PB_x * (
                                  1.5 / sqrt(p) / p + PB_x * PB_x / sqrt(p)))

    elif (la == 3 and lb == 5):
        return SQRT_PI * (6.5625 / sqrt(p) / p / p / p / p + 5.625 / sqrt(p) / p / p / p * PA_x * PA_x + PB_x * PA_x * (
                28.125 / sqrt(p) / p / p / p + 3.75 / sqrt(p) / p / p * PA_x * PA_x) + PB_x * PB_x * (
                                  18.75 / sqrt(p) / p / p / p + 22.5 / sqrt(
                              p) / p / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * (
                                  3.75 / sqrt(p) / p / p + 7.5 / sqrt(
                              p) / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PA_x * (
                                  22.5 / sqrt(p) / p / p + 5 / sqrt(
                              p) / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * PB_x * PA_x * (
                                  1.5 / sqrt(p) / p + PA_x * PA_x / sqrt(p)))

    elif (la == 4 and lb == 4):
        return SQRT_PI * (6.5625 / sqrt(p) / p / p / p / p + 11.25 / sqrt(p) / p / p / p * PB_x * PB_x + 0.75 / sqrt(
            p) / p / p * PB_x * PB_x * PB_x * PB_x + PA_x * PB_x * (30 / sqrt(p) / p / p / p + 12 / sqrt(
            p) / p / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PB_x * (
                                  12 / sqrt(p) / p / p + 8 / sqrt(p) / p * PB_x * PB_x) + PA_x * PA_x * (
                                  11.25 / sqrt(p) / p / p / p + 27 / sqrt(p) / p / p * PB_x * PB_x + 3 / sqrt(
                              p) / p * PB_x * PB_x * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * (
                                  0.75 / sqrt(p) / p / p + 3 / sqrt(
                              p) / p * PB_x * PB_x + PB_x * PB_x * PB_x * PB_x / sqrt(
                              p)))

    # 9: 54, 45

    elif (la == 5 and lb == 4):
        return SQRT_PI * (
                PB_x * (
                    26.25 / sqrt(p) / p / p / p / p + 7.5 / sqrt(p) / p / p / p * PB_x * PB_x) + PA_x * PA_x * PB_x * (
                        75 / sqrt(p) / p / p / p + 30 / sqrt(
                    p) / p / p * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * PB_x * (
                        15 / sqrt(p) / p / p + 10 / sqrt(p) / p * PB_x * PB_x) + PA_x * (
                        32.8125 / sqrt(p) / p / p / p / p + 56.25 / sqrt(p) / p / p / p * PB_x * PB_x + 3.75 / sqrt(
                    p) / p / p * PB_x * PB_x * PB_x * PB_x) + PA_x * PA_x * PA_x * (
                        18.75 / sqrt(p) / p / p / p + 45 / sqrt(p) / p / p * PB_x * PB_x + 5 / sqrt(
                    p) / p * PB_x * PB_x * PB_x * PB_x) + PA_x * PA_x * PA_x * PA_x * PA_x * (
                        0.75 / sqrt(p) / p / p + 3 / sqrt(p) / p * PB_x * PB_x + PB_x * PB_x * PB_x * PB_x / sqrt(p)))

    elif (la == 4 and lb == 5):
        return SQRT_PI * (
                PA_x * (
                    26.25 / sqrt(p) / p / p / p / p + 7.5 / sqrt(p) / p / p / p * PA_x * PA_x) + PB_x * PB_x * PA_x * (
                        75 / sqrt(p) / p / p / p + 30 / sqrt(
                    p) / p / p * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * PA_x * (
                        15 / sqrt(p) / p / p + 10 / sqrt(p) / p * PA_x * PA_x) + PB_x * (
                        32.8125 / sqrt(p) / p / p / p / p + 56.25 / sqrt(p) / p / p / p * PA_x * PA_x + 3.75 / sqrt(
                    p) / p / p * PA_x * PA_x * PA_x * PA_x) + PB_x * PB_x * PB_x * (
                        18.75 / sqrt(p) / p / p / p + 45 / sqrt(p) / p / p * PA_x * PA_x + 5 / sqrt(
                    p) / p * PA_x * PA_x * PA_x * PA_x) + PB_x * PB_x * PB_x * PB_x * PB_x * (
                        0.75 / sqrt(p) / p / p + 3 / sqrt(p) / p * PA_x * PA_x + PA_x * PA_x * PA_x * PA_x / sqrt(p)))

    # 10: 55

    elif (la == 5 and lb == 5):
        return SQRT_PI * (PA_x * PA_x * PA_x * PA_x * PA_x * PB_x * PB_x * PB_x * PB_x * PB_x / sqrt(p) + (
                2.5 / sqrt(p) / p * PA_x * PA_x * PA_x * PB_x * PB_x * PB_x + 32.8125 / sqrt(p) / p / p / p / p) * (
                                  2 * PA_x + PB_x) * (PA_x + 2 * PB_x) + 29.53125 / sqrt(p) / p / p / p / p / p + (
                                  9.375 / sqrt(p) / p / p / p + 3.75 / sqrt(p) / p / p * PA_x * PB_x) * (
                                  PA_x * PA_x * PA_x * PA_x + 10 * PA_x * PA_x * PA_x * PB_x + 20 * PA_x * PA_x * PB_x * PB_x + 10 * PA_x * PB_x * PB_x * PB_x + PB_x * PB_x * PB_x * PB_x))

    elif (la > 0 and lb == 0):
        return PA_x * _IDefold(a, b, PA_x, PB_x, la - 1, 0) + 1.0 / (2.0 * (a + b)) * (
                    (la - 1) * _IDefold(a, b, PA_x, PB_x, la - 2, 0))
    elif (la == 0 and lb > 0):
        return PB_x * _IDefold(a, b, PA_x, PB_x, 0, lb - 1) + 1 / (2 * (a + b)) * (
                    (lb - 1) * _IDefold(a, b, PA_x, PB_x, 0, lb - 2))
    else:
        return PA_x * _IDefold(a, b, PA_x, PB_x, la - 1, lb) + 1 / (2 * (a + b)) * (
                    (la - 1) * _IDefold(a, b, PA_x, PB_x, la - 2, lb) + lb * _IDefold(a, b, PA_x, PB_x, la - 1, lb - 1))


from QuanChemComp.core.AnalyticInteg.gtoMath import norm_GTO, K_GTO


@numba.njit
def SxyzDefold(a, b, a_array, b_array, la=0, ma=0, na=0, lb=0, mb=0, nb=0, Norm=False):
    Na = 1
    Nb = 1
    if la < 0 or ma < 0 or na < 0 or lb < 0 or mb < 0 or nb < 0:
        return 0
    if not Norm:
        Na = norm_GTO(a, la, ma, na)
        Nb = norm_GTO(b, lb, mb, nb)
    dAB_2 = ((a_array - b_array) ** 2).sum()
    K = K_GTO(a, b, dAB_2)
    p_array = b / (a + b) * b_array + a / (a + b) * a_array
    PA_array = p_array - a_array
    PB_array = p_array - b_array
    Ix = _IDefold(a, b, PA_array[0], PB_array[0], la, lb)
    Iy = _IDefold(a, b, PA_array[1], PB_array[1], ma, mb)
    Iz = _IDefold(a, b, PA_array[2], PB_array[2], na, nb)
    return Na * Nb * K * Ix * Iy * Iz


@numba.njit
def SxyzRecur(a, b, a_array, b_array, la=0, ma=0, na=0, lb=0, mb=0, nb=0, Norm=False):
    Na = 1
    Nb = 1
    if la < 0 or ma < 0 or na < 0 or lb < 0 or mb < 0 or nb < 0:
        return 0
    if not Norm:
        Na = norm_GTO(a, la, ma, na)
        Nb = norm_GTO(b, lb, mb, nb)
    dAB_2 = ((a_array - b_array) ** 2).sum()
    K = K_GTO(a, b, dAB_2)
    p_array = b / (a + b) * b_array + a / (a + b) * a_array
    PA_array = p_array - a_array
    PB_array = p_array - b_array
    Ix = _Sij(a, b, PA_array[0], PB_array[0], la, lb)
    Iy = _Sij(a, b, PA_array[1], PB_array[1], ma, mb)
    Iz = _Sij(a, b, PA_array[2], PB_array[2], na, nb)
    return Na * Nb * K * Ix * Iy * Iz


@numba.njit
def _Sij(a, b, PA_x, PB_x, la, lb):
    if la < 0 or lb < 0:
        return 0
    elif la == 0 and lb == 0:
        return SQRT_PI / ((a + b) ** 0.5)
    elif la > 0 and lb == 0:
        return PA_x * _Sij(a, b, PA_x, PB_x, la - 1, lb) + 1 / (2 * (a + b)) * (
                (la - 1) * _Sij(a, b, PA_x, PB_x, la - 2, lb))
    elif la == 0 and lb > 0:
        return PB_x * _Sij(a, b, PA_x, PB_x, la, lb - 1) + 1 / (2 * (a + b)) * (
                (lb - 1) * _Sij(a, b, PA_x, PB_x, la, lb - 2))
    else:
        return PA_x * _Sij(a, b, PA_x, PB_x, la - 1, lb) + 1 / (2 * (a + b)) * (
                (la - 1) * _Sij(a, b, PA_x, PB_x, la - 2, lb) + lb * _Sij(a, b, PA_x, PB_x, la - 1, lb - 1))


if __name__ == "__main__":
    # _SxyzPrint(la=0, ma=0, na=0, lb=0, mb=0, nb=0)
    # sympy.pprint(Ix)
    # sympy.pprint(Iy)
    # sympy.pprint(Iz)
    # print(sympy.latex(sympy.simplify(Ix * Iy * Iz)))
    # a, b, PA_x, PB_x=sympy.symbols("a, b, PA_x, PB_x")
    # print(_IDefold(a,b,PA_x,PB_x,6,0))

    a = 1
    b = 1
    la = 0
    ma = 0
    na = 0
    lb = 0
    mb = 0
    nb = 0
    import numpy as np

    a_array = np.array([0, 0, 0])
    b_array = np.array([0, 0, 0])
    print(SxyzDefold(a, b, a_array, b_array, la, ma, na, lb, mb, nb))
