# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : kinetic.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #


from QuanChemComp.core.AnalyticInteg.gtoMath import norm_GTO, K_GTO
from QuanChemComp.core.AnalyticInteg.overlap import _Sij

from QuanChemComp.core.math import SQRT_PI, PI

import numba


@numba.njit
def kinetDefold(a, b, a_array, b_array, la, ma, na, lb, mb, nb, Norm=False):
    """
    Return Kinetic Integrate of GTO A |aAl_a m_a n_a> and B |bBl_b m_b n_b>
    :param a:
    :param b:
    :param a_array: numpy.array
    :param b_array: numpy.array
    :param la:
    :param ma:
    :param na:
    :param lb:
    :param mb:
    :param nb:
    :return:
    """
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
    Kl = _Tij(a, b, PA_array[0], PB_array[0], la, lb)*_Sij(a, b, PA_array[1], PB_array[1], ma, mb)*_Sij(a, b, PA_array[2], PB_array[2], na, nb)
    Km = _Sij(a, b, PA_array[0], PB_array[0], la, lb)*_Tij(a, b, PA_array[1], PB_array[1], ma, mb)*_Sij(a, b, PA_array[2], PB_array[2], na, nb)
    Kn = _Sij(a, b, PA_array[0], PB_array[0], la, lb)*_Sij(a, b, PA_array[1], PB_array[1], ma, mb)*_Tij(a, b, PA_array[2], PB_array[2], na, nb)
    return Na*Nb*K*(Kl + Km + Kn)

@numba.njit
def _Tij(a, b, PA_x, PB_x, la, lb):
    if la<0 or lb<0:
        return 0
    elif la==0 and lb==0:
        return (a-2*a**2*(PA_x**2+1/(2*(a+b))))*SQRT_PI/(a+b)**0.5
    elif la==0 and lb >0:
        return PB_x*_Tij(a, b, PA_x, PB_x, la, lb-1)+1/(2*(a + b))*((lb-1)*_Tij(a, b, PA_x, PB_x, la, lb-2))\
               +a/(a+b)*(2*b*_Sij(a, b, PA_x, PB_x, la, lb)-(lb-1)*_Sij(a, b, PA_x, PB_x, la, lb-2))
    elif la>0 and lb==0:
        return PA_x*_Tij(a, b, PA_x, PB_x, la-1, lb)+1/(2*(a + b))*((la-1)*_Tij(a, b, PA_x, PB_x, la-2, lb))\
               +b/(a+b)*(2*a*_Sij(a, b, PA_x, PB_x, la, lb)-(la-1)*_Sij(a, b, PA_x, PB_x, la-2, lb))
    else:
        return PA_x*_Tij(a, b, PA_x, PB_x, la-1, lb)+1/(2*(a + b))*((la-1)*_Tij(a, b, PA_x, PB_x, la-2, lb)+lb*_Tij(a, b, PA_x, PB_x, la-1, lb-1))\
               +b/(a+b)*(2*a*_Sij(a, b, PA_x, PB_x, la, lb)-(la-1)*_Sij(a, b, PA_x, PB_x, la-2, lb))

if __name__ == "__main__":
    lalist = range(0, 4)
    malist = range(0, 4)
    nalist = range(0, 4)
    lblist = range(0, 4)
    mblist = range(0, 4)
    nblist = range(0, 4)
    a = 1
    b = 1

    import numpy as np

    a_array = np.array([0, 0, 0])
    b_array = np.array([0, 0, 0])
    import time

    k = kinetDefold(a, b, a_array, b_array, 0, 0, 0, 0, 0, 0)

    print(k)

    st = time.time()
    for la in lalist:
        for lb in lblist:
            for ma in malist:
                for mb in mblist:
                    for na in nalist:
                        for nb in nblist:
                            k = kinetDefold(a, b, a_array, b_array, la, ma, na, lb, mb, nb)
    et = time.time()

    print(k, "\n", et - st, "s")
