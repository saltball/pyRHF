# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : potential.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from QuanChemComp.core.AnalyticInteg.GammaFunc import FmDefold
from QuanChemComp.core.math import Comb, PI, SQRT_PI, PI5_2
from QuanChemComp.core.AnalyticInteg.gtoMath import norm_GTO, K_GTO
import numba
import numpy as np


@numba.njit
def _Eijt(i: int, j: int, t: int, p, PAx, PBx):
    """
    **Warning: return WITHOUT product $K_{AB}^x$**
    :param i:
    :param j:
    :param t:
    :param p:
    :param PAx:
    :param PBx:
    :return: Without product K_{AB}^x
    """
    # TODO: get from overlap integrate.
    if i < 0 or j < 0 or t < 0:
        return 0
    elif t > i + j:
        return 0
    elif i == 0 and j == 0 and t == 0:
        return 1
    elif (i == 0 or j == 0) and t > 0:
        if i == 0:
            return 1 / ((2 * p) ** t) * Comb(j, t) * _Eijt(0, j - t, 0, p, PAx, PBx)
        elif j == 0:
            return 1 / ((2 * p) ** t) * Comb(i, t) * _Eijt(i - t, 0, 0, p, PAx, PBx)
    elif t == 0 and j != 0:
        # print("Here!",i,j,t)
        return PBx * _Eijt(i, j - 1, t, p, PAx, PBx) + \
               1 / (2 * p) * (i * _Eijt(i - 1, j - 1, t, p, PAx, PBx)
                              + (j - 1) * _Eijt(i, j - 2, t, p, PAx, PBx))
    elif t == 0 and i != 0:
        return PAx * _Eijt(i - 1, j, t, p, PAx, PBx) + \
               1 / (2 * p) * ((i - 1) * _Eijt(i - 2, j, t, p, PAx, PBx)
                              + j * _Eijt(i - 1, j - 1, t, p, PAx, PBx))
    else:
        return 1 / (2 * p * t) * (i * _Eijt(i - 1, j, t - 1, p, PAx, PBx) + j * _Eijt(i, j - 1, t - 1, p, PAx, PBx))


@numba.njit
def _Rtuvn(t, u, v, n, p, PCx, PCy, PCz, PC2):
    """

    :param t:
    :param u:
    :param v:
    :param n:
    :param p:
    :param PCx:
    :param PCy:
    :param PCz:
    :param PC2:
    :return:
    """
    if t < 0 or u < 0 or v < 0:
        return 0
    if t == 0 and u == 0 and v == 0 and n == 0:
        return FmDefold(0, p * PC2)
    elif t == 0 and u == 0 and v == 0:
        return (-2 * p) ** n * FmDefold(n, p * PC2)
    elif t > 0:
        return (t - 1) * _Rtuvn(t - 2, u, v, n + 1, p, PCx, PCy, PCz, PC2) \
               + PCx * _Rtuvn(t - 1, u, v, n + 1, p, PCx, PCy, PCz, PC2)
    elif u > 0:
        return (u - 1) * _Rtuvn(t, u - 2, v, n + 1, p, PCx, PCy, PCz, PC2) \
               + PCy * _Rtuvn(t, u - 1, v, n + 1, p, PCx, PCy, PCz, PC2)
    elif v > 0:
        return (v - 1) * _Rtuvn(t, u, v - 2, n + 1, p, PCx, PCy, PCz, PC2) \
               + PCz * _Rtuvn(t, u, v - 1, n + 1, p, PCx, PCy, PCz, PC2)


@numba.njit
def V_tuvab(i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2):
    """

    :param i:
    :param j:
    :param k:
    :param l:
    :param m:
    :param n:
    :param p:
    :param Kab: exp(-ab/(a+b)*AB^2)
    :param rPA:
    :param rPB:
    :param rPC:
    :param PC2:
    :return:
    """
    # PC2=(rPC**2).sum()
    V_c = 2 * PI / p * Kab
    Vt = 0
    for t in numba.prange(i + j + 1):
        Vu = 0
        for u in numba.prange(k + l + 1):
            Vv = 0
            for v in numba.prange(m + n + 1):
                Vv = Vv + _Eijt(m, n, v, p, rPA[2], rPB[2]) * _Rtuvn(t, u, v, 0, p, rPC[0], rPC[1], rPC[2], PC2)
            Vu = Vu + _Eijt(k, l, u, p, rPA[1], rPB[1]) * Vv
        Vt = Vt + _Eijt(i, j, t, p, rPA[0], rPB[0]) * Vu
    return V_c * Vt


@numba.njit
def phi_2c1e_nNorm(t, u, v, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2):
    """
    Calculate the one electron integrate `$\\left<aAlmn\\mid \\frac{1}{r_{C}}mid bBlmn\\right>$`
    Warning: Not Normalized.
    :param t:
    :param u:
    :param v:
    :param i:
    :param j:
    :param k:
    :param l:
    :param m:
    :param n:
    :param p:
    :param Kab:
    :param rPA:
    :param rPB:
    :param rPC:
    :param PC2:
    :return:
    """
    if t < 0 or u < 0 or v < 0:
        return 0

    elif t == 0 and u == 0 and v == 0:
        return V_tuvab(i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)

    elif t > 0:
        # All do increase the angular momentum number of A.
        return 2 * p * (phi_2c1e_nNorm(t - 1, u, v, i + 1, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)
                        - rPA[0] * phi_2c1e_nNorm(t - 1, u, v, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)
                        - (t - 1) * phi_2c1e_nNorm(t - 2, u, v, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2))
    elif u > 0:
        # All do increase the angular momentum number of A.
        return 2 * p * (phi_2c1e_nNorm(t, u - 1, v, i, j, k + 1, l, m, n, p, Kab, rPA, rPB, rPC, PC2)
                        - rPA[1] * phi_2c1e_nNorm(t, u - 1, v, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)
                        - (u - 1) * phi_2c1e_nNorm(t, u - 2, v, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2))
    elif v > 0:
        # All do increase the angular momentum number of A.
        return 2 * p * (phi_2c1e_nNorm(t, u, v - 1, i, j, k, l, m + 1, n, p, Kab, rPA, rPB, rPC, PC2)
                        - rPA[2] * phi_2c1e_nNorm(t, u, v - 1, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2)
                        - (v - 1) * phi_2c1e_nNorm(t, u, v - 2, i, j, k, l, m, n, p, Kab, rPA, rPB, rPC, PC2))


@numba.njit
def phi_2c1e(i, j, k, l, m, n, a, b, rA, rB, rC, Norm=False):
    p = a + b
    Na = 1
    Nb = 1
    rP = (a * rA + b * rB) / p
    rPA = rP - rA
    rPB = rP - rB
    rPC = rP - rC
    AB2 = ((rA - rB) ** 2).sum()
    PC2 = (rPC ** 2).sum()
    Kab = K_GTO(a, b, AB2)
    if not Norm:
        Na = norm_GTO(a, i, j, k)
        Nb = norm_GTO(b, l, m, n)
    return Na * Nb * phi_2c1e_nNorm(0, 0, 0, i, l, j, m, k, n, p, Kab, rPA, rPB, rPC, PC2)


@numba.njit
def _g_cdtuv(it, iu, iv, lc, ld, mc, md, nc, nd, p, q, alp, rQC, rQD, rPQ, PQ2):
    """

    :param it:
    :param iu:
    :param iv:
    :param lc:
    :param ld:
    :param mc:
    :param md:
    :param nc:
    :param nd:
    :param p:
    :param q:
    :param alp:
    :param rQC:
    :param rQD:
    :param rPQ:
    :param PQ2:
    :return:
    """
    # p=a+b
    qt = lc + ld
    qu = mc + md
    qv = nc + nd
    # q=c+d
    Vt = 0
    for t in numba.prange(qt + 1):
        for u in numba.prange(qu + 1):
            for v in numba.prange(qv + 1):
                Vt = Vt + (-1) ** (t + u + v) * _Eijt(lc, ld, t, q, rQC[0], rQD[0]) \
                     * _Eijt(mc, md, u, q, rQC[1], rQD[1]) \
                     * _Eijt(nc, nd, v, q, rQC[2], rQD[2]) * _RRtuvtuv(it + t, iu + u, iv + v, alp, rPQ, PQ2)
    return Vt


@numba.njit
def _RRtuvtuv(tt, uu, vv, alp, rPQ: np.array, PQ2):
    """
    $2 * PI5_2 / (p * q * np.sqrt(p + q))$ is needed to multiple.
    :param tt:
    :param uu:
    :param vv:
    :param alp:
    :param rPQ:
    :param PQ2:
    :return:
    """
    return _Rtuvn(tt, uu, vv, 0, alp, rPQ[0], rPQ[1], rPQ[2], PQ2)


@numba.njit
def gabcd_nNorm(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, a, b, c, d, Kab_Kcd, rPA, rPB, rQC, rQD, rPQ, PQ2):
    """

    Warning: Not Normalized.
    :param la:
    :param lb:
    :param lc:
    :param ld:
    :param ma:
    :param mb:
    :param mc:
    :param md:
    :param na:
    :param nb:
    :param nc:
    :param nd:
    :param a:
    :param b:
    :param c:
    :param d:
    :param Kab_Kcd:
    :param rPA:
    :param rPB:
    :param rQC:
    :param rQD:
    :param rPQ:
    :param PQ2:
    :return:
    """
    pt = la + lb
    pu = ma + mb
    pv = na + nb
    p = a + b
    q = c + d
    alp = p * q / (p + q)
    Vt = 0
    for t in numba.prange(pt + 1):
        for u in numba.prange(pu + 1):
            for v in numba.prange(pv + 1):
                Vt = Vt + _Eijt(la, lb, t, p, rPA[0], rPB[0]) \
                     * _Eijt(ma, mb, u, p, rPA[1], rPB[1]) \
                     * _Eijt(na, nb, v, p, rPA[2], rPB[2]) \
                     * _g_cdtuv(t, u, v, lc, ld, mc, md, nc, nd, p, q, alp, rQC,
                                rQD,
                                rPQ, PQ2)
    return Kab_Kcd * 2 * PI5_2 / (p * q * np.sqrt(p + q)) * Vt


@numba.njit
def gabcd(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, a, b, c, d, rA, rB, rC, rD,
          Norm=False):
    Na = 1
    Nb = 1
    Nc = 1
    Nd = 1
    p = a + b
    q = c + d
    rP = (a * rA + b * rB) / p
    rQ = (c * rC + d * rD) / q
    rPA = rP - rA
    rPB = rP - rB
    rQC = rQ - rC
    rQD = rQ - rD
    rAB = rA - rB
    rCD = rC - rD
    rPQ = rP - rQ
    AB2 = (rAB ** 2).sum()
    CD2 = (rCD ** 2).sum()
    PQ2 = (rPQ ** 2).sum()
    Kab_Kcd = K_GTO(a, b, AB2) * K_GTO(c, d, CD2)

    if not Norm:
        Na = norm_GTO(a, la, ma, na)
        Nb = norm_GTO(b, lb, mb, nb)
        Nc = norm_GTO(c, lc, mc, nc)
        Nd = norm_GTO(d, ld, md, nd)
    return Na * Nb * Nc * Nd * gabcd_nNorm(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, a, b, c, d, Kab_Kcd, rPA,
                                           rPB, rQC, rQD, rPQ,
                                           PQ2)


if __name__ == "__main__":
    import os
    import time

    # E = _Eijt(0, 0, 0, 25, 4, -1.5)
    # R = _Rtuvn(0, 0, 0, 0, 20, 5, 5, 5, 75)
    # print(E, R)
    # listi = [0, 1, 2, 3, 4, 5, 6, 7]
    # listj = [0, 1, 2, 3, 4, 5, 6, 7]
    # listt = [0, 1]
    #
    # st = time.time()
    # for i in listi:
    #     for j in listj:
    #         for t in listt:
    #             E = _Eijt(i, j, t, 23, 4, -1.5)
    #             R = _Rtuvn(i, j, t, 0, 20, 5, 5, 5, 75)
    #         # print("i={},j={},E={}".format(i,j,E))
    # et = time.time()
    # print(et - st, "s")

    # print(_Eijt.inspect_types(),_Rtuvn.inspect_types())

    # st = time.time()
    # for i in listi:
    #     for j in listj:
    #         for t in listt:
    #             E = _Eijt.py_func(i, j, t, 23, 4, -1.5)
    #         # print("i={},j={},E={}".format(i,j,E))
    # et = time.time()
    # print(et - st, "s")
    # A = np.array([0, 0, 0])
    # a = 1
    # B = np.array([0, 0, 0])
    # b = 1
    # C = np.array([1, 0, 0])
    # c = 1
    # D = np.array([1, 0, 0])
    # d = 1
    # p = a + b
    # P = (a * A + b * b) / p
    # rPA = A - P
    # rPB = B - P
    # rPC = C - P
    # PC2 = (rPC ** 2).sum()
    # Kab = np.exp(-(a * b) / p * PC2)
    # phi_2c1e_nNorm(0, 0, 0, 0, 0, 0, 0, 0, 0, p, Kab, rPA, rPB, rPC, PC2)
    # gabcd(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c, d, A, B, C, D)
    # st = time.time()
    # for i in range(100):
    #     print(gabcd(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i+1, i+1, i+1, i+1, A, B, C, D))
    # et = time.time()
    # print(et - st, "s")
    # print(norm_GTO(4.9297,0,0,0))
    A = np.array([0.30926, 0.30926, 0.30926])
    B = np.array([0.30926, 0.30926, 0.30926])
    C = np.array([0.30926, 0.30926, 0.30926])
    D = np.array([0.30926, 0.30926, 0.30926])

    a = b = c = d = 4.9297

    st = time.time()
    print(gabcd(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c, d, A, B, C, D))
    et = time.time()
    print(et - st, "s")

    # st = time.time()
    # for i in range(100):
    #     print(phi_2c1e_nNorm(0, 0, 0, 0, 0, 0, 0, 0, 0, i+1, Kab, rPA, rPB, rPC, PC2))
    # et = time.time()
    # print(et - st, "s")
