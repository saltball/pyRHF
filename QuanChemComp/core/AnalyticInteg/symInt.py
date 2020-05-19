# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : symInt.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import sympy

SQRT_PI = sympy.symbols("SQRT_PI")


def _Sij(a, b, PA_x, PB_x, la, lb):
    p=sympy.symbols("p")
    if la < 0 or lb < 0:
        return 0
    elif la == 0 and lb == 0:
        return SQRT_PI / ((p) ** 0.5)
    elif la > 0 and lb == 0:
        return PA_x * _Sij(a, b, PA_x, PB_x, la - 1, lb) + 1 / (2 * (p)) * (
                    (la - 1) * _Sij(a, b, PA_x, PB_x, la - 2, lb))
    elif la == 0 and lb > 0:
        return PB_x * _Sij(a, b, PA_x, PB_x, la, lb - 1) + 1 / (2 * (p)) * (
                    (lb - 1) * _Sij(a, b, PA_x, PB_x, la, lb - 2))
    else:
        return PA_x * _Sij(a, b, PA_x, PB_x, la - 1, lb) + 1 / (2 * (p)) * (
                    (la - 1) * _Sij(a, b, PA_x, PB_x, la - 2, lb) + lb * _Sij(a, b, PA_x, PB_x, la - 1, lb - 1))

# def _Sij(a, b, PA_x, PB_x, la, lb):
#     if la < 0 or lb < 0:
#         return 0
#     else:
#         return sympy.symbols("Sij{}{}".format(la,lb))

def _Tij(a, b, PA_x, PB_x, la, lb):
    p=sympy.symbols("p")
    if la<0 or lb<0:
        return 0
    elif la==0 and lb==0:
        return (a-2*a**2*(PA_x**2+1/(2*(p))))*SQRT_PI/p**0.5
    elif la==0 and lb >0:
        return PB_x*_Tij(a, b, PA_x, PB_x, la, lb-1)+1/(2*(p))*((lb-1)*_Tij(a, b, PA_x, PB_x, la, lb-2))\
               +a/(p)*(2*b*_Sij(a, b, PA_x, PB_x, la, lb)-(lb-1)*_Sij(a, b, PA_x, PB_x, la, lb-2))
    elif la>0 and lb==0:
        return PA_x*_Tij(a, b, PA_x, PB_x, la-1, lb)+1/(2*(p))*((la-1)*_Tij(a, b, PA_x, PB_x, la-2, lb))\
               +b/p*(2*a*_Sij(a, b, PA_x, PB_x, la, lb)-(la-1)*_Sij(a, b, PA_x, PB_x, la-2, lb))
    else:
        return PA_x*_Tij(a, b, PA_x, PB_x, la-1, lb)+1/(2*(p))*((la-1)*_Tij(a, b, PA_x, PB_x, la-2, lb)+lb*_Tij(a, b, PA_x, PB_x, la-1, lb-1))\
               +b/(p)*(2*a*_Sij(a, b, PA_x, PB_x, la, lb)-(la-1)*_Sij(a, b, PA_x, PB_x, la-2, lb))



if __name__ == "__main__":
    a, b, PA_x, PB_x = sympy.symbols("a,b,PA_x,PB_x")
    la = 7
    lb = 0
    print(sympy.sympify( sympy.expand(_Sij(a, b, PA_x, PB_x, la, lb))))
