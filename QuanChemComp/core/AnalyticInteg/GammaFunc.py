# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : GammaFunc.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numba


def _Fm(m, w):
    import sympy
    """
    Analytic Incomplete gamma function from `sympy` Package.
    :param m:
    :param w:
    :return:
    """
    if not w == 0:
        # print(m)
        return sympy.symbols("Fm({})".format(m))
    else:
        return 1 / (2 * m + 1)


@numba.njit
def FmDefold(m, w):
    """
    Pade approximate Boys function Used by Xv, Guangxian.
    徐光宪; 黎乐民; 王德民. 量子化学：基本原理和从头计算法, 第二版.; 科学出版社: 北京, 2007. C.2 p.67
    :param m:
    :param w:
    :return: $F_m(w)$
    """
    if m == 0:
        if w < 16.3578:
            return ((
                            1.0 +
                            0.21327130243142 * w ** 1 +
                            0.0629344460255614 * w ** 2 +
                            0.00769838037756759 * w ** 3 +
                            0.00075843319712716 * w ** 4 +
                            5.64691197633667e-05 * w ** 5) /
                    (
                            1 +
                            0.879937801660182 * w ** 1 +
                            0.338450368470103 * w ** 2 +
                            0.0738522953299624 * w ** 3 +
                            0.0101431553402629 * w ** 4 +
                            0.000955528842975585 * w ** 5 +
                            7.20266520392572e-05 * w ** 6)) ** (0.5)
        else:
            return (1.2533141373155001 / ((2 * w) ** (0.5)))
    if m == 1:
        if w < 17.4646:
            return ((
                            0.4807498567691362 +
                            0.0295195994716045 * w ** 1 +
                            0.0128790985465415 * w ** 2 +
                            0.000998165499553218 * w ** 3 +
                            9.70927983276419e-05 * w ** 4 +
                            4.93839847029699e-06 * w ** 5) /
                    (
                            1 +
                            0.461403194579124 * w ** 1 +
                            0.108494164372449 * w ** 2 +
                            0.0171462934845042 * w ** 3 +
                            0.00196918657845508 * w ** 4 +
                            0.000160138863265254 * w ** 5 +
                            8.57708713007233e-06 * w ** 6)) ** (1.5)
        else:
            return (1.2533141373155001 / ((2 * w) ** (1.5)))
    if m == 2:
        if w < 15.2368:
            return ((
                            0.5253055608807534 +
                            -0.00575763488635418 * w ** 1 +
                            0.00731474973333076 * w ** 2 +
                            0.000251276149443393 * w ** 3 +
                            2.64336244559094e-05 * w ** 4) /
                    (
                            1 +
                            0.274754154712841 * w ** 1 +
                            0.0425364830353043 * w ** 2 +
                            0.00493902790955943 * w ** 3 +
                            0.000437251500927601 * w ** 4 +
                            2.88914662393981e-05 * w ** 5)) ** (2.5)
        else:
            return (3.7599424119465006 / ((2 * w) ** (2.5)))
    if m == 3:
        if w < 16.0419:
            return ((
                            0.5735131987446477 +
                            -0.0290110430424666 * w ** 1 +
                            0.00561884370781462 * w ** 2 +
                            3.01628267382713e-05 * w ** 3 +
                            1.10671035361856e-05 * w ** 4) /
                    (
                            1 +
                            0.171637608242892 * w ** 1 +
                            0.0187571417256877 * w ** 2 +
                            0.00178536829675118 * w ** 3 +
                            0.000137360778130936 * w ** 4 +
                            7.91915206883054e-06 * w ** 5)) ** (3.5)
        else:
            return (18.799712059732503 / ((2 * w) ** (3.5)))
    if m == 4:
        if w < 16.8955:
            return ((
                            0.613685849032916 +
                            -0.0452693111179624 * w ** 1 +
                            0.00490070062899003 * w ** 2 +
                            -5.61789719979307e-05 * w ** 3 +
                            5.50814626951998e-06 * w ** 4) /
                    (
                            1 +
                            0.108051989937231 * w ** 1 +
                            0.00855924943430755 * w ** 2 +
                            0.000724968571389473 * w ** 3 +
                            5.02338223156067e-05 * w ** 4 +
                            2.49107837399141e-06 * w ** 5)) ** (4.5)
        else:
            return (131.5979844181275 / ((2 * w) ** (4.5)))
    if m == 5:
        if w < 17.7822:
            return ((
                            0.6466300385008377 +
                            -0.0566143259316101 * w ** 1 +
                            0.00455916894577203 * w ** 2 +
                            -8.94152721395639e-05 * w ** 3 +
                            3.28096732308082e-06 * w ** 4) /
                    (
                            1 +
                            0.0662932958471386 * w ** 1 +
                            0.00383724443872493 * w ** 2 +
                            0.000327167659811839 * w ** 3 +
                            2.10430437682548e-05 * w ** 4 +
                            8.83562935089333e-07 * w ** 5)) ** (5.5)
        else:
            return (1184.3818597631475 / ((2 * w) ** (5.5)))
    if m == 6:
        if w < 15.8077:
            return ((
                            0.6739444475794731 +
                            -0.0503249167534352 * w ** 1 +
                            0.00273135625430953 * w ** 2 +
                            -3.107336248191e-05 * w ** 3) /
                    (
                            1 +
                            0.0586609328033371 * w ** 1 +
                            0.00194044691497128 * w ** 2 +
                            0.000109442742502192 * w ** 3 +
                            6.13406236401726e-06 * w ** 4)) ** (6.5)
        else:
            return (13028.200457394623 / ((2 * w) ** (6.5)))
    if m == 7:
        if w < 16.5903:
            return ((
                            0.6969278698605402 +
                            -0.0548201062615785 * w ** 1 +
                            0.00253099908233175 * w ** 2 +
                            -3.33589469427863e-05 * w ** 3) /
                    (
                            1 +
                            0.0389873128779298 * w ** 1 +
                            0.000569890860832729 * w ** 2 +
                            4.22187129333708e-05 * w ** 3 +
                            2.86010059144633e-06 * w ** 4)) ** (7.5)
        else:
            return (169366.60594613012 / ((2 * w) ** (7.5)))
    if m == 8:
        if w < 17.3336:
            return ((
                            0.7165414254774158 +
                            -0.058161800607816 * w ** 1 +
                            0.00238525529084601 * w ** 2 +
                            -3.29989020317093e-05 * w ** 3) /
                    (
                            1 +
                            0.0240929282666615 * w ** 1 +
                            -0.000202677647499956 * w ** 2 +
                            1.1982067597446e-05 * w ** 3 +
                            1.45762086904409e-06 * w ** 4)) ** (8.5)
        else:
            return (2540499.089191952 / ((2 * w) ** (8.5)))
    if m == 9:
        if w < 15.6602:
            return ((
                            0.7334902710845949 +
                            -0.03348439939014 * w ** 1 +
                            0.000846637494147059 * w ** 2) /
                    (
                            1 +
                            0.0495875606944471 * w ** 1 +
                            0.000946642302340943 * w ** 2 +
                            1.0836777224979e-05 * w ** 3)) ** (9.5)
        else:
            return (43188484.51626318 / ((2 * w) ** (9.5)))
    if m == 10:
        if w < 16.5258:
            return ((
                            0.7482976009670683 +
                            -0.0335292171805959 * w ** 1 +
                            0.000749168957990503 * w ** 2) /
                    (
                            1 +
                            0.04214922932021683 * w ** 1 +
                            0.000582840985360327 * w ** 2 +
                            2.37676790577455e-06 * w ** 3)) ** (10.5)
        else:
            return (820581205.8090004 / ((2 * w) ** (10.5)))
    if m == 11:
        if w < 17.5395:
            return ((
                            0.7613579445345817 +
                            -0.0332669773790348 * w ** 1 +
                            0.000668720489602687 * w ** 2) /
                    (
                            1 +
                            0.0363057685289467 * w ** 1 +
                            0.000345646100984643 * w ** 2 +
                            -1.9087233037345e-06 * w ** 3)) ** (11.5)
        else:
            return (17232205321.989006 / ((2 * w) ** (11.5)))
    if m == 12:
        if w < 18.5783:
            return ((
                            0.7729738455277437 +
                            -0.0326241966410798 * w ** 1 +
                            0.000598705175467956 * w ** 2) /
                    (
                            1 +
                            0.0318680048277695 * w ** 1 +
                            0.000202419662347765 * w ** 2 +
                            -3.62095173837973e-06 * w ** 3)) ** (12.5)
        else:
            return (396340722405.7472 / ((2 * w) ** (12.5)))
    if m == 13:
        if w < 19.6511:
            return ((
                            0.7833810369372723 +
                            -0.0317754368014894 * w ** 1 +
                            0.000537678595933584 * w ** 2) /
                    (
                            1 +
                            0.0284036027081815 * w ** 1 +
                            0.000113673420662576 * w ** 2 +
                            -4.16076810552774e-06 * w ** 3)) ** (13.5)
        else:
            return (9908518060143.68 / ((2 * w) ** (13.5)))
    if m == 14:
        if w < 20.7839:
            return ((
                            0.7927659082294096 +
                            -0.0308755854748829 * w ** 1 +
                            0.000485046451960769 * w ** 2) /
                    (
                            1 +
                            0.0255694625434059 * w ** 1 +
                            5.4201019205508e-05 * w ** 2 +
                            -4.24759498527876e-06 * w ** 3)) ** (14.5)
        else:
            return (267529987623879.34 / ((2 * w) ** (14.5)))
    if m == 15:
        if w < 21.9998:
            return ((
                            0.8012778112031649 +
                            -0.03001438066719997 * w ** 1 +
                            0.000439983032427912 * w ** 2) /
                    (
                            1 +
                            0.0231478878674366 * w ** 1 +
                            1.05546581596674e-05 * w ** 2 +
                            -4.18932957034726e-06 * w ** 3)) ** (15.5)
        else:
            return (7758369641092501.0 / ((2 * w) ** (15.5)))
    if m == 16:
        if w < 20.9225:
            return ((
                            0.8090378723962302 +
                            -0.0288346417991609 * w ** 1 +
                            0.000397161796318408 * w ** 2) /
                    (
                            1 +
                            0.0215021933120724 * w ** 1 +
                            -1.2859245745395e-06 * w ** 2 +
                            -3.62120651688135e-06 * w ** 3)) ** (16.5)
        else:
            return (2.4050945887386752e+17 / ((2 * w) ** (16.5)))


def _FmPrint(m, w=0):
    """
    Print the approximate Incomplete gamma function Python Code Used by Xv, Guangxian.
    徐光宪; 黎乐民; 王德民. 量子化学：基本原理和从头计算法, 第二版.; 科学出版社: 北京, 2007. C.2 p.67
    :param m:
    :param w:
    :return:
    """
    a0 = (2 * m + 1) ** (-2 / (2 * m + 1))
    F = []
    # F0
    F.append(
        [[0.213271302431420, 0.629344460255614e-1, 0.769838037756759e-2, 0.758433197127160e-3, 0.564691197633667e-4],
         [0.879937801660182, 0.338450368470103, 0.738522953299624e-1, 0.101431553402629e-1, 0.955528842975585e-3,
          0.720266520392572e-4], 16.3578]
    )
    # F1
    F.append(
        [[0.295195994716045e-1, 0.128790985465415e-1, 0.998165499553218e-3, 0.970927983276419e-4, 0.493839847029699e-5],
         [0.461403194579124, 0.108494164372449, 0.171462934845042e-1, 0.196918657845508e-2, 0.160138863265254e-3,
          0.857708713007233e-5], 17.4646]
    )
    # F2
    F.append(
        [[-0.575763488635418e-2, 0.731474973333076e-2, 0.251276149443393e-3, 0.264336244559094e-4],
         [0.274754154712841, 0.425364830353043e-1, 0.493902790955943e-2, 0.437251500927601e-3, 0.288914662393981e-4],
         15.2368]
    )
    # F3
    F.append(
        [[-0.290110430424666e-1, 0.561884370781462e-2, 0.301628267382713e-4, 0.110671035361856e-4],
         [0.171637608242892, 0.187571417256877e-1, 0.178536829675118e-2, 0.137360778130936e-3, 0.791915206883054e-5],
         16.0419]
    )
    # F4
    F.append(
        [[-0.452693111179624e-1, 0.490070062899003e-2, -0.561789719979307e-4, 0.550814626951998e-5],
         [0.108051989937231, 0.855924943430755e-2, 0.724968571389473e-3, 0.502338223156067e-4, 0.249107837399141e-5],
         16.8955]
    )
    # F5
    F.append(
        [[-0.566143259316101e-1, 0.455916894577203e-2, -0.894152721395639e-4, 0.328096732308082e-5],
         [0.662932958471386e-1, 0.383724443872493e-2, 0.327167659811839e-3, 0.210430437682548e-4, 0.883562935089333e-6],
         17.7822]
    )
    # F6
    F.append(
        [[-0.503249167534352e-1, 0.273135625430953e-2, -0.310733624819100e-4],
         [0.586609328033371e-1, 0.194044691497128e-2, 0.109442742502192e-3, 0.613406236401726e-5], 15.8077]
    )
    # F7
    F.append(
        [[-0.548201062615785e-1, 0.253099908233175e-2, -0.333589469427863e-4],
         [0.389873128779298e-1, 0.569890860832729e-3, 0.422187129333708e-4, 0.286010059144633e-5], 16.5903]
    )
    # F8
    F.append(
        [[-0.581618006078160e-1, 0.238525529084601e-2, -0.329989020317093e-4],
         [0.240929282666615e-1, -0.202677647499956e-3, 0.119820675974460e-4, 0.145762086904409e-5], 17.3336]
    )
    # F9
    F.append(
        [[-0.334843993901400e-1, 0.846637494147059e-3],
         [0.495875606944471e-1, 0.946642302340943e-3, 0.108367772249790e-4], 15.6602]
    )
    # F10
    F.append(
        [[-0.335292171805959e-1, 0.749168957990503e-3],
         [0.4214922932021683e-1, 0.582840985360327e-3, 0.237676790577455e-5], 16.5258]
    )
    # F11
    F.append(
        [[-0.332669773790348e-1, 0.668720489602687e-3],
         [0.363057685289467e-1, 0.345646100984643e-3, -0.190872330373450e-5], 17.5395]
    )
    # F12
    F.append(
        [[-0.326241966410798e-1, 0.598705175467956e-3],
         [0.318680048277695e-1, 0.202419662347765e-3, -0.362095173837973e-5], 18.5783]
    )
    # F13
    F.append(
        [[-0.317754368014894e-1, 0.537678595933584e-3],
         [0.284036027081815e-1, 0.113673420662576e-3, -0.416076810552774e-5], 19.6511]
    )
    # F14
    F.append(
        [[-0.308755854748829e-1, 0.485046451960769e-3],
         [0.255694625434059e-1, 0.542010192055080e-4, -0.424759498527876e-5], 20.7839]
    )
    # F15
    F.append(
        [[-0.3001438066719997e-1, 0.439983032427912e-3],
         [0.231478878674366e-1, 0.105546581596674e-4, -0.418932957034726e-5], 21.9998]
    )
    # F16
    F.append(
        [[-0.288346417991609e-1, 0.397161796318408e-3],
         [0.215021933120724e-1, -0.128592457453950e-5, -0.362120651688135e-5], 20.9225]
    )
    if m in range(17):
        str1 = "if m=={}:\nif w<{}:\n return ((\n".format(m, F[m][2]) + str(a0)
        for index, item in enumerate(F[m][0]):
            str1 = str1 + "+\n{}*w**{}".format(item, index + 1)
        str1 = str1 + ")/\n(\n1"
        for index, item in enumerate(F[m][1]):
            str1 = str1 + "+\n{}*w**{}".format(item, index + 1)
        str1 = str1 + "))**({})".format(m + 0.5)
        print(str1)
        from QuanChemComp.core.math import DoubleFactorial
        import math
        str0 = "else:\n return ({}/((2*w)**({})))".format((DoubleFactorial(2 * m - 1)) * (math.pi / 2) ** 0.5, m + 0.5)
        print(str0)
    else:
        print("m={} Not Support By Print, Please use Func 'sympy.lowergamma'".format(m))


# def _Fmwfigure():
#     """
#     Draw figure of Incomplete gamma function and the scatters of Factorial values.
#     :return:
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import sympy
#
#     x = np.linspace(0, 20, 500)
#
#     y1 = ((
#                       1 + 0.213271302431420 * x + 0.0629344460255614 * x * x + 0.00769838037756759 * x * x * x + 0.000758433197127160 * x ** 4 + 0.0000564691197633667 * x ** 5) / (
#                       1 + 0.879937801660182 * x + 0.338450368470103 * x ** 2 + 0.0738522953299624 * x ** 3 + 0.0101431553402629 * x ** 4 + 0.000955528842975585 * x ** 5 + 0.0000720266520392572 * x ** 6)) ** (
#                      0 + 0.5)
#     y2 = []
#     for item in x:
#         y2.append(_Fm(0, item))
#     y3 = (1 / (2 * x) ** 0.5) * (sympy.pi / 2) ** 0.5
#     x = np.linspace(1, 16, 151)
#     y4 = []
#     y5 = []
#     for item in x:
#         print(item)
#         y4.append(sympy.factorial(item-1))
#     for item in x[::10]:
#         y5.append(sympy.gamma(item))
#
#     fig, ax = plt.subplots(1,2,)
#     ax[1].set_xlabel('X', fontsize=14, fontfamily='monospace', fontstyle='italic')
#     # ax[1].set_ylabel('Y', fontsize=14, fontfamily='monospace', fontstyle='italic')
#     ax[1].xaxis.set_tick_params(labelsize=14)
#     ax[1].yaxis.set_tick_params(labelsize=14)
#     ax[1].set_xlim(0, 11)
#     ax[1].set_ylim(0.1, 1e7)
#     start, end = ax[1].get_xlim()
#     ax[1].xaxis.set_ticks(np.arange(start, end+1, 1))
#     # ax.plot(x,y1)
#     # ax.plot(x,y2)
#     # ax.plot(x,y3)
#     ax[1].plot(x, y4, label='$Y=\Gamma(X)$')
#     ax[1].scatter(x[::10], y5,label='$Y=(X-1)!$',marker="+",s=100,c="C1")
#     # plt.plot(x, y2 - y1)
#     ax[1].set_yscale('log')
#
#     ax[0].set_xlabel('X', fontsize=14, fontfamily='monospace', fontstyle='italic')
#     ax[0].set_ylabel('Y', fontsize=14, fontfamily='monospace', fontstyle='italic')
#     ax[0].xaxis.set_tick_params(labelsize=14)
#     ax[0].yaxis.set_tick_params(labelsize=14)
#     ax[0].set_xlim(0, 7)
#     ax[0].set_ylim(0, 150)
#     start, end = ax[0].get_xlim()
#     ax[0].xaxis.set_ticks(np.arange(start, end + 1, 1))
#
#     ax[0].plot(x, y4,label='$Y=\Gamma(X)$')
#     ax[0].scatter(x[::10], y5,label='$Y=(X-1)!$',marker="+",s=100,c="C1")
#     ax[0].legend()
#     ax[1].legend()
#     plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25,
#                         wspace=0.35)
#     plt.show()

def _FmSpeed(m):
    # Incomplete gamma function Used by Xv, Guangxian, et al.
    import numpy as np
    x = np.linspace(0, 100, 10000)

    import time

    y2 = []

    time_start = time.time()
    for item in x:
        y2.append(_Fm(m, item))
    time_end = time.time()

    time2 = time_end - time_start
    print('Builtin method [time cost]', time_end - time_start, 's')
    y3 = []

    time_start = time.time()
    for item in x:
        y3.append(FmDefold(m, item))
    time_end = time.time()

    time3 = time_end - time_start
    print('FmDefold method [time cost]', time_end - time_start, 's')
    if time3 != 0:
        print(time2 / time3 + 0.000000000000000000000000000000000000000001, "times")
    import matplotlib.pyplot as plt
    yy = np.array(y2) - np.array(y3)
    try:
        print("MAX ERROR:{}".format(max(abs(yy))))
    except TypeError:
        pass
    plt.plot(yy)
    plt.show()