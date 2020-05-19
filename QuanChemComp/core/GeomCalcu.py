# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : GeomCalcu.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
import numpy as np


def tensor_angle(t1, t2):
    """
    Calculate the angle in degree from t1 to t2.
    :param t1:
    :param t2:
    :return:
    """
    anglecos = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
    if anglecos > 1:
        return 180.0
    elif anglecos < -1:
        return 0.0
    else:
        return np.arccos(anglecos)/np.pi * 180.0


def martrix_dis_calcu(tensor, dim=3):
    """
    Calculate the distance between in tensor.

    >>> a = np.array([1,2,3])
    >>> b = np.array([-1,2,-3])
    >>> c = np.array([-1,-2,-3])
    >>> d = np.array([1,-2,3])
    >>> martrix_dis_calcu(np.array([a,b,c,d]))
    array([[0.        , 6.32455532, 7.48331477, 4.        ],
           [6.32455532, 0.        , 4.        , 7.48331477],
           [7.48331477, 4.        , 0.        , 6.32455532],
           [4.        , 7.48331477, 6.32455532, 0.        ]])

    :param tensor1:
    :param tensor2:
    :return:

    """
    num = int(tensor.size / dim)
    tensor = tensor.reshape([num, 1, dim])
    result = 0
    for i in range(dim):
        G = np.dot(tensor[:, :, i], tensor[:, :, i].transpose())
        H = np.tile(np.diag(G), (num, 1))
        result = result + (H + H.transpose() - 2 * G)
    return np.sqrt(result)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    import time
    a = np.array([1, 2, 3])
    b = np.array([-1, 2, -3])
    c = np.array([-1, -2, -3])
    d = np.array([1, -2, 3])
    st=time.time()
    martrix_dis_calcu(np.array([a, b, c, d,a+1,b+1,c+1,d+1,a+1,b+1,c+1,d+1,a+1,b+1,c,d+1,a+1,b+1,c+1,d+1,a+1,b+1,c+1,d+2]))
    et=time.time()
    print("time cost:{}s".format(et-st))
