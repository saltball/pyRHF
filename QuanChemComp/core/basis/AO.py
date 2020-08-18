# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : AO.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

class AO(object):

    def __init__(self, id, z, r, angum, l, coef, norm, expon):
        """

        :param id:
        :param z:
        :param r:
        :param angum:
        :param l:
        :param coef:
        :param norm:
        :param expon:
        """
        self.id = id
        self.z = z
        self.r = r
        self.angum = angum
        self.l = l
        self.norm = norm
        self.coef = coef
        self.expon = expon

    def get_lmn(self):
        if self.angum == 0:
            return (0, 0, 0)
        else:
            return self.l