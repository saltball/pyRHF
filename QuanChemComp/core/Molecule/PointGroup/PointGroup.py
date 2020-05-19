# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : PointGroup.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #


class PointGroup(object):
    def __init__(self, atom_list, geom):
        self.name = "C1"

    @staticmethod
    def calculate_group(atom_list, geom):
        if True:  # TODO
            return "C1"


class PG_C1(PointGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "C1"
