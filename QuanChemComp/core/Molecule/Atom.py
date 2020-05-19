# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : Atom.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from QuanChemComp.lib.elements import ELEMENTS

class Atom(object):
    def __init__(self,number = 0, symbol = "X", isotopemassnum = -1, **kwargs):
        if number != 0 and symbol == "X":
            self.number = number
            self.symbol = ELEMENTS[number].symbol
        elif number == 0 and symbol != "X":
            try:
                self.number = ELEMENTS[symbol].number
                self.symbol = symbol
            except KeyError:
                raise KeyError("Undefined Element Symbol {}".format(symbol))
        self.electrons = number
        self.protons = number
        self.massnum = isotopemassnum
        if self.massnum != -1 and number > 0:
            pass
        else:
            self.massnum = ELEMENTS[number].nominalmass
        try:
            self.exactmass = ELEMENTS[number].isotopes[isotopemassnum].mass
        except KeyError:
            raise KeyError("Isotope massnum {} of Element {} is Undefined.".format(isotopemassnum,symbol))
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return "{}(AtomNumber{} AtomMass:{})".format(self.symbol,self.number,self.massnum)
