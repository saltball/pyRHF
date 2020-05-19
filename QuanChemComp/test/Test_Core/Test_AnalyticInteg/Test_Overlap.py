import unittest

import numpy as np

from QuanChemComp.core.AnalyticInteg.overlap import *
from QuanChemComp.core.AnalyticInteg.overlap import _Sij, _IDefold
from QuanChemComp.test.test_main import CALCU_PERCISION

a_array = np.array([0, 0, 0])
b_array = np.array([0, 0, 0])


class TestOverLap(unittest.TestCase):

    def test_self_normalized(self):
        a = 1
        b = 1
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 0, 0, 0, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 0, 0, 1, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 1, 0, 1, 1, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 2, 0, 0, 2, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 3, 0, 0, 3, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 2, 1, 0, 2, 1, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 1, 1, 1, 1, 1), 1, CALCU_PERCISION)

    def test_self_orthogonal(self):
        a = 1
        b = 1
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 0, 0, 0, 1, 0), 0, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 0, 0, 0, 1, 0), 0, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 1, 0, 0, 0, 1), 0, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 0, 1, 0, 1, 0), 0, CALCU_PERCISION)

    def test_expanded(self):
        a = 4
        b = 2
        PA_x = 2
        PB_x = 3
        for la in range(8):
            for lb in range(8):
                self.assertAlmostEqual(_Sij(a, b, PA_x, PB_x, la, lb), _IDefold(a, b, PA_x, PB_x, la, lb), 9)


if __name__ == '__main__':
    unittest.main()
