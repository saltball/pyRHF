import unittest

from QuanChemComp.core.AnalyticInteg.kinetic import kinetDefold
from QuanChemComp.test.test_main import CALCU_PERCISION


class Test_Kinetic(unittest.TestCase):
    def test_Kinetic(self):
        la = 0
        ma = 0
        na = 0
        lb = 0
        mb = 0
        nb = 0
        a = 1
        b = 1

        import numpy as np

        a_array = np.array([0, 0, 0])
        b_array = np.array([0, 0, 0])

        self.assertAlmostEqual(kinetDefold(a, b, a_array, b_array, la, ma, na, lb, mb, nb),
                               1.5, CALCU_PERCISION)


if __name__ == '__main__':
    unittest.main()
