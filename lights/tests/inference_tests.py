# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.inference import QNMCEM
from lights.tests.simu_tests import get_train_data


class Test(unittest.TestCase):

    def test_QNMCEM(self):
        """Tests QNMCEM Algorithm
        """
        X, Y, T, delta = get_train_data()
        qnmcem = QNMCEM(fixed_effect_time_order=1, max_iter=3, initialize=False)
        qnmcem.fit(X, Y, T, delta)


if __name__ == "main":
    unittest.main()
