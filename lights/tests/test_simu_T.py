# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.inference import QNMCEM
from lights.tests.test_simu import get_train_data


class Test(unittest.TestCase):

    def test_QNMCEM(self):
        """Tests QNMCEM Algorithm
        """
        X, Y, T, delta = get_train_data(20)
        print(Y.shape)
        # qnmcem = QNMCEM(fixed_effect_time_order=1, max_iter=10, initialize=True,
        #                 print_every=1, asso_functions='all',
        #                 MC_sep=False, compute_obj=True)
        qnmcem = QNMCEM(fixed_effect_time_order=1, max_iter=10, compute_obj=True, print_every=1)
        qnmcem.fit(X, Y, T, delta)
        # C_index = qnmcem.score(X, Y, T, delta)
        # print(C_index)


if __name__ == "main":
    unittest.main()