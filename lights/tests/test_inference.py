# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.inference import prox_QNMCEM
from lights.tests.test_simu import get_train_data


class Test(unittest.TestCase):

    def test_prox_QNMCEM(self):
        """Tests prox_QNMCEM Algorithm
        """
        X, Y, T, delta, S_k = get_train_data(20)
        learner = prox_QNMCEM(fixed_effect_time_order=1, max_iter=10,
                        print_every=1, asso_functions='all', compute_obj=True)
        learner.fit(X, Y, T, delta, S_k)
        C_index = learner.score(X, Y, T, delta)
        print(C_index)


if __name__ == "main":
    unittest.main()
