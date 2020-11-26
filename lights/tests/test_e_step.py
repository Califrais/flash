# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.tests.test_simu import get_train_data
from lights.tests.testing_data import CreateTestingData
from lights.model.e_step_functions import EstepFunctions


class Test(unittest.TestCase):
    """A class to test E_step functions
    """

    def setUp(self):
        data = CreateTestingData()
        X, Y, T, delta = data.X, data.Y, data.T, data.delta
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        theta = data.theta
        ext_feat = data.ext_feat
        asso_functions = data.asso_functions
        self.S = data.S
        self.E_func = EstepFunctions(X, T, delta, ext_feat, L, p, alpha,
                                asso_functions, theta)

    def test_g1(self):
        """Tests the g1 function
        """
        self.setUp()
        g1 = self.E_func.g1(self.S, broadcast=False)
        g1_0_1 = np.exp(np.array([130/3, 221/6, 143/6, 142/3]))
        g1_1_3 = np.exp(np.array([147, 172.5, 145.5, 61.5]))
        np.testing.assert_almost_equal(g1[0, 0, :, 0], g1_0_1)
        np.testing.assert_almost_equal(g1[0, 1, :, 2], g1_1_3)


    def test_g2(self):
        """Tests the g2 function
        """
        self.setUp()
        g2 = self.E_func.g2(self.S, broadcast=False)
        # values of g2 at first group and first sample
        g2_0_1 = np.array([121/3, 203/6, 125/6, 133/3])
        # values of g2 at second group and third sample
        g2_1_3 = np.array([153, 178.5, 151.5, 67.5])
        np.testing.assert_almost_equal(g2[0, 0], g2_0_1)
        np.testing.assert_almost_equal(g2[1, 2], g2_1_3)


if __name__ == "main":
    unittest.main()
