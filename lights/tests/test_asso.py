# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.tests.testing_data import CreateTestingData
from lights.model.associations import AssociationFunctionFeatures


class Test(unittest.TestCase):
    """A class to test association functions
    """

    def setUp(self):
        data = CreateTestingData()
        alpha, L = data.fixed_effect_time_order, data.n_long_features
        self.asso_func = AssociationFunctionFeatures(data.asso_functions,
                                                     data.T_u, alpha, L)

    def test_asso_feat(self):
        """Tests the linear predictor association function
        """
        self.setUp()
        F_f, F_r = self.asso_func.get_asso_feat()
        # values of fixed_feature for first subject
        F_f_0 = np.array([[1, 2, 4, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 4, 0, 0, 0, 0, 0, 0],
                          [2, 2, 8/3, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 2, 4, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 4, 0, 0, 0],
                          [0, 0, 0, 2, 2, 8/3, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 2, 4],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 4],
                          [0, 0, 0, 0, 0, 0, 2, 2, 8/3]])

        F_r_1 = np.array([[1, 3, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [3, 9/2, 0, 0, 0, 0],
                          [0, 0, 1, 3, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 3, 9/2, 0, 0],
                          [0, 0, 0, 0, 1, 3],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 3, 9/2]])
        np.testing.assert_almost_equal(F_f[0], F_f_0)
        np.testing.assert_almost_equal(F_r[1], F_r_1)

if __name__ == "main":
    unittest.main()
