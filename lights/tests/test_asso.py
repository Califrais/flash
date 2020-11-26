# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.tests.testing_data import CreateTestingData
from lights.model.associations import AssociationFunctions


class Test(unittest.TestCase):
    """A class to test association functions
    """
    def setUp(self):
        data = CreateTestingData()
        X, Y, T, delta = data.X, data.Y, data.T, data.delta
        T_u = np.unique(T)
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        theta = data.theta
        beta = np.array([theta["beta_0"], theta["beta_1"]])
        self.asso_func = AssociationFunctions(T_u, data.S, beta, alpha, L)

    def test_lp_asso(self):
        """Tests the linear predictor association function
        """
        self.setUp()
        phi = self.asso_func.linear_predictor()
        phi_0_1 = np.array([[9,  8, 11,  5,],
                            [22, 22, 25, 18],
                            [41, 42, 45, 37]])
        phi_1_3 = np.array([[8, 8, 0, 7],
                            [10, 8, -4, 7],
                            [10, 6, -10, 5]])
        np.testing.assert_almost_equal(phi[0, :, :, 0], phi_0_1)
        np.testing.assert_almost_equal(phi[1, :, :, 2], phi_1_3)

    def test_re_asso(self):
        """Tests the random effects association function
        """
        self.setUp()
        phi = self.asso_func.random_effects()
        phi_0_0 = np.array([[1, 2, 3, 2, 3, 3],
                            [-1, 3, 2, 4, 5, 1],
                            [2, 3, -1, -2, 1, -3],
                            [-3, 2, 3, -3, 4, 1]])
        np.testing.assert_almost_equal(phi[0, 0], phi_0_0)

    def test_tps_asso(self):
        """Tests the time dependent slope association function
        """
        self.setUp()
        phi = self.asso_func.time_dependent_slope()
        phi_0_1 = np.array([[10, 11, 11, 10],
                            [16, 17, 17, 16],
                            [22, 23, 23, 22]])
        phi_1_3 = np.array([[3, 1, -3, 1],
                            [1, -1, -5, -1],
                            [-1, -3, -7, -3]])
        np.testing.assert_almost_equal(phi[0, :, :, 0], phi_0_1)
        np.testing.assert_almost_equal(phi[1, :, :, 2], phi_1_3)

    def test_ce_asso(self):
        """Tests the cumulative effects association function
        """
        self.setUp()
        phi = self.asso_func.cumulative_effects()
        phi_0_1 = np.array([[5, 3.5, 6.5, 1],
                            [20, 18, 24, 12],
                            [51, 49.5, 58.5, 39]])
        np.testing.assert_almost_equal(phi[0, :, :, 0], phi_0_1)

    def test_dlp_asso(self):
        """Tests the derivative linear predictor association function
        """
        self.setUp()
        phi = self.asso_func.derivative_linear_predictor()
        phi_0_1 = np.array([[1, 1, 1],
                            [1, 2, 4],
                            [1, 3, 9]])
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

    def test_dre_asso(self):
        """Tests the derivative random effects association function
        """
        self.setUp()
        phi = self.asso_func.derivative_random_effects()
        phi_0_1 = np.zeros((3, 6))
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

    def test_dtps_asso(self):
        """Tests the derivative time dependent slope association function
        """
        self.setUp()
        phi = self.asso_func.derivative_time_dependent_slope()
        phi_0_1 = np.array([[0, 1, 2],
                            [0, 1, 4],
                            [0, 1, 6]])
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

    def test_dce_asso(self):
        """Tests the derivative cumulative effects association function
        """
        self.setUp()
        phi = self.asso_func.derivative_cumulative_effects()
        phi_0_1 = np.array([[1, 0.5, 1/3],
                            [2, 2, 8/3],
                            [3, 4.5, 9]])
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)


if __name__ == "main":
    unittest.main()
