# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np

from lights.tests.testing_data import CreateTestingData
from lights.model.associations import AssociationFunctions

class Asso_Test():
    """A class to create testing association functions
    """
    def __init__(self):
        data = CreateTestingData()
        X, Y, T, delta = data.X, data.Y, data.T, data.delta
        T_u = np.unique(T)
        N = data.N
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        theta = data.theta
        beta = np.array([theta["beta_0"], theta["beta_1"]])
        self.asso_func = AssociationFunctions(T_u, data.S, beta, alpha, L)


class Test(unittest.TestCase):
    """
    A class to test association functions
    """
    def test_lp_asso(self):
        """Tests the the linear predictor association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.linear_predictor()
        phi_0_1 = np.array([[482, 492, 495, 478],
                         [1161, 1178, 1181, 1157],
                         [1542, 1562, 1565, 1538]])
        phi_1_3 = np.array([[-80, -102, -154, -103],
                            [-262, -298, -378, -299],
                            [-370, -412, -504, -413]])
        np.testing.assert_almost_equal(phi[0, :, :, 0], phi_0_1)
        np.testing.assert_almost_equal(phi[1, :, :, 2], phi_1_3)

    def test_re_asso(self):
        """Tests the the random effects association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.random_effects()
        phi_0_0 = np.array([[1, 2, 3, 2, 3, 3],
                      [-1, 3, 2, 4, 5, 1],
                      [2, 3, -1, -2, 1, -3],
                      [-3, 2, 3, -3, 4, 1]])
        np.testing.assert_almost_equal(phi[0, 0], phi_0_0)

    def test_tps_asso(self):
        """Tests the the time dependent slope association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.time_dependent_slope()
        phi_0_1 = np.array([[76, 77, 77, 76],
                            [118, 119, 119, 118],
                            [136, 137, 137, 136]])
        phi_1_3 = np.array([[-19, -21, -25, -21],
                            [-33, -35, -39, -35],
                            [-39, -41, -45, -41]])
        np.testing.assert_almost_equal(phi[0, :, :, 0], phi_0_1)
        np.testing.assert_almost_equal(phi[1, :, :, 2], phi_1_3)

    def test_ce_asso(self):
        """Tests the the cumulative effects association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.cumulative_effects()
        phi_0_1 = np.array([[2040, 2088, 2124, 1992],
                            [7619, 7761.5, 7818.5, 7543],
                            [11660, 11858, 11924, 11572]])
        np.testing.assert_almost_equal(phi[0, :, :, 0], phi_0_1)

    def test_dlp_asso(self):
        """Tests the the derivative linear predictor association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.derivative_linear_predictor()
        phi_0_1 = np.array([[1, 12, 144],
                           [1, 19, 361],
                           [1, 22, 484]])
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

    def test_dre_asso(self):
        """Tests the the derivative random effects association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.derivative_random_effects()
        phi_0_1 = np.zeros((3, 6))
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

    def test_dtps_asso(self):
        """Tests the the derivative time dependent slope association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.derivative_time_dependent_slope()
        phi_0_1 = np.array([[0, 1, 24],
                            [0, 1, 38],
                            [0, 1, 44]])
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

    def test_dce_asso(self):
        """Tests the the derivative cumulative effects association function
        """
        asso_func = Asso_Test().asso_func
        phi = asso_func.derivative_cumulative_effects()
        phi_0_1 = np.array([[12, 72, 576],
                            [19, 180.5, 6859/3],
                            [22, 242, 10648/3]])
        np.testing.assert_almost_equal(phi[0, 0, :, 0], phi_0_1)

if __name__ == "main":
    unittest.main()
