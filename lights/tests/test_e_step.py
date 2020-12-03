# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.tests.testing_data import CreateTestingData
from lights.model.e_step_functions import EstepFunctions


class Test(unittest.TestCase):
    """A class to test E_step functions
    """

    def setUp(self):
        data = CreateTestingData()
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        theta, asso_functions = data.theta, data.asso_functions
        self.n_samples = data.n_samples
        self.S, self.n_MC = data.S, data.S.shape[0]
        self.E_func = EstepFunctions(data.X, data.T, data.delta, data.ext_feat,
                                     L, p, alpha, asso_functions, theta)
        self.ind_1, self.ind_2 = data.ind_1, data.ind_2

    def test_g1(self):
        """Tests the g1 function
        """
        self.setUp()
        g1 = self.E_func.g1(self.S, broadcast=False)
        g1_0_1 = np.exp(np.array([130 / 3, 221 / 6, 143 / 6, 142 / 3]))
        g1_1_3 = np.exp(np.array([147, 172.5, 145.5, 61.5]))
        np.testing.assert_almost_equal(g1[0, 0, :, 0], g1_0_1)
        np.testing.assert_almost_equal(g1[0, 1, :, 2], g1_1_3)

    def test_g2(self):
        """Tests the g2 function
        """
        self.setUp()
        g2 = self.E_func.g2(self.S, broadcast=False)
        # values of g2 at first group and first sample
        g2_0_1 = np.array([46, 53, 26, 1])
        # values of g2 at second group and second sample
        g2_1_3 = np.array([61, 52, 82, 31.5])
        np.testing.assert_almost_equal(g2[0, 0], g2_0_1)
        np.testing.assert_almost_equal(g2[1, 1], g2_1_3)

    def test_g5(self):
        """Tests the g5 function
        """
        self.setUp()
        g5 = self.E_func.g5(self.S, broadcast=False)
        g5_0_1 = np.array(
            [[1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 8 / 3],
             [1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 8 / 3],
             [1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 8 / 3]])
        g5_1_3 = np.array(
            [[1, 3, 9, 0, 0, 0, 0, 0, 0, 0, 1, 6, 3, 9 / 2, 9],
             [1, 3, 9, 0, 0, 0, 0, 0, 0, 0, 1, 6, 3, 9 / 2, 9],
             [1, 3, 9, 0, 0, 0, 0, 0, 0, 0, 1, 6, 3, 9 / 2, 9]])
        np.testing.assert_almost_equal(g5[0, 0, 0], g5_0_1)
        np.testing.assert_almost_equal(g5[1, 0, 1], g5_1_3)

    def test_g6(self):
        """Tests the g6 function
        """
        self.setUp()
        g6 = self.E_func.g6(self.S)
        g6_0_1 = np.exp(130 / 3) * np.array(
            [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1 / 2, 1 / 3],
             [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1 / 2, 1 / 3],
             [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1 / 2, 1 / 3]])
        np.testing.assert_almost_equal(g6[0, 0, 0, 0, :, :, 0], g6_0_1)

    def test_g7(self):
        """Tests the g7 function
        """
        self.setUp()
        g7 = self.E_func.g7(self.S, broadcast=False)
        g7_0_1 = np.array(
            [9, 6, 7, 1, 2, 3, 2, 3, 3, 10, 8, 4, 5, 8 / 3, 14 / 3])
        np.testing.assert_almost_equal(g7[0, 0, 0], g7_0_1)

    def test_g8(self):
        """Tests the g8 function
        """
        self.setUp()
        g8 = self.E_func.g8(self.S)
        g8_0_1 = np.exp(130 / 3) * \
                 np.array(
                     [9, 6, 7, 1, 2, 3, 2, 3, 3, 10, 8, 4, 5, 8 / 3, 14 / 3])
        np.testing.assert_almost_equal(g8[0, 0, 0, 0, :, 0], g8_0_1)

    def test_g9(self):
        """Tests the g9 function
        """
        self.setUp()
        g9 = self.E_func.g9(self.S)
        # values of g2 at first group and first sample
        g9_0_1 = np.array([-1019, -260])
        np.testing.assert_almost_equal(g9[0, 0, 0], g9_0_1)

    def test_f(self):
        """Tests the f_data_given_latent
        """
        self.setUp()
        f = self.E_func.f_data_given_latent(self.S, self.ind_1, self.ind_2)
        # TODO: Update later (after handling overflow problem)

    def test_Lambda_g(self):
        """Tests the Lambda_g function
        """
        self.setUp()
        g8 = self.E_func.g8(self.S)
        f = self.E_func.f_data_given_latent(self.S, self.ind_1, self.ind_2)
        E_g8 = self.E_func.Lambda_g(g8, f)
        # TODO: Update later

    def test_Eg(self):
        """Tests the expection of g functions
        """
        self.setUp()
        g8 = self.E_func.g8(self.S)
        f = self.E_func.f_data_given_latent(self.S, self.ind_1, self.ind_2)
        n_samples, n_MC, K = self.n_samples, self.n_MC, 2
        Lambda_1 = self.E_func.Lambda_g(np.ones(shape=(n_samples, K, n_MC)), f)
        pi_xi = 1 / (1 + np.exp(np.array([-3, -4, -6])))
        Eg = self.E_func.Eg(g8, Lambda_1, pi_xi, f)
        # TODO: Update later


if __name__ == "main":
    unittest.main()
