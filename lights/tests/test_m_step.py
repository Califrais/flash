# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.model.m_step_functions import MstepFunctions
from lights.tests.testing_data import CreateTestingData
from lights.base.base import get_ext_from_vect


class Test(unittest.TestCase):

    def setUp(self):
        data = CreateTestingData()
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        l_pen = 2.
        eta_elastic_net = .2
        eta_sp_gp_l1 = .3
        fit_intercept = False
        self.M_func = MstepFunctions(fit_intercept, data.X, data.T, data.delta,
                                     L, p, l_pen, eta_elastic_net, eta_sp_gp_l1,
                                     data.nb_asso_feat, alpha)
        self.xi_ext = np.array([0, 2, 1, 0])
        self.pi_est = np.array([.2, .4, .7])
        self.data = data
        self.E_g1 = np.array([[1, 2],
                              [3, 5],
                              [4, 6]])
        self.E_g2 = np.array([1, 4, 5])
        self.E_g8 = np.array([1, 5, 2])

    def test_P_func(self):
        """Tests the P function
        """
        self.setUp()
        P = self.M_func.P_func(self.pi_est, self.xi_ext)
        P_ = .093
        np.testing.assert_almost_equal(P, P_, 3)

    def test_grad_P(self):
        """Tests the gradient of P function
        """
        self.setUp()
        grad_P = self.M_func.grad_P(self.pi_est, self.xi_ext)
        grad_P_ = np.array([-.133, -.069, .133, .069])
        np.testing.assert_almost_equal(grad_P, grad_P_, 3)

    def test_R_func(self):
        """Tests the R function
        """
        self.setUp()
        ind = self.data.T.reshape(-1, 1) >= np.unique(self.data.T)
        R = self.M_func.R_func(self.pi_est, self.E_g1, self.E_g2, self.E_g8,
                               self.data.theta["baseline_hazard"], ind)
        R_ = 21.1
        np.testing.assert_almost_equal(R, R_, 3)

    def test_grad_R(self):
        """Tests the gradient of R function
        """
        self.setUp()
        ind = self.data.T.reshape(-1, 1) >= np.unique(self.data.T)
        theta = self.data.theta
        beta_ext = get_ext_from_vect(theta["beta_0"])
        gamma_ext = get_ext_from_vect(theta["gamma_0"])
        baseline_hazard = theta["baseline_hazard"]
        phi = theta["phi"]
        ext_feat = self.data.ext_feat
        E_g5 = np.arange(1, 136).reshape((3, 3, 15))
        E_g6 = np.arange(1, 271).reshape((3, 2, 3, 15))
        E_gS = np.arange(1, 19).reshape(3, 6)
        grad_R = self.M_func.grad_R(beta_ext, gamma_ext, self.pi_est, E_g5,
                                    E_g6,
                                    E_gS, baseline_hazard, ind, ext_feat, phi)
        grad_R_ = np.array([2579.433, 2694.2, 3027.733, 1956.267, 2423.667,
                            4571.267, 4180.3, 4514, 5677.2, -2579.433, -2694.2,
                            -3027.733, -1956.267, -2423.667, -4571.267, -4180.3,
                            -4514, -5677.2])
        np.testing.assert_almost_equal(grad_R, grad_R_, 3)

    def test_Q_func(self):
        """Tests the Q function
        """
        self.setUp()
        ind_1 = self.data.T.reshape(-1, 1) == np.unique(self.data.T)
        ind_2 = self.data.T.reshape(-1, 1) >= np.unique(self.data.T)
        baseline_hazard = self.data.theta["baseline_hazard"]
        Q = self.M_func.Q_func(self.pi_est, np.log(self.E_g1), self.E_g1,
                               baseline_hazard, ind_1, ind_2)
        Q_ = 23.115
        np.testing.assert_almost_equal(Q, Q_, 3)

    def test_grad_Q(self):
        """Test the gradient of Q function
        """
        self.setUp()
        ind_1 = self.data.T.reshape(-1, 1) == np.unique(self.data.T)
        ind_2 = self.data.T.reshape(-1, 1) >= np.unique(self.data.T)
        baseline_hazard = self.data.theta["baseline_hazard"]
        E_g1 = np.arange(1, 7).reshape((3, 2))
        E_g7 = np.arange(1, 91).reshape((3, 2, 15))
        E_g8 = np.arange(1, 91).reshape(3, 2, 15)
        grad_Q = self.M_func.grad_Q(self.pi_est, E_g1, E_g7, E_g8,
                                    baseline_hazard, ind_1, ind_2)
        grad_Q_ = np.array([12.267, -12.267, 57.2, -57.2, 265.3, -265.3, 270.1,
                            -270.1, 274.9, -274.9, 279.7, -279.7, 284.5, -284.5,
                            289.3, -289.3, 294.1, -294.1, 298.9, -298.9, 303.7,
                            -303.7, 308.5, -308.5, 313.3, -313.3, 318.1, -318.1,
                            322.9, -322.9, 327.7, -327.7, 332.5, -332.5])
        np.testing.assert_almost_equal(grad_Q, grad_Q_, 3)


if __name__ == "main":
    unittest.main()
