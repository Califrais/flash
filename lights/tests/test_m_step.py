# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.model.m_step_functions import MstepFunctions
from lights.tests.testing_data import CreateTestingData

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
        """TODO
        """

    def test_grad_R(self):
        """TODO
        """

    def test_Q_func(self):
        """TODO
        """

    def test_grad_Q(self):
        """TODO
        """


if __name__ == "main":
    unittest.main()
