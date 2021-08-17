# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.model.e_step_functions import EstepFunctions
from lights.model.m_step_functions import MstepFunctions
from lights.tests.testing_data import CreateTestingData
from lights.base.base import get_ext_from_vect, get_times_infos


class Test(unittest.TestCase):

    def setUp(self):
        data = CreateTestingData()
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        n_samples, X, delta = data.n_samples, data.X, data.delta
        K, r_l = 2, 2
        r = L * r_l
        l_pen = 2.
        eta_elastic_net = .2
        fit_intercept = False
        T = data.T
        T_u = np.unique(T)
        J = len(T_u)
        asso_functions = data.asso_functions
        nb_asso_param = L * len(asso_functions)
        if 're' in asso_functions:
            nb_asso_param += L
        _, self.ind_1, self.ind_2 = get_times_infos(T, T_u)
        self.delta_T = T_u - np.append(0, T_u[:-1])
        self.M_func = MstepFunctions(fit_intercept, data.X, data.T, data.delta,
                                     L, p, l_pen, eta_elastic_net, alpha,
                                     asso_functions)
        self.xi_ext = np.array([0, 2, 1, 0])
        self.pi_est = np.array([[.2, .4, .7], [.8, .6, .3]])
        self.data = data
        self.E_g1 = np.arange(1, 13).reshape(n_samples, J, K)
        self.E_g2 = np.array([1, 4, 5])
        self.E_g4 = .5 * np.ones(shape=(n_samples, r, r))
        self.E_g5 = np.ones(shape=(n_samples, r))
        self.E_g7 = np.arange(1, 181).reshape((n_samples, J, nb_asso_param, K))
        self.E_g8 = np.arange(1, 181).reshape((n_samples, J, nb_asso_param, K))

    def test_P_func(self):
        """Tests the P function
        """
        self.setUp()
        P = self.M_func.P_func(self.pi_est[0], self.xi_ext)
        P_ = .093
        np.testing.assert_almost_equal(P, P_, 3)

    def test_grad_P(self):
        """Tests the gradient of P function
        """
        self.setUp()
        grad_P = self.M_func.grad_P(self.pi_est[0], self.xi_ext)
        grad_P_ = np.array([-.133, -.069, .133, .069])
        np.testing.assert_almost_equal(grad_P, grad_P_, 3)

    def test_Q_func(self):
        """Tests the Q function
        """
        self.setUp()
        theta = self.data.theta
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        beta, gamma = self.data.beta, self.data.gamma
        E_g1 = lambda v: self.E_g1
        E_log_g1 = lambda v: np.log(self.E_g1)
        args = {"pi_est": self.pi_est, "E_g5": self.E_g5,
                    "phi": phi, "beta": beta,
                    "baseline_hazard": baseline_hazard,
                    "extracted_features": self.data.ext_feat,
                    "ind_1": self.ind_1, "ind_2": self.ind_2,
                    "E_g1": E_g1,"E_log_g1": E_log_g1,
                    "group": 0, "delta_T": self.delta_T}
        Q = self.M_func.Q_func(gamma[0], {**args})
        Q_ = 55.774
        np.testing.assert_almost_equal(Q, Q_, 3)

    def test_grad_Q(self):
        """Test the gradient of Q function
        """
        self.setUp()
        theta = self.data.theta
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        beta, gamma = self.data.beta, self.data.gamma
        E_g1 = lambda v: self.E_g1
        E_log_g1 = lambda v: np.log(self.E_g1)
        E_g7 = self.E_g7
        E_g8 = lambda v: self.E_g8
        args = {"pi_est": self.pi_est, "E_g5": self.E_g5,
                "phi": phi, "beta": beta,
                "baseline_hazard": baseline_hazard,
                "extracted_features": self.data.ext_feat,
                "ind_1": self.ind_1, "ind_2": self.ind_2,
                "E_g1": E_g1, "E_log_g1": E_log_g1,
                "E_g7": E_g7, "E_g8": E_g8, "group": 0, "delta_T": self.delta_T}
        grad_Q = self.M_func.grad_Q(gamma[0], {**args})
        grad_Q_ = np.array([707.967, 721.9, 735.833, 749.767, 763.7, 777.633,
                            791.567, 805.5, 819.433, 833.367, 847.3, 861.233,
                            875.167, 889.1,903.033])
        np.testing.assert_almost_equal(grad_Q, grad_Q_, 3)


if __name__ == "main":
    unittest.main()
