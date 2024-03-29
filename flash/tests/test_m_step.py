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
        self.M_func = MstepFunctions(fit_intercept, data.X, data.delta,
                                     p, l_pen, eta_elastic_net)
        self.xi_ext = np.array([0, 2, 1, 0])
        self.pi_est = np.array([[.2, .4, .7], [.8, .6, .3]])
        self.data = data
        self.E_g1 = np.ones(shape=(n_samples, r))
        self.E_g2 = .5 * np.ones(shape=(n_samples, r, r))
        self.E_g3 = np.arange(1, 181).reshape((n_samples, J, nb_asso_param, K))
        self.E_g4 = np.arange(1, 13).reshape(n_samples, J, K)
        self.E_g5 = np.arange(1, 181).reshape((n_samples, J, nb_asso_param, K))

    def test_P_func(self):
        """Tests the P function
        """
        self.setUp()
        P = self.M_func.P_func(self.pi_est[1], self.xi_ext)
        P_ = 1.967
        np.testing.assert_almost_equal(P, P_, 3)

    def test_grad_P(self):
        """Tests the gradient of P function
        """
        self.setUp()
        grad_P = self.M_func.grad_P(self.pi_est[1], self.xi_ext)
        grad_P_ = np.array([-.131, .793, .131, -.793])
        np.testing.assert_almost_equal(grad_P, grad_P_, 3)

    def test_Q_func(self):
        """Tests the Q function
        """
        self.setUp()
        theta = self.data.theta
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        beta, gamma = self.data.beta, self.data.gamma
        E_g4 = lambda v: self.E_g4
        E_log_g4 = lambda v: np.log(self.E_g4)
        args = {"pi_est": self.pi_est, "E_g1": self.E_g1,
                    "phi": phi, "beta": beta,
                    "baseline_hazard": baseline_hazard,
                    "extracted_features": self.data.ext_feat,
                    "ind_1": self.ind_1, "ind_2": self.ind_2,
                    "E_g4": E_g4,"E_log_g4": E_log_g4, "group": 0}
        Q = self.M_func.Q_func(gamma[0], {**args})
        Q_ = 41.607
        np.testing.assert_almost_equal(Q, Q_, 3)

    def test_grad_Q(self):
        """Test the gradient of Q function
        """
        self.setUp()
        theta = self.data.theta
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        beta, gamma = self.data.beta, self.data.gamma
        E_g4 = lambda v: self.E_g4
        E_log_g4 = lambda v: np.log(self.E_g4)
        E_g3 = self.E_g3
        E_g5 = lambda v: self.E_g5
        args = {"pi_est": self.pi_est, "E_g1": self.E_g1,
                "phi": phi, "beta": beta,
                "baseline_hazard": baseline_hazard,
                "extracted_features": self.data.ext_feat,
                "ind_1": self.ind_1, "ind_2": self.ind_2,
                "E_g4": E_g4, "E_log_g4": E_log_g4,
                "E_g3": E_g3, "E_g5": E_g5, "group": 0}
        grad_Q = self.M_func.grad_Q(gamma[0], {**args})
        grad_Q_ = np.array([525.8, 535.4, 545, 554.6, 564.2, 573.8, 583.4,
                            593, 602.6, 612.2, 621.8, 631.4, 641, 650.6, 660.2])
        np.testing.assert_almost_equal(grad_Q, grad_Q_, 3)


if __name__ == "main":
    unittest.main()
