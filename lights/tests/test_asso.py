# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.tests.test_simu import get_testing_data
from lights.model.associations import AssociationFunctions
from lights.model.e_step_functions import EstepFunctions
from lights.base.base import extract_features


class Test(unittest.TestCase):

    def test_lp_asso(self):
        """Tests the the linear predictor association function
        """
        X, Y, T, delta, simu = get_testing_data()
        T_u = np.unique(T)
        N = 2
        alpha = simu.fixed_effect_time_order
        L, p = simu.n_long_features, simu.n_time_indep_features
        beta = simu.fixed_effect_coeffs
        theta = simu.theta

        ext_feat = extract_features(Y, alpha)  # Features extraction
        asso_functions = 'all'

        E_func = EstepFunctions(X, T, delta, ext_feat, L, p, alpha,
                                asso_functions, theta)
        S = E_func.construct_MC_samples(N)
        asso_func = AssociationFunctions(T_u, S, beta, alpha, L)


if __name__ == "main":
    unittest.main()
