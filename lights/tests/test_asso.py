# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np

from lights.tests.testing_data import CreateTestingData
from lights.model.associations import AssociationFunctions
from lights.model.e_step_functions import EstepFunctions
from lights.base.base import extract_features


class Test(unittest.TestCase):

    def test_lp_asso(self):
        """Tests the the linear predictor association function
        """
        data = CreateTestingData()
        X, Y, T, delta = data.X, data.Y, data.T, data.delta
        T_u = np.unique(T)
        N = data.N
        alpha = data.fixed_effect_time_order
        L, p = data.n_long_features, data.n_time_indep_features
        theta = data.theta

        ext_feat = extract_features(Y, alpha)  # Features extraction
        asso_functions = 'all'

        E_func = EstepFunctions(X, T, delta, ext_feat, L, p, alpha,
                                asso_functions, theta)
        S = E_func.construct_MC_samples(N)
        beta = np.array([theta["beta_0"], theta["beta_1"]])
        asso_func = AssociationFunctions(T_u, S, beta, alpha, L)

        #TODO: compute the expected value manually


if __name__ == "main":
    unittest.main()
