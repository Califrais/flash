# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.base.base import extract_features
from lights.simulation import SimuJointLongitudinalSurvival
from lights.init.mlmm import MLMM
from lights.init.ulmm import ULMM
import numpy as np


class Test(unittest.TestCase):

    @staticmethod
    def get_Y_without_subgroups():
        """Simulate longitudinal data with no latent subgroups

        Returns
        -------
        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        beta : `np.ndarray`, shape=(q,)
            Simulated fixed effect coefficient vector

        D : `np.ndarray`, shape=(2*n_long_features, 2*n_long_features)
            Variance-covariance matrix that accounts for dependence between the
            different longitudinal outcome. Here r = 2*n_long_features since
            one choose affine random effects, so all r_l=2

        phi : `np.ndarray`, shape=(n_long_features,)
            Variance vector for the error term of the longitudinal processes
        """
        simu = SimuJointLongitudinalSurvival(n_samples=100,
                                             n_time_indep_features=5,
                                             n_long_features=3, seed=123,
                                             high_risk_rate=0)
        Y = simu.simulate()[1]
        beta = simu.fixed_effect_coeffs[0]
        D = simu.long_cov
        phi_l = simu.std_error ** 2
        n_long_features = simu.n_long_features
        phi = np.repeat(phi_l, n_long_features).reshape(-1, 1)
        return Y, beta, D, phi

    def _test_initializer(self, initializer):
        """Tests an initialization algorithm estimation
        """
        Y, beta_, D_, phi_ = self.get_Y_without_subgroups()
        fixed_effect_time_order = initializer.fixed_effect_time_order
        extracted_features = extract_features(Y, fixed_effect_time_order)
        initializer.fit(extracted_features)
        beta = initializer.fixed_effect_coeffs
        D, phi = initializer.long_cov, initializer.phi

        decimal = 0
        np.testing.assert_almost_equal(beta, beta_, decimal=decimal)
        np.testing.assert_almost_equal(D, D_, decimal=decimal)
        np.testing.assert_almost_equal(phi, phi_, decimal=decimal)

    def test_ULMM(self):
        """Tests ULMM estimation
        """
        fixed_effect_time_order = 1  # q_l=2 in the simulations
        ulmm = ULMM(fixed_effect_time_order=fixed_effect_time_order)
        self._test_initializer(ulmm)

    def test_MLMM(self):
        """Tests MLMM estimation
        """
        fixed_effect_time_order = 1  # q_l=2 in the simulations
        mlmm = MLMM(fixed_effect_time_order=fixed_effect_time_order,
                    initialize=False, tol=1e-4)
        self._test_initializer(mlmm)


if __name__ == "main":
    unittest.main()
