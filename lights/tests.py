# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.base.base import extract_features
from lights.simulation import SimuJointLongitudinalSurvival
from lights.init.mlmm import MLMM
from lights.init.ulmm import ULMM
from lights.inference import QNMCEM
import numpy as np
import pandas as pd


class Test(unittest.TestCase):

    @staticmethod
    def get_train_data(n_samples: int = 100):
        """Simulate data with specific seed
        """
        simu = SimuJointLongitudinalSurvival(n_samples=n_samples,
                                             n_time_indep_features=5,
                                             n_long_features=3, seed=123)
        X, Y, T, delta = simu.simulate()
        return X, Y, T, delta

    @staticmethod
    def get_Y_without_subgroups():
        """Simulate longitudinal data with no latent subgroups
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

    def test_SimuJointLongitudinalSurvival(self):
        """Test simulation of joint longitudinal and survival data
        """
        # Simulate data with specific seed
        X_, Y_, T_, delta_ = self.get_train_data(3)

        X = np.array(
            [[-0.9303042, -0.2824466, -1.1174361, -0.6888221, -1.4116331],
             [-0.4572906, -1.0588467, -0.19195, -0.7252352, 0.7797693],
             [1.3875948, 1.3412933, 1.3093861, 1.4140573, 0.6318637]])
        Y = pd.Series(
            [-0.13013141057043676, 4.880010452597576, 4.932839320006761,
             2.369292433812722, 4.238491778861706, 8.746396959115105,
             0.716335046844125, 4.767072897258849, 8.318728903335831,
             4.776731390501273, 6.573748926645436, 7.870938320604929,
             11.17552558555164, 7.112140386018595, 8.003975097441721,
             8.116707725260465, 13.476228627464568, 14.29213736872467,
             11.960818810617521, 10.939347415088234, 11.721248706341996,
             14.087026073436203],
            index=[1.6633365408297114, 2.946162760452098, 5.1975375610980405,
                   6.73992539423363, 6.760911635862214, 6.83359628093724,
                   7.0253229998602915, 7.109211645246458, 7.355322501073678,
                   8.454530094282653, 10.614343651792947, 10.810120667631654,
                   11.011299912590156, 11.310961001771295, 12.07378935010831,
                   12.21582777322665, 12.335812206852118, 13.63384495716701,
                   13.685732767613162, 14.011326583503408, 14.132190486000294,
                   15.437598944099662])
        T = np.array([22, 19, 12])
        delta = np.array([0, 0, 1], dtype=np.ushort)

        np.testing.assert_almost_equal(X, X_)
        pd.testing.assert_series_equal(Y, Y_.iloc[0, 0])
        np.testing.assert_almost_equal(T, T_)
        np.testing.assert_almost_equal(delta, delta_)

    def _test_initializer(self, initializer):
        """Test an initialization algorithm estimation
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
        """Test ULMM estimation
        """
        fixed_effect_time_order = 1  # q_l=2 in the simulations
        ulmm = ULMM(fixed_effect_time_order=fixed_effect_time_order)
        self._test_initializer(ulmm)

    def test_MLMM(self):
        """Test MLMM estimation
        """
        fixed_effect_time_order = 1  # q_l=2 in the simulations
        mlmm = MLMM(fixed_effect_time_order=fixed_effect_time_order,
                    initialize=False, tol=1e-4)
        self._test_initializer(mlmm)

    def test_QNMCEM(self):
        """Test QNMCEM Algorithm
        """
        X, Y, T, delta = self.get_train_data()
        qnmcem = QNMCEM(fixed_effect_time_order=1, max_iter=3, initialize=True)
        qnmcem.fit(X, Y, T, delta)


if __name__ == "main":
    unittest.main()
