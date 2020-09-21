# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.simulation import SimuJointLongitudinalSurvival
from lights.mlmm import MLMM
from lights.inference import QNMCEM
import numpy as np
import pandas as pd
from lights.base import Learner


class Test(unittest.TestCase):

    def test_SimuJointLongitudinalSurvival(self):
        """Test simulation of joint longitudinal and survival data
        """
        # Simulate data with specific seed
        simu = SimuJointLongitudinalSurvival(n_samples=3,
                                             n_time_indep_features=5,
                                             n_long_features=3, seed=123)
        X_, Y_, T_, delta_ = simu.simulate()

        X = np.array(
            [[-1.3854538, -1.4562842, -1.7882523, -1.387485, -1.3576753],
             [-0.9514631, -2.1464529, -0.9993042, -1.420507, 0.8139507],
             [0.7412221, -0.0128858, 0.2805381, 0.5195572, 0.6673799]])
        Y = pd.Series(
            [-0.13013141, 4.88001045, 4.93283932, 2.36929243, 4.23849178,
             8.74639696, 0.71633505, 4.7670729, 8.3187289, 4.77673139,
             6.57374893, 7.87093832, 11.17552559, 7.11214039, 8.0039751,
             8.11670773, 13.47622863, 14.29213737, 11.96081881, 10.93934742,
             11.72124871, 14.08702607, 11.87008894, 16.63307775, 11.78707803,
             13.37559342, 16.49980905],
            index=[1.6633365408297114, 2.946162760452098, 5.1975375610980405,
                   6.73992539423363, 6.760911635862214, 6.83359628093724,
                   7.0253229998602915, 7.109211645246458, 7.355322501073678,
                   8.454530094282653, 10.614343651792947, 10.810120667631654,
                   11.011299912590156, 11.310961001771295, 12.07378935010831,
                   12.21582777322665, 12.335812206852118, 13.63384495716701,
                   13.685732767613162, 14.011326583503408, 14.132190486000294,
                   15.437598944099662, 19.44354615720463, 20.263615119583594,
                   20.284007639052884, 20.825373947825646, 20.877803345057956])
        T = np.array([31.4839339, 27.198228, 25.9627587])
        delta = np.array([0, 0, 1], dtype=np.ushort)

        np.testing.assert_almost_equal(X, X_)
        pd.testing.assert_series_equal(Y, Y_.iloc[0, 0])
        np.testing.assert_almost_equal(T, T_)
        np.testing.assert_almost_equal(delta, delta_)

    @staticmethod
    def get_train_data(seed: int = 1):
        """Get train data for specific tests
        """
        simu = SimuJointLongitudinalSurvival(seed=seed)
        X, Y, T, delta = simu.simulate()
        return X, Y, T, delta

    def test_MLMM(self):
        """Test MLMM estimation
        """
        simu = SimuJointLongitudinalSurvival(n_samples=100,
                                             n_time_indep_features=5,
                                             n_long_features=3, seed=123, high_risk_rate=0)
        # simulation with no latent subgroups
        Y = simu.simulate()[1]
        beta_ = simu.fixed_effect_coeffs[0]
        D_ = simu.long_cov
        phi_l = simu.std_error
        n_long_features = simu.n_long_features
        phi_ = np.repeat(phi_l, n_long_features)

        # TODO Van Tuan : here we simulate with high_risk_rate=0 so with no
        #  latent subgroups. We expect MLMM to recover approximately the true
        #  beta_, D_, phi_ . Use assert_almost_equal with a decimal not to
        #  big to allow estimation error ;)

        fixed_effect_time_order = 1
        mlmm = MLMM(fixed_effect_time_order=fixed_effect_time_order)
        # TODO: extracted features to test MLMM case (similar when calling from inference.py)
        extracted_features = Learner.extract_features(Y, fixed_effect_time_order)
        mlmm.fit(extracted_features)
        beta, D, phi = np.concatenate(mlmm.beta), mlmm.D, np.concatenate(mlmm.phi)

        decimal = 2
        np.testing.assert_almost_equal(beta, beta_, decimal=decimal)
        np.testing.assert_almost_equal(D, D_, decimal=decimal)
        np.testing.assert_almost_equal(phi, phi_, decimal=decimal)

    def test_QNMCEM(self):
        """Test QNMCEM Algorithm
        """
        simu = SimuJointLongitudinalSurvival(seed=123)
        X, Y, T, delta = simu.simulate()
        D = simu.long_cov
        # TODO Sim : check parameters estimation


if __name__ == "main":
    unittest.main()
