# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.simulation import SimuJointLongitudinalSurvival
import numpy as np


class Test(unittest.TestCase):

    def test_SimuJointLongitudinalSurvival(self):
        """Test simulation of joint longitudinal and survival data
        """
        # Simulate data with specific seed
        simu = SimuJointLongitudinalSurvival(n_samples=3,
                                             n_time_indep_features=5,
                                             n_long_features=3,
                                             seed=123, verbose=False)
        X_, Y_, T_, delta_ = simu.simulate()

        T = np.array([23.4050397, 20.21906185, 15.93688448])
        delta = np.array([0, 0, 1], dtype=np.ushort)
        X = np.array(
            [[-1.3854538, -1.4562842, -1.7882523, -1.387485, -1.3576753],
             [-0.9514631, -2.1464529, -0.9993042, -1.420507,  0.8139507],
             [0.7412221, -0.0128858,  0.2805381,  0.5195572,  0.6673799]])

        np.testing.assert_almost_equal(X, X_)
        np.testing.assert_almost_equal(T, T_)
        np.testing.assert_almost_equal(delta, delta_)

    @staticmethod
    def get_train_data(seed: int = 1):
        """Get train data for specific tests
        """
        np.random.seed(seed)
        simu = SimuJointLongitudinalSurvival()
        features, times, censoring = simu.simulate()
        return features, times, censoring


if __name__ == "main":
    unittest.main()
