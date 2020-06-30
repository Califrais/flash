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
        simu = SimuJointLongitudinalSurvival(n_samples=10000,
                                             n_time_indep_features=30,
                                             n_long_features=50,
                                             seed=123, verbose=False)
        X_, Y_, T_, delta_ = simu.simulate()

        T = np.array([
            1.5022119, 5.93102441, 6.82837051, 0.50940341, 0.14859682,
            30.22922996, 3.54945974, 0.8671229, 1.4228358, 0.11483298
        ])

        delta = np.array([1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                             dtype=np.ushort)

        X = np.array([[1.4912667, 0.80881799, 0.26977298], [
            1.23227551, 0.50697013, 1.9409132
        ], [1.8891494, 1.49834791,
            2.41445794], [0.19431319, 0.80245126, 1.02577552], [
                                 -1.61687582, -1.08411865, -0.83438387
                             ], [2.30419894, -0.68987056,
                                 -0.39750262],
                             [-0.28826405, -1.23635074, -0.76124386], [
                                 -1.32869473, -1.8752391, -0.182537
                             ], [0.79464218, 0.65055633, 1.57572506],
                             [0.71524202, 1.66759831, 0.88679047]])

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
