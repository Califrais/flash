# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.simulation import SimuJointLongitudinalSurvival
import numpy as np
import pandas as pd


def get_train_data(n_samples: int = 100):
    """Simulate data with specific seed

    Parameters
    ----------
    n_samples : `int`, default=100
        Desired number of samples

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_time_indep_features)
        The simulated time-independent features matrix

    Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
        The simulated longitudinal data. Each element of the dataframe is
        a pandas.Series

    T : `np.ndarray`, shape=(n_samples,)
        The simulated censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        The simulated censoring indicator
    """
    simu = SimuJointLongitudinalSurvival(n_samples=n_samples,
                                         n_time_indep_features=5,
                                         n_long_features=3, seed=123)
    X, Y, T, delta = simu.simulate()
    return X, Y, T, delta


class Test(unittest.TestCase):

    def test_SimuJointLongitudinalSurvival(self):
        """Tests simulation of joint longitudinal and survival data
        """
        # Simulate data with specific seed
        X_, Y_, T_, delta_ = get_train_data(3)

        X = np.array(
            [[1.38679898, 1.03342361, 0.50786304, -1.39564826, 0.40860729],
             [-0.93338165, -1.35278999, -1.39697848, 0.89562312, 0.96820651],
            [-0.45341733, 0.31936638, 0.88911544, 0.50002514, -1.3768138]])
        Y = pd.Series(
            [-0.84189968, 4.34368637, 3.41790352, -0.15353011, 1.72882284,
             6.2651141, -1.85589427, 2.24769228, 6.03957464, 2.43284816,
             2.87485685, 4.22842134],
            index=[1.6954916374868019,  5.250813641738383,  7.004534113277996,
               7.029335447848361,   7.12335527159349,  7.389115747704781,
               7.511934099306935,  7.888701036025391,  9.291062836214603,
              11.587603197022105, 11.818782743239492, 12.464584733063488])
        T = np.array([16, 32, 11])
        delta = np.array([1, 1, 0], dtype=np.ushort)

        np.testing.assert_almost_equal(X, X_)
        pd.testing.assert_series_equal(Y, Y_.iloc[0, 0])
        np.testing.assert_almost_equal(T, T_)
        np.testing.assert_almost_equal(delta, delta_)


if __name__ == "main":
    unittest.main()
