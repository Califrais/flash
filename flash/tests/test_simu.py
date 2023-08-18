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

    S_k : `list`
        Set of nonactive group for 2 classes
    """
    simu = SimuJointLongitudinalSurvival(n_samples=n_samples,
                                         n_time_indep_features=5,
                                         n_long_features=3, seed=123)
    X, Y, T, delta, S_k = simu.simulate()
    return X, Y, T, delta, S_k


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
            [8.262191, 9.394127, 9.008120, 12.359450, 13.497597],
            index=[6.731722, 6.806727, 7.333566, 10.970851, 12.200360])
        T = np.array([16, 12, 21])
        delta = np.array([0, 0, 1], dtype=np.ushort)

        np.testing.assert_almost_equal(X, X_)
        pd.testing.assert_series_equal(Y, Y_.iloc[0, 0])
        np.testing.assert_almost_equal(T, T_)
        np.testing.assert_almost_equal(delta, delta_)


if __name__ == "main":
    unittest.main()
