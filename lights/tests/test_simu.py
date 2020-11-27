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
            [[1.386799, 1.0334236, 0.507863, -1.3956483, 0.4086073],
             [-0.9333817, -1.35279, -1.3969785, 0.8956231, 0.9682065],
             [-0.4534173, 0.3193664, 0.8891154, 0.5000251, -1.3768138]])
        Y = pd.Series(
            [-0.5385306223921831, 4.932741121587971, 4.147877394152085,
             0.5784366627985951, 2.4683445199288148, 7.02599079884744,
             -1.0851485857213383, 3.048712840685921, 6.953281112459232,
             3.531091712970734, 3.991676690327542, 5.397134199261322,
             9.346627903899002, 5.106227515132062, 5.6398559021760075],
            index=[1.6954916374868019, 5.250813641738383, 7.004534113277996,
                   7.029335447848361, 7.12335527159349, 7.389115747704781,
                   7.511934099306935, 7.888701036025391, 9.291062836214603,
                   11.587603197022105, 11.818782743239492, 12.464584733063488,
                   14.581720879859912, 14.636693280979431, 15.02833203039368])
        T = np.array([19, 38, 11])
        delta = np.array([1, 1, 0], dtype=np.ushort)

        np.testing.assert_almost_equal(X, X_)
        pd.testing.assert_series_equal(Y, Y_.iloc[0, 0])
        np.testing.assert_almost_equal(T, T_)
        np.testing.assert_almost_equal(delta, delta_)


if __name__ == "main":
    unittest.main()
