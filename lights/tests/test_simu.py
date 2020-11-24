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


def get_testing_data():
    """Simulate data with specific seed

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

    simu : simulation object
        The simulation object
    """
    simu = SimuJointLongitudinalSurvival(n_samples=3, n_long_features=2,
                                         n_time_indep_features=2, seed=123)
    X, Y, T, delta = simu.simulate()
    return X, Y, T, delta, simu


class Test(unittest.TestCase):

    def test_SimuJointLongitudinalSurvival(self):
        """Tests simulation of joint longitudinal and survival data
        """
        # Simulate data with specific seed
        X_, Y_, T_, delta_ = get_train_data(3)

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


if __name__ == "main":
    unittest.main()
