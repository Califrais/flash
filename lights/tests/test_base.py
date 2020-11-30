# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
import pandas as pd
from lights.base.base import get_vect_from_ext, get_xi_from_xi_ext, \
                            extract_features, block_diag


class Test(unittest.TestCase):

    def test_extract_features(self):
        """Test extract_features functions
        """
        data = [[pd.Series(np.array([2, 3, 5]), index=[1, 2, 3]),
                 pd.Series(np.array([1, 3]), index=[2, 4])],
                [pd.Series(np.array([2, 3]), index=[2, 4]),
                 pd.Series(np.array([3, 4, 5]), index=[1, 3, 5])]]
        columns = ['long_feature_%s' % (l + 1) for l in range(2)]
        Y = pd.DataFrame(data=data, columns=columns)
        q_l = 2 # fixed_effect_time_order
        (U, V, y, N), (U_L, V_L, y_L, N_L) = extract_features(Y, q_l)
        U_1 = np.array([[1, 1, 1, 0, 0, 0],
                        [1, 2, 4, 0, 0, 0],
                        [1, 3, 9, 0, 0, 0],
                        [0, 0, 0, 1, 2, 4],
                        [0, 0, 0, 1, 4, 16]])
        V_2 = np.array([[1, 2, 0, 0],
                        [1, 4, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 3],
                        [0, 0, 1, 5]])
        y_1 = np.array([2, 3, 5, 1, 3]).reshape(-1, 1)
        N_2 = [2, 3]

        np.testing.assert_almost_equal(U[0], U_1)
        np.testing.assert_almost_equal(V[1], V_2)
        np.testing.assert_almost_equal(y[0], y_1)
        np.testing.assert_almost_equal(N[1], N_2)

        # for l_th order
        U_1 = np.array([[1, 1, 1],
                        [1, 2, 4],
                        [1, 3, 9],
                        [1, 2, 4],
                        [1, 4, 16]])
        np.testing.assert_almost_equal(U_L[0], U_1)

    def test_get_vect_from_ext(self):
        """Test get_vect_from_ext function
        """
        v_ext = np.array([1, 0, 3, 0, 5, 0])
        v = get_vect_from_ext(v_ext)
        v_ = np.array([1, -5, 3])
        np.testing.assert_almost_equal(v, v_)

    def test_get_xi_from_xi_ext(self):
        """Test get_xi_from_xi_ext function
        """
        xi_ext = np.array([2, 1, 0, 3, 0, 0, 5, 0])
        xi_0, xi = get_xi_from_xi_ext(xi_ext, fit_intercept=True)
        xi_0_ = 2
        xi_ = np.array([1, -5, 3])
        np.testing.assert_almost_equal(xi_0, xi_0_)
        np.testing.assert_almost_equal(xi, xi_)

if __name__ == "main":
    unittest.main()
