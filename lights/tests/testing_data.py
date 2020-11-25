# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>


import numpy as np
import pandas as pd
from lights.base.base import extract_features

class CreateTestingData:
    """A class to create testing data
        """

    def __init__(self):
        self.n_long_features = 3
        self.n_samples = 3
        self.n_time_indep_features = 2
        self.fixed_effect_time_order = 2
        self.N = 2 # number of Monte-Carlo samples

        self.X = np.array([[-1, 2], [2, 1], [0, 3]])
        data = [[pd.Series(np.array([2, 3, 5]),index=[1, 2, 3]),
                pd.Series(np.array([3, 4]),index=[1, 3]),
                pd.Series(np.array([1, 3]),index=[2, 4])],
                [pd.Series(np.array([2, 3]), index=[2, 4]),
                 pd.Series(np.array([3, 4, 5]), index=[1, 3, 5]),
                 pd.Series(np.array([2, 3, 6]), index=[1, 3, 4])],
                [pd.Series(np.array([3, 5]), index=[2, 3]),
                 pd.Series(np.array([1, 3, 4]), index=[1, 2, 5]),
                 pd.Series(np.array([2, 3]), index=[2, 4])]]
        columns = ['long_feature_%s' % (l + 1) for l in range(self.n_long_features)]
        self.Y = pd.DataFrame(data=data, columns=columns)
        self.T = np.array([22, 19, 12])
        self.delta = np.array([1, 0, 1], dtype=np.ushort)
        baseline_hazard = pd.Series(data=np.array([5, 8, 12]), index=self.T)
        beta_0 = np.array([1, 2, 3, -3, 2, 2, -1, 3, -1]).reshape(-1, 1)
        beta_1 = np.array([-1, -2, 2, 2, 3, 1, 1, 2, -1]).reshape(-1, 1)
        gamma_0 = np.array([-1, 1, 1, 0, 2, -1, 1, 2, -2, 3, -1, 0,
                            -2, 1, 3, 2, 0]).reshape(-1, 1)
        gamma_1 = np.array([2, -2, 1, 1, -3, 1, 2, -1, -1, -3, 1, 1,
                            0, 2, 1, 3, -1]).reshape(-1, 1)
        D = np.array([[2, 1, 3, 3, 2, 3],
                      [1, 3, 4, 4, 3, 1],
                      [3, 4, 1, 2, 1, 2],
                      [3, 4, 2, 2, 2, 2],
                      [2, 3, 1, 2, 2, 1],
                      [3, 1, 2, 2, 1, 1]])
        phi = np.array([1, 2, 3])
        xi = np.array([1, 2])
        self.theta = {
            "beta_0": beta_0,
            "beta_1": beta_1,
            "long_cov": D,
            "phi": phi,
            "xi": xi,
            "baseline_hazard": baseline_hazard,
            "gamma_0": gamma_0,
            "gamma_1": gamma_1
        }
        # Features extraction
        self.ext_feat = extract_features(self.Y, self.fixed_effect_time_order)
        self.S = np.array([[1, 2, 3, 2, 3, 3],
                      [-1, 3, 2, 4, 5, 1],
                      [2, 3, -1, -2, 1, -3],
                      [-3, 2, 3, -3, 4, 1]])
