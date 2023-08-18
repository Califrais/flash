# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np

import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
from rpy2 import robjects

from dataset import Dataset
from scipy.stats import beta


def load_PBC_Seq():
    robjects.r.source(os.getcwd() + "/load_PBC_Seq.R")
    time_indep_feat = ['drug', 'age', 'sex']
    time_dep_feat = ['serBilir', 'albumin', 'SGOT', 'platelets',
                     'prothrombin', 'alkaline', 'serChol']
    data_R = robjects.r["load"]()
    # TODO: encoder and normalize
    # preprocessing
    data_competing = pd.DataFrame(data_R).T
    data_competing.columns = data_R.colnames
    data_competing = data_competing[(data_competing > -1e-4).all(axis=1)]
    for feat in time_dep_feat:
        data_competing[feat] = np.log(data_competing[feat].values)
    id_list = np.unique(data_competing["id"])
    n_samples = len(id_list)
    n_long_features = len(time_dep_feat)
    data_lights = data_competing.drop_duplicates(subset=["id"])

    # generate t_max
    a = 2
    b = 5
    np.random.seed(0)
    r = beta.rvs(a, b, size=n_samples)
    T = data_lights["T_survival"].values
    t_max = T * (1 - r)

    Y = []
    t_max_R = []
    Y_rep = pd.DataFrame(columns=["id", "time", "kind", "value"])
    for i in range(n_samples):
        tmp = data_competing[(data_competing["id"] == id_list[i])
                             & (data_competing["T_long"] < t_max[i])]
        if tmp.empty:
            t_max[i] = data_competing[
                (data_competing["id"] == id_list[i])]["T_long"].values[0]
            n_i = 1
        else:
            n_i = tmp.shape[0]
        data_competing = data_competing[(data_competing["id"] != id_list[i]) |
                    ((data_competing["id"] == id_list[i]) &
                     (data_competing["T_long"] <= t_max[i]))]
        y_i = []
        for l in range(n_long_features):
            Y_il = data_competing[["T_long", time_dep_feat[l]]][
                (data_competing["id"] == id_list[i]) & (data_competing['T_long'] <= t_max[i])]
            # TODO: Add value of 1/365 (the first day of survey instead of 0)
            y_il = Y_il[time_dep_feat[l]].values
            t_il = Y_il["T_long"].values + 1 / 365
            n_il = len(t_il)
            y_i += [pd.Series(y_il, index=t_il)]
            tmp = {"id": [i] * n_il,
                   "time": t_il,
                   "kind": ["long_feat_" + str(l)] * n_il,
                   "value": y_il}
            Y_rep = Y_rep.append(pd.DataFrame(tmp), ignore_index=True)
        Y.append(y_i)
        t_max_R += [t_max[i]] * n_i
    data_lights[time_dep_feat] = Y
    data_lights["T_max"] = t_max
    data_competing["T_max"] = np.array(t_max_R).flatten()

    dataset = Dataset(
        name="PBC_Seq",
        time_indep_columns=time_indep_feat,
        time_dep_columns=time_dep_feat,
        df_lights=data_lights,
        df_competing=data_competing,
        Y_rep=Y_rep
    )
    return dataset


# if __name__ == "__main__":
#
#     load_PBC_Seq()
