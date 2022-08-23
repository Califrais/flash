import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri
import numpy as np
from lights.simulation import SimuJointLongitudinalSurvival
import pandas as pd
from scipy.stats import beta
def load_data(simu, seed):
    if simu:
        n_long_features = 5
        n_time_indep_features = 10
        n_samples = 200
        simu = SimuJointLongitudinalSurvival(seed=seed,
                                             n_long_features=n_long_features,
                                             n_samples=n_samples,
                                             n_time_indep_features=n_time_indep_features,
                                             sparsity=0.5, grid_time=True)
        X, Y, T, delta, _, Y_rep, t_max = simu.simulate()
        id = np.arange(n_samples)
        time_dep_feat = ['long_feature_%s' % (l + 1)
                         for l in range(n_long_features)]
        time_indep_feat = ['X_%s' % (l + 1)
                           for l in range(n_time_indep_features)]

        data_lights = pd.DataFrame(data=np.column_stack((id, T, delta, X, Y)),
                            columns=["id", "T_survival", "delta"] +
                                    time_indep_feat + time_dep_feat)
        df1 = pd.DataFrame(data=np.column_stack((id, T, delta, X)),
                           columns=["id", "T_survival", "delta"] + time_indep_feat)
        for i in range(n_samples):
            Y_i_ = []
            for l in range(n_long_features):
                Y_il = Y.loc[i][l]
                times_il = Y_il.index.values
                y_il = Y_il.values.flatten().tolist()
                n_il = len(times_il)
                Y_i_.append(y_il)
            Y_i = np.column_stack(
                (np.array([id[i]] * n_il), times_il, np.array([t_max[i]] * n_il), np.array(Y_i_).T))
            if i == 0:
                Y_ = Y_i
            else:
                Y_ = np.row_stack((Y_, Y_i))
        data_lights["T_max"] = t_max
        df2 = pd.DataFrame(data=Y_, columns=["id", "T_long", "T_max"] + time_dep_feat)
        data = pd.merge(df2, df1, on="id")
    else:
        # load PBC Seq
        robjects.r.source(os.getcwd() + "/competing_methods/load_PBC_Seq.R")
        time_indep_feat = ['drug', 'age', 'sex']
        time_dep_feat = ['serBilir', 'albumin', 'SGOT', 'platelets',
                         'prothrombin', 'alkaline', 'serChol']
        data_R = robjects.r["load"]()
        # TODO: encoder and normalize
        data = pd.DataFrame(data_R).T
        data.columns = data_R.colnames
        data = data[(data > -1e-4).all(axis=1)]
        for feat in time_dep_feat:
            data[feat] = np.log(data[feat].values)
        id_list = np.unique(data["id"])
        n_samples = len(id_list)
        n_long_features = len(time_dep_feat)
        data_lights = data.drop_duplicates(subset=["id"])

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
            tmp = data[(data["id"] == id_list[i]) & (data["T_long"] < t_max[i])]
            if tmp.empty:
                t_max[i] = data[(data["id"] == id_list[i])]["T_long"].values[0]
                n_i = 1
            else:
                n_i = tmp.shape[0]
            data = data[(data["id"] != id_list[i]) |
                    ((data["id"] == id_list[i]) & (data["T_long"] <= t_max[i]))]
            y_i = []
            for l in range(n_long_features):
                Y_il = data[["T_long", time_dep_feat[l]]][
                    (data["id"] == id_list[i]) & (data['T_long'] <= t_max[i])]
                # TODO: Add value of 1/365 (the first day of survey instead of 0)
                y_il = Y_il[time_dep_feat[l]].values
                t_il = Y_il["T_long"].values + 1/365
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
        data["T_max"] = np.array(t_max_R).flatten()
    return (data, data_lights, Y_rep, time_dep_feat, time_indep_feat)

def extract_lights_feat(data, time_indep_feat, time_dep_feat):
    X = np.float_(data[time_indep_feat].values)
    Y = data[time_dep_feat]
    T = np.float_(data[["T_survival"]].values.flatten())
    delta = np.int_(data[["delta"]].values.flatten())

    return (X, Y, T, delta)

def extract_R_feat(data):
    data_id = data.drop_duplicates(subset=["id"])
    T = data_id[["T_survival"]].values.flatten()
    delta = data_id[["delta"]].values.flatten()
    with robjects.conversion.localconverter(robjects.default_converter +
                                            pandas2ri.converter +
                                            numpy2ri.converter):
        data_R = robjects.conversion.py2rpy(data)
        T_R = robjects.conversion.py2rpy(T)
        delta_R = robjects.conversion.py2rpy(delta)

    return (data_R, T_R, delta_R)
