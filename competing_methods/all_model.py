import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
pandas2ri.activate()
import numpy as np
from flash.base.base import normalize
from flash.simulation import SimuJointLongitudinalSurvival
import pandas as pd
from scipy.stats import beta
from sklearn.preprocessing import OneHotEncoder
def load_data(data_name):
    if data_name == "FLASH_simu":
        n_long_features = 5
        n_time_indep_features = 10
        n_samples = 500
        simu = SimuJointLongitudinalSurvival(seed=123, n_long_features=n_long_features,
                                     n_samples=n_samples ,n_time_indep_features=n_time_indep_features)
        X, Y, T, delta = simu.simulate()
        time_dep_feat = ['time_dep_feat%s' % (l + 1)
                         for l in range(n_long_features)]
        time_indep_feat = ['X_%s' % (l + 1)
                           for l in range(n_time_indep_features)]
        id_list = np.unique(Y.id)
        df_ = pd.DataFrame(data=np.column_stack((id_list, T, delta, X)),
                               columns=["id", "T_survival", "delta"] + time_indep_feat)
        data = pd.merge(Y, df_, on="id")
    elif data_name == "FLASH_simu_probit":
        n_long_features = 5
        n_time_indep_features = 10
        n_samples = 500
        simu = SimuJointLongitudinalSurvival(seed=123, n_long_features=n_long_features,
                                             n_samples=n_samples,
                                             n_time_indep_features=n_time_indep_features,
                                             probit=True)
        X, Y, T, delta = simu.simulate()
        time_dep_feat = ['time_dep_feat%s' % (l + 1)
                         for l in range(n_long_features)]
        time_indep_feat = ['X_%s' % (l + 1)
                           for l in range(n_time_indep_features)]
        id_list = np.unique(Y.id)
        df_ = pd.DataFrame(data=np.column_stack((id_list, T, delta, X)),
                               columns=["id", "T_survival", "delta"] + time_indep_feat)
        data = pd.merge(Y, df_, on="id")
    elif data_name == "PBCseq":
        # load PBC Seq
        robjects.r.source(os.getcwd() + "/competing_methods/load_PBC_Seq.R")
        time_indep_feat = ['drug', 'age', 'sex']
        time_dep_feat = ['serBilir', 'albumin', 'SGOT', 'platelets',
                         'prothrombin', 'alkaline', 'serChol']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        # remove outliers
        data = data[(data[time_dep_feat] > -1e-4).all(axis=1)]

        #creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on time-indep columns 
        data['drug'] = encoder.fit_transform(data[['drug']]).toarray()
        data['sex'] = encoder.fit_transform(data[['sex']]).toarray()
        data['id'] = data['id'].astype(int)
                
        
    elif data_name == "Aids":
        # load aids
        robjects.r.source(os.getcwd() + "/competing_methods/load_aids.R")
        time_indep_feat = ['drug', 'gender', 'prevOI', 'AZT']
        time_dep_feat = ['CD4']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)

        #creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on time-indep colums
        data['drug'] = encoder.fit_transform(data[['drug']]).toarray()
        data['gender'] = encoder.fit_transform(data[['gender']]).toarray()
        data['prevOI'] = encoder.fit_transform(data[['prevOI']]).toarray()
        data['AZT'] = encoder.fit_transform(data[['AZT']]).toarray()
        data['id'] = data['id'].astype(int)
    elif data_name == "joineRML_simu":
        encoder = OneHotEncoder(handle_unknown='ignore')
        robjects.r.source(os.getcwd() + "/competing_methods/load_simulated_joineRML.R")
        time_indep_feat = ["ctsxl", "binxl"]
        time_dep_feat = ['Y.1', 'Y.2']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        data['id'] = data['id'].astype(int)
    else: 
        raise ValueError('Data name is not defined')
    
    return (data, time_dep_feat, time_indep_feat)

def truncate_data(data):
    id_list = np.unique(data["id"])
    n_samples = len(id_list)

    for i in range(n_samples):
        T_long_id = data[(data["id"] == id_list[i])]["T_long"].values.reshape(-1, 1)
        data.loc[data["id"] == id_list[i], ['T_long']] = T_long_id - min(T_long_id)
        #if len(data[(data["id"] == id_list[i])]["T_long"].values) < 2:
        #    data = data[(data["id"] != id_list[i])]
    
    data_truncated = data.copy(deep=True)
    id_list = np.unique(data["id"])
    n_samples = len(id_list)

    # generate t_max
    a = 2
    b = 5
    np.random.seed(0)
    r = beta.rvs(a, b, size=n_samples)
    T = data_truncated.drop_duplicates(subset=["id"])["T_survival"].values
    t_max = T * (1 - r)

    t_max_R = []
    for i in range(n_samples):
        tmp = data_truncated[(data_truncated["id"] == id_list[i]) & (data_truncated["T_long"] < t_max[i])]
        if tmp.empty:
            t_max[i] = data_truncated[(data["id"] == id_list[i])]["T_long"].values[0] + .01
            n_i = data_truncated[(data_truncated["id"] == id_list[i]) & (data_truncated["T_long"] < t_max[i])].shape[0]
            data_truncated = data_truncated[(data_truncated["id"] != id_list[i]) |
                ((data_truncated["id"] == id_list[i]) & (data_truncated["T_long"] <= t_max[i]))]
        else:
            n_i = tmp.shape[0]
            times_i = tmp["T_long"].values
            if n_i > 10:
                times_i = np.sort(np.random.choice(tmp["T_long"].values, size=10,
                                             replace=False))
                n_i = 10           
            data_truncated = data_truncated[(data_truncated["id"] != id_list[i]) |
                    ((data_truncated["id"] == id_list[i]) & (data_truncated["T_long"].isin(list(times_i))))]
        t_max_R += [t_max[i]] * n_i
    data_truncated["T_max"] = np.array(t_max_R).flatten()

    return (data_truncated)

def extract_flash_feat(data, time_indep_feat, time_dep_feat):
    data_id = data.drop_duplicates(subset=["id"])
    X = np.float_(data_id[time_indep_feat].values)
    Y = data[["id", "T_long"] + time_dep_feat]
    T = np.float_(data_id[["T_survival"]].values.flatten())
    X = normalize(X)
    Y[time_dep_feat] = normalize(Y[time_dep_feat].values)
    delta = np.int_(data_id[["delta"]].values.flatten())

    return (X, Y, T, delta)

def extract_R_feat(data, time_indep_feat, time_dep_feat):
    data_id = data.drop_duplicates(subset=["id"])
    T = data_id[["T_survival"]].values.flatten()
    delta = data_id[["delta"]].values.flatten()
    data[time_indep_feat] = normalize(data[time_indep_feat].values)
    data[time_dep_feat] = normalize(data[time_dep_feat].values)
    with robjects.conversion.localconverter(robjects.default_converter +
                                            pandas2ri.converter +
                                            numpy2ri.converter):
        data_R = robjects.conversion.py2rpy(data)

    return (data_R, T, delta)