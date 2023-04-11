import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
#os.environ['R_HOME'] = "/usr/lib/R"
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
pandas2ri.activate()
import numpy as np
from lights.inference import prox_QNEM
from lights.base.base import normalize
from prettytable import PrettyTable
from lifelines.utils import concordance_index as c_index_score
from time import time
from lights.simulation import SimuJointLongitudinalSurvival
import pandas as pd
from scipy.stats import beta
from sklearn.preprocessing import OneHotEncoder
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.metrics import brier_score
from lifelines import KaplanMeierFitter
def load_data(data_name):
    if data_name == "simu":
        n_long_features = 5
        n_time_indep_features = 10
        n_samples = 500
        gap = .8
        sparsity=0.5
        simu = SimuJointLongitudinalSurvival(seed=123, n_long_features=n_long_features,
                                     n_samples=n_samples ,n_time_indep_features=n_time_indep_features,
                                     sparsity=sparsity, grid_time=True,
                                     fixed_effect_mean_low_risk = (-.6, .1),
                                     fixed_effect_mean_high_risk = (.05, .2),
                                     shape = .1, scale=.05, coeff_val_asso_high_risk = .4,
                                     coeff_val_asso_low_risk = .1, cov_corr_long = .01,
                                     coeff_val_time_indep = .2, gap = gap)
        X, Y, T, delta = simu.simulate()
        time_dep_feat = ['time_dep_feat%s' % (l + 1)
                         for l in range(n_long_features)]
        time_indep_feat = ['X_%s' % (l + 1)
                           for l in range(n_time_indep_features)]
        id_list = np.unique(Y.id)
        df_ = pd.DataFrame(data=np.column_stack((id_list, T, delta, X)),
                               columns=["id", "T_survival", "delta"] + time_indep_feat)
        data = pd.merge(Y, df_, on="id")
    elif data_name == "PBC":
        # load PBC Seq
        robjects.r.source(os.getcwd() + "/competing_methods/load_PBC_Seq.R")
        time_indep_feat = ['drug', 'age', 'sex']
        time_dep_feat = ['serBilir', 'albumin', 'SGOT', 'platelets',
                         'prothrombin', 'alkaline', 'serChol']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        data = data[(data[time_dep_feat] > -1e-4).all(axis=1)]
        for feat in time_dep_feat:
            data[feat] = np.log(data[feat].values)

        #creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on time-indep columns 
        data['drug'] = encoder.fit_transform(data[['drug']]).toarray()
        data['sex'] = encoder.fit_transform(data[['sex']]).toarray()
        data['id'] = data['id'].astype(int)
    elif data_name == "aids":
        # load aids
        robjects.r.source(os.getcwd() + "/competing_methods/load_aids.R")
        time_indep_feat = ['drug', 'gender', 'prevOI', 'AZT']
        time_dep_feat = ['CD4']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        #data = data[(data[time_dep_feat] > -1e-4).all(axis=1)]
        #for feat in time_dep_feat:
        #    data[feat] = np.log(data[feat].values)

        #creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on time-indep colums
        data['drug'] = encoder.fit_transform(data[['drug']]).toarray()
        data['gender'] = encoder.fit_transform(data[['gender']]).toarray()
        data['prevOI'] = encoder.fit_transform(data[['prevOI']]).toarray()
        data['AZT'] = encoder.fit_transform(data[['AZT']]).toarray()
        data['id'] = data['id'].astype(int)
    elif data_name == "data_lcmm":
        # load data_lcmm
        robjects.r.source(os.getcwd() + "/competing_methods/load_data_lcmm.R")
        time_indep_feat = ['X1', 'X2', 'X3', 'X4']
        time_dep_feat = ['Ydep1', 'Ydep2', 'Ydep3']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        data['id'] = data['id'].astype(int)
    elif data_name == "liver":
        # load epileptic
        encoder = OneHotEncoder(handle_unknown='ignore')
        robjects.r.source(os.getcwd() + "/competing_methods/load_liver.R")
        time_indep_feat = ['treatment']
        time_dep_feat = ['prothrombin']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        data['treatment'] = encoder.fit_transform(data[['treatment']]).toarray()
        data['id'] = data['id'].astype(int)    
    elif data_name == "mental":
        # load epileptic
        encoder = OneHotEncoder(handle_unknown='ignore')
        robjects.r.source(os.getcwd() + "/competing_methods/load_mental.R")
        time_indep_feat = ['treat']
        time_dep_feat = ['longitudinal']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        data['treat'] = encoder.fit_transform(data[['treat']]).toarray()
        data['id'] = data['id'].astype(int)
        
        longitudinal_columns = ["Y.t8", "Y.t6", "Y.t4", "Y.t2", "Y.t1", "Y.t0"]
        longitudinal_time_columns = [8, 6, 4, 2, 1, 0]
        Y = pd.DataFrame(columns=["id", "T_long", "longitudinal"])
        id_list = data.id.values
        T = data.T_survival.values
        delta = data.delta.values
        X = data.treat.values
        for idx in id_list:
            for col_idx in range(len(longitudinal_columns)):
                if data[data.id == idx][longitudinal_columns[col_idx]].values != -2147483648:
                    time = longitudinal_time_columns[col_idx:]
                    long_val = data[data.id == idx][longitudinal_columns[col_idx:]].values.flatten()
                    tmp_df = pd.DataFrame(np.array([[int(idx)] * len(time), time, long_val.astype(float)]).T, 
                                          columns=["id", "T_long", "longitudinal"])
                    Y = Y.append(tmp_df)
                    break
                    
        df_ = pd.DataFrame(data=np.column_stack((id_list, T, delta, X)),
                               columns=["id", "T_survival", "delta"] + time_indep_feat)
        data = pd.merge(Y, df_, on="id")
    elif data_name == "renal":
        # load epileptic
        encoder = OneHotEncoder(handle_unknown='ignore')
        robjects.r.source(os.getcwd() + "/competing_methods/load_renal.R")
        time_indep_feat = ["weight", "age", "gender"]
        time_dep_feat = ['gfr']
        data_R = robjects.r["load"]()
        data =  robjects.conversion.rpy2py(data_R)
        data['gender'] = encoder.fit_transform(data[['gender']]).toarray()
        data['id'] = data['id'].astype(int) 
        
        id_list = np.unique(data["id"])
        n_samples = len(id_list)
        for i in range(n_samples):
            data_i = data[(data["id"] == id_list[i])]
            n_i = data_i.shape[0]
            times_i = data_i["T_long"].values
            if n_i > 10:
                times_i = np.sort(np.random.choice(times_i, size=15,
                                             replace=False))
                n_i = 10           
            data = data[(data["id"] != id_list[i]) |
                    ((data["id"] == id_list[i]) & (data["T_long"].isin(list(times_i))))]
    else: 
        raise ValueError('Data name is not defined')
    
    return (data, time_dep_feat, time_indep_feat)

def truncate_data(data):
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

def extract_lights_feat(data, time_indep_feat, time_dep_feat):
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

def all_model(n_runs = 1, simu=True):
    seed = 0
    test_size = .2
    data, time_dep_feat, time_indep_feat = load_data(simu)

    t = PrettyTable(['Algos', 'C_index', 'time'])
    for i in range(n_runs):
        seed += 1
        nb_test_sample = int(test_size * len(data_index))
        np.random.seed(seed)
        data_index = np.unique(data.id.values)
        nb_test_sample = int(test_size * len(data_index))
        test_index = np.random.choice(data_index, size=nb_test_sample, replace=False)
        data_train = data[data.id.isin(test_index)]
        data_test = data[data.id.isin(test_index)]
        
        X_lights_train, Y_lights_train, T_train, delta_train = \
            extract_lights_feat(data_train, time_indep_feat, time_dep_feat)
        X_lights_test, Y_lights_test, T_test, delta_test = \
            extract_lights_feat(data_test, time_indep_feat, time_dep_feat)
        
        data_R_train, T_R_train, delta_R_train = extract_R_feat(data_train)
        data_R_test, T_R_test, delta_R_test = extract_R_feat(data_test)

        # The penalized Cox model.
        robjects.r.source(os.getcwd() + "/competing_methods/CoxNet.R")
        X_R_train = robjects.r["Cox_get_long_feat"](data_R_train, time_dep_feat,
                                                  time_indep_feat)
        X_R_test = robjects.r["Cox_get_long_feat"](data_R_test, time_dep_feat,
                                                 time_indep_feat)
        best_lambda = robjects.r["Cox_cross_val"](X_R_train, T_R_train, delta_R_train)
        start = time()
        trained_CoxPH = robjects.r["Cox_fit"](X_R_train, T_R_train,
                                              delta_R_train, best_lambda)
        Cox_pred = robjects.r["Cox_score"](trained_CoxPH, X_R_test)
        Cox_marker = np.array(Cox_pred[:])
        Cox_c_index = c_index_score(T_test, Cox_marker, delta_test)
        Cox_c_index = max(Cox_c_index, 1 - Cox_c_index)
        t.add_row(["Cox", "%g" % Cox_c_index, "%.3f" % (time() - start)])

        # Multivariate joint latent class model.
        start = time()
        robjects.r.source(os.getcwd() + "/competing_methods/MJLCMM.R")
        trained_long_model, trained_mjlcmm = robjects.r["MJLCMM_fit"](data_R_train,
                                             robjects.StrVector(time_dep_feat),
                                             robjects.StrVector(time_indep_feat))
        MJLCMM_pred = robjects.r["MJLCMM_score"](trained_long_model,
                                                 trained_mjlcmm,
                                                 time_indep_feat, data_R_test)
        MJLCMM_marker = np.array(MJLCMM_pred.rx2('pprob')[2])
        MJLCMM_c_index = c_index_score(T_test, MJLCMM_marker, delta_test)
        MJLCMM_c_index = max(MJLCMM_c_index, 1 - MJLCMM_c_index)
        t.add_row(["MJLCMM", "%g" % MJLCMM_c_index, "%.3f" % (time() - start)])

        # Multivariate shared random effect model.
        start = time()
        robjects.r.source(os.getcwd() + "/competing_methods/JMBayes.R")
        trained_JMBayes = robjects.r["fit"](data_R_train,
                                            robjects.StrVector(time_dep_feat),
                                            robjects.StrVector(time_indep_feat))
        JMBayes_marker = np.array(robjects.r["score"](trained_JMBayes, data_R_test))
        JMBayes_c_index = c_index_score(T_test, JMBayes_marker, delta_test)
        JMBayes_c_index = max(JMBayes_c_index, 1 - JMBayes_c_index)
        t.add_row(["JMBayes", "%g" % JMBayes_c_index, "%.3f" % (time() - start)])

        # lights
        start = time()
        fixed_effect_time_order = 1
        learner = prox_QNMCEM(fixed_effect_time_order=fixed_effect_time_order,
                              max_iter=5, initialize=True, print_every=1,
                              compute_obj=True, simu=simu,
                              asso_functions=["lp", "re"],
                              l_pen_SGL=0.02, eta_sp_gp_l1=.9, l_pen_EN=0.02)
        learner.fit(X_lights_train, Y_lights_train, T_train, delta_train)
        prediction_times = data_lights_test[["T_max"]].values.flatten()
        lights_marker = learner.predict_marker(X_lights_test, Y_lights_test, prediction_times)
        lights_c_index = c_index_score(T_test, lights_marker, delta_test)
        lights_c_index = max(lights_c_index, 1 - lights_c_index)
        t.add_row(["lights", "%g" % lights_c_index, "%.3f" % (time() - start)])

        print(t)

def brier_score_customize(survival_train, survival_test, estimate, times):
    test_time, test_event = survival_test['time'], survival_test['indicator']
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = np.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the brier scores at each time point
    brier_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = np.mean(
            np.square(est) * is_case.astype(int) / prob_cens_y
            + np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]
        )

    return times, brier_scores


def compute_brier_score(surv_train, surv_test, marker, times):
    T_train, delta_train = surv_train['time'], surv_train['indicator']
    T_test, delta_test = surv_test['time'], surv_test['indicator']
    gr1 = marker > np.mean(marker)
    surv1 = KaplanMeierFitter()
    surv1.fit(T_test[gr1], delta_test[gr1])
    surv2 = KaplanMeierFitter()
    surv2.fit(T_test[~gr1], delta_test[~gr1])
    score = 0.0
    for i in range(len(times)):
        if gr1[i]:
            est_i = marker[i] * surv1.survival_function_at_times(times[i]) 
            + (1 - marker[i]) * surv2.survival_function_at_times(times[i])
        else:
            est_i = (1 - marker[i]) * surv1.survival_function_at_times(times[i]) 
            + marker[i] * surv2.survival_function_at_times(times[i])
        score += brier_score_customize(surv_train, surv_test, np.array([est_i]), times[i:i+1])[1]
    score = score / len(times)
    
    return score