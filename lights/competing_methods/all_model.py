import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
from rpy2 import robjects
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from lights.inference import prox_QNMCEM
from prettytable import PrettyTable
from lifelines.utils import concordance_index as c_index_score
from time import time

def all_model(models=None, data=None, simu=False):

    #initialization
    t = PrettyTable(['Algos', 'C_index', 'time'])
    robjects.r.source("preprocessing")
    models = ["Cox", "lcmm", "JMBayes", "Lights"]
    result = list()
    time_dep_feat = ["serBilir", "SGOT", "albumin"]
    time_indep_feat = ["age", "drug", "sex"]
    data_train, data_test, data_train_id, data_test_id, tmax = \
        robjects.r["preprocessing"](simu)
    T_train = data_train_id.rx2('T_survival')
    delta_train = data_train_id.rx2('delta')
    T_test = data_test_id.rx2('T_survival')
    delta_test = data_test_id.rx2('delta')

    # The penalized time-dependent Cox model.
    robjects.r.source("CoxNet")
    X_train = robjects.r["Cox_get_long_feat"](data_train, time_dep_feat,
                                              time_indep_feat, alpha=2)
    X_test = robjects.r["Cox_get_long_feat"](data_test, time_dep_feat,
                                             time_indep_feat, alpha=2)
    best_lambda = robjects.r["Cox_cross_val"](X_train, T_train, delta_train)
    start = time()
    trained_CoxPH = robjects.r["Cox_fit"](X_train, T_train,
                                          delta_train, best_lambda)
    Cox_pred = robjects.r["Cox_score"](trained_CoxPH, X_test)
    Cox_marker = np.array(Cox_pred[:])
    Cox_c_index = c_index_score(T_test, Cox_marker, delta_test)
    Cox_c_index = max(Cox_c_index, 1 - Cox_c_index)
    t.add_row(["Cox", "%g" % Cox_c_index, "%.3f" %(time() - start)])

    # Multivariate joint latent class model.
    start = time()
    robjects.r.source("MJLCMM")
    trained_long_model, trained_mjlcmm = robjects.r["MJLCMM_fit"](data_train,
                                         robjects.StrVector(time_dep_feat),
                                         robjects.StrVector(time_indep_feat),
                                         alpha=2)
    MJLCMM_pred = robjects.r["MJLCMM_score"](trained_long_model,
                                             trained_mjlcmm,
                                             time_indep_feat, data_test)
    MJLCMM_marker = np.array(MJLCMM_pred.rx2('pprob')[2])
    MJLCMM_c_index = c_index_score(T_test, MJLCMM_marker, delta_test)
    MJLCMM_c_index = max(MJLCMM_c_index, 1 - MJLCMM_c_index)
    t.add_row(["MJLCMM", "%g" % MJLCMM_c_index, "%.3f" %(time() - start)])

    # Multivariate shared random effect model.
    start = time()
    robjects.r.source("JMBayes")
    trained_JMBayes = robjects.r["fit"](data_train, data_train_id,
                                        robjects.StrVector(time_dep_feat),
                                        robjects.StrVector(time_indep_feat))
    JMBayes_pred = robjects.r["score"](trained_JMBayes, data_test,
                                       data_test_id, tmax)
    JMBayes_marker = np.array(JMBayes_pred.rx2('full.results')[0])
    JMBayes_c_index = c_index_score(T_test, JMBayes_marker, delta_test)
    JMBayes_c_index = max(JMBayes_c_index, 1 - JMBayes_c_index)
    t.add_row(["JMBayes", JMBayes_c_index, time() - start])

    # Lights
    # data preprocessing
    # survival training data
    n_long_features = len(time_dep_feat)
    survival_data_training = pd.DataFrame(data_train_id).T
    survival_data_training.columns = data_train_id.colnames
    n_training_samples = survival_data_training.shape[0]
    X_train = survival_data_training[time_indep_feat]
    labelencoder = LabelEncoder()
    X_train["drug"] = labelencoder.fit_transform(X_train["drug"])
    X_train["sex"] = labelencoder.fit_transform(X_train["sex"])
    X_train = X_train.values
    X_train[:, 1] = StandardScaler().fit_transform(
        X_train[:, 1].reshape(-1, 1)).flatten()
    T_train = survival_data_training[["T_survival"]].values.flatten()
    delta_train = survival_data_training[["delta"]].values.flatten()

    # longitunal training data
    data_training_DF = pd.DataFrame(data_train).T
    data_training_DF.columns = data_train.colnames
    Y_train_ = data_training_DF[["id", "T_long"] + time_dep_feat]
    Y_train_[time_dep_feat] = StandardScaler().fit_transform(
        Y_train_[time_dep_feat])
    Y_train = pd.DataFrame(columns=time_dep_feat)
    id_list = np.unique(Y_train_["id"])
    for i in range(n_training_samples):
        y_i = []
        for l in range(n_long_features):
            Y_il = Y_train_[["T_long", time_dep_feat[l]]][
                Y_train_["id"] == id_list[i]]
            # TODO: Add value of 1/365 (the first day of survey instead of 0)
            y_i += [pd.Series(Y_il[time_dep_feat[l]].values,
                              index=Y_il["T_long"].values + 1 / 365)]
        Y_train.loc[i] = y_i

    # survival testing data
    survival_data_testing = pd.DataFrame(data_test_id).T
    survival_data_testing.columns = data_test_id.colnames
    n_testing_samples = survival_data_testing.shape[0]
    X_test = survival_data_testing[time_indep_feat]
    labelencoder = LabelEncoder()
    X_test["drug"] = labelencoder.fit_transform(X_test["drug"])
    X_test["sex"] = labelencoder.fit_transform(X_test["sex"])
    X_test = X_test.values
    X_test[:, 1] = StandardScaler().fit_transform(
        X_test[:, 1].reshape(-1, 1)).flatten()
    T_test = survival_data_testing[["T_survival"]].values.flatten()
    delta_test = survival_data_testing[["delta"]].values.flatten()

    # longitunal testing data
    data_testing_DF = pd.DataFrame(data_test).T
    data_testing_DF.columns = data_test.colnames
    Y_test_ = data_testing_DF[["id", "T_long"] + time_dep_feat]
    Y_test_[time_dep_feat] = StandardScaler().fit_transform(
        Y_test_[time_dep_feat])
    Y_test = pd.DataFrame(columns=time_dep_feat)
    id_list = np.unique(Y_test_["id"])
    for i in range(n_testing_samples):
        y_i = []
        for l in range(n_long_features):
            Y_il = Y_test_[["T_long", time_dep_feat[l]]][
                Y_test_["id"] == id_list[i]]
            y_i += [pd.Series(Y_il[time_dep_feat[l]].values,
                              index=Y_il["T_long"].values + 1 / 365)]
        Y_test.loc[i] = y_i

    # declare learner here
    start = time()
    fixed_effect_time_order = 1
    learner = prox_QNMCEM(fixed_effect_time_order=fixed_effect_time_order,
                          max_iter=10, initialize=True, print_every=1,
                          compute_obj=True, simu=False,
                          asso_functions=["lp", "re"],
                          l_pen_SGL=0.02, eta_sp_gp_l1=.9, l_pen_EN=0.02)
    learner.fit(X_train, Y_train, T_train, delta_train)

    lights_marker = learner.predict_marker(X_test, Y_test)
    lights_c_index = c_index_score(T_test, lights_marker, delta_test)
    lights_c_index = max(lights_c_index, 1 - lights_c_index)
    t.add_row(["lights", lights_c_index, time() - start])

    print(t)

all_model()





