import numpy as np
from sklearn.model_selection import KFold
from flash.inference import ext_EM
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time
from scipy.stats import beta
import pandas as pd
from competing_methods.all_model import load_data, extract_flash_feat, truncate_data
from flash.base.base import normalize

def cross_validate(data, time_indep_feat, time_dep_feat, fc_parameters, l_pen_EN_list, l_pen_SGL_list,
                   n_folds=10, shuffle=True):
    """Apply n_folds randomized search cross-validation using the given
    data, to select the best penalization hyper-parameters

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    Y : `pandas.DataFrame`
        The longitudinal data.

    T : `np.ndarray`, shape=(n_samples,)
        Censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        Censoring indicator

    l_pen_EN_list : `list`
        List of level of penalization for the ElasticNet

    l_pen_SGL_list : `list`
        List of level of penalization for the Sparse Group Lasso

    n_folds : `int`, default=10
        Number of folds. Must be at least 2.

    shuffle : `bool`, default=True
        Whether to shuffle the data before splitting into batches

    """
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    X, _, _, _ = extract_flash_feat(data, time_indep_feat, time_dep_feat)
    def learners(params):
        scores = []
        for n_fold, (idx_train, idx_test) in enumerate(cv.split(X)):
            learner = ext_EM(fc_parameters= fc_parameters, verbose=False,
                             print_every=1, max_iter=40)
            id_test = np.unique(data["id"])[idx_test]
            data_train = data[~data.id.isin(id_test)]
            data_test = data[data.id.isin(id_test)]
            data_test_truncated = truncate_data(data_test)
            X_train, Y_train, T_train, delta_train = extract_flash_feat(
                                                                data_train,
                                                                time_indep_feat,
                                                                time_dep_feat)
            X_test, Y_test, T_test, delta_test = extract_flash_feat(
                                                                data_test_truncated,
                                                                time_indep_feat,
                                                                time_dep_feat)
            X_train = normalize(X_train)
            Y_train[time_dep_feat] = normalize(Y_train[time_dep_feat].values)
            X_test = normalize(X_test)
            Y_test[time_dep_feat] = normalize(Y_test[time_dep_feat].values)

            learner.l_pen_EN = params['l_pen_EN']
            learner.l_pen_SGL = params['l_pen_SGL']
            try:
                learner.fit(X_train, Y_train, T_train, delta_train)
                score = compute_Cindex(learner, X_test, Y_test, T_test, delta_test)
            except (ValueError, np.linalg.LinAlgError) as e:
                scores = np.nan
                break
            else:
                scores.append(score)
        return {'loss': -np.mean(scores), 'status': STATUS_OK}

    params = {}
    trials = pd.DataFrame(columns=["l_pen_EN", "l_pen_SGL", "loss"])
    for log_l_pen_EN in l_pen_EN_list:
        for log_l_pen_SGL in l_pen_SGL_list:
            params['l_pen_EN'] = log_l_pen_EN
            params['l_pen_SGL'] = log_l_pen_SGL
            loss = learners(params)["loss"]
            new_row = {"l_pen_EN" : params['l_pen_EN'],
                       "l_pen_SGL" : params['l_pen_SGL'],
                       "loss" : loss}
            trials = trials.append(new_row, ignore_index=True)

    best = trials[trials.loss == trials.loss.min()].squeeze()

    return best, trials

def truncate_features(Y, t_max):
    id_list = list(np.unique(Y['id'].values))
    n_samples = len(id_list)
    for i in range(n_samples):
        times_i = Y[(Y["id"] == id_list[i])].T_long.values
        if all(times_i > t_max[i]):
            t_max[i] = times_i[0]
        Y = Y[(Y["id"] != id_list[i]) | ((Y["id"] == id_list[i])
                                         & (Y["T_long"] <= t_max[i]))]
    return Y

def compute_Cindex(learner, X, Y, T, delta):
    n_samples = X.shape[0]
    t_max = np.multiply(T, 1 - beta.rvs(2, 5, size=n_samples))
    Y_= truncate_features(Y.copy(), t_max)
    score = learner.score(X, Y_, T, delta)

    return score
def risk_prediction(model="lights", n_run=2, simulation=False, test_size=.3):
    seeds = [1, 123]
    running_time = []
    score = []
    for idx in range(n_run):
        start_time = time()
        data, time_indep_feat, time_dep_feat = load_data(simu=simulation, seed=seeds[idx])
        id = np.unique(data.id.values)
        nb_test_sample = int(test_size * len(id))
        id_test = np.random.choice(id, size=nb_test_sample, replace=False)
        data_train = data[~data.id.isin(id_test)]
        data_test = data[data.id.isin(id_test)]
        X_lights_train, Y_lights_train, T_train, delta_train = \
            extract_lights_feat(data, time_indep_feat, time_dep_feat)
        X_lights_test, Y_lights_test, T_test, delta_test = \
            extract_lights_feat(data, time_indep_feat, time_dep_feat)

        data_train = data[~data.id.isin(id_test)]
        data_test = data[data.id.isin(id_test)]
        data_R_train, T_R_train, delta_R_train = extract_R_feat(data_train)
        data_R_test, T_R_test, delta_R_test = extract_R_feat(data_test)

        # cross_validation of lights
        if model == "lights":
            fc_parameters = {
                    "mean": None,
                    "median": None,
                    "quantile": [{"q": 0.25}, {"q": 0.75}],
                    "standard_deviation": None,
                    "skewness": None,
            }
            fixed_effect_time_order = 1
            zeta_gamma_max = 1
            n_folds = 5
            max_eval = 50
            best_param, trials = cross_validate(X_lights_train, Y_lights_train,
                                                T_train, delta_train,
                                                fc_parameters,
                                                fixed_effect_time_order,
                                                zeta_gamma_max=zeta_gamma_max,
                                                n_folds=n_folds,
                                                max_eval=max_eval)
            l_pen_EN, l_pen_SGL = best_param.values()
            learner = ext_EM(fixed_effect_time_order=fixed_effect_time_order,
                                max_iter=5, initialize=True, print_every=1,
                                l_pen_SGL=l_pen_SGL, eta_sp_gp_l1=.9, l_pen_EN=l_pen_EN,
                                fc_parameters=fc_parameters)
            learner.fit(X_lights_train, Y_lights_train, T_train, delta_train)

            c_index = compute_Cindex(learner, X_lights_test, Y_lights_test,
                                        T_test, delta_test)
        exe_time = time() - start_time
        running_time.append(exe_time)
        score.append(max(c_index, 1 - c_index))
        # seed += 1

    return score, running_time