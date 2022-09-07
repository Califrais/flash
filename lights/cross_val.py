import numpy as np
from sklearn.model_selection import KFold
from lights.inference import prox_QNEM
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time
from scipy.stats import beta
import pandas as pd
from lights.data_loader.load_data import load_data, extract_lights_feat, extract_R_feat

def cross_validate(X, Y, T, delta, fc_parameters, fixed_effect_time_order
                   , simu=True, n_folds=3, verbose=False,
                   adaptative_grid_el=True, shuffle=True, tol=1e-5,
                   warm_start=True, eta_elastic_net=.1,
                   zeta_gamma_max = None, zeta_xi_max = None,
                   max_iter=20, max_iter_lbfgs=50, max_iter_proxg=50,
                   max_eval=50):
    """Apply n_folds randomized search cross-validation using the given
    data, to select the best penalization hyper-parameters

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
        The simulated longitudinal data. Each element of the dataframe is
        a pandas.Series

    T : `np.ndarray`, shape=(n_samples,)
        Censored times of the event of interest

    S_k : `list`
        Set of nonactive group for 2 classes (will be useful in case of
        simulated data).

    simu : `bool`, defaut=True
        If `True` we do the inference with simulated data.

    delta : `np.ndarray`, shape=(n_samples,)
        Censoring indicator

    n_folds : `int`, default=10
        Number of folds. Must be at least 2.

    adaptative_grid_el : `bool`, default=True
        If `True`, adapt the ElasticNet strength parameter grid using the
        KKT conditions

    shuffle : `bool`, default=True
        Whether to shuffle the data before splitting into batches

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    warm_start : `bool`, default=True
        If true, learning will start from the last reached solution

    eta_elastic_net : `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta <= 1.
        For eta = 0 this is ridge (L2) regularization
        For eta = 1 this is lasso (L1) regularization
        For 0 < eta < 1, the regularization is a linear combination
        of L1 and L2

    zeta_gamma_max: `float`
        The interval upper bound for gamma

    zeta_xi_max: `float`
        The interval upper bound for xi

    max_iter: `int`, default=100
        Maximum number of iterations of the prox-QNMCEM algorithm

    max_iter_lbfgs: `int`, default=50
        Maximum number of iterations of the L-BFGS-B solver

    max_iter_proxg: `int`, default=50
        Maximum number of iterations of the proximal gradient solver

    max_eval: `int`, default=50
        Maximum number of trials of the Hyperopt
    """
    n_samples = T.shape[0]
    cv = KFold(n_splits=n_folds, shuffle=shuffle)

    if adaptative_grid_el:
        # from KKT conditions
        zeta_xi_max = 1. / (1. - eta_elastic_net) * (.5 / n_samples) \
                      * np.absolute(X).sum(axis=0).max()


    def learners(params):
        scores = []
        for n_fold, (idx_train, idx_test) in enumerate(cv.split(X)):
            learner = prox_QNEM(tol=tol, warm_start=warm_start, simu=simu,
                                   fixed_effect_time_order= fixed_effect_time_order,
                                   fc_parameters= fc_parameters, max_iter=max_iter,
                                   max_iter_lbfgs=max_iter_lbfgs, verbose=verbose,
                                   max_iter_proxg=max_iter_proxg, print_every=1)
            X_train, X_test = X[idx_train], X[idx_test]
            T_train, T_test = T[idx_train], T[idx_test]
            id_test = np.unique(Y.id.values)[idx_test]
            Y_test = Y[Y.id.isin(id_test)]
            Y_train = Y[~Y.id.isin(id_test)]
            delta_train, delta_test = delta[idx_train], delta[idx_test]
            learner.l_pen_EN = params['l_pen_EN']
            learner.l_pen_SGL = params['l_pen_SGL']
            try:
                learner.fit(X_train, Y_train, T_train, delta_train)
            except ValueError:
                scores = np.nan
                break
            else:
                scores.append(compute_Cindex(learner, X_test, Y_test, T_test,
                                             delta_test))
        return {'loss': -np.mean(scores), 'status': STATUS_OK}

    fspace = {
        'l_pen_EN': hp.uniform('l_pen_EN', zeta_xi_max * 1e-4, zeta_xi_max),
        'l_pen_SGL': hp.uniform('l_pen_SGL', zeta_gamma_max * 1e-4, zeta_gamma_max)
    }

    trials = Trials()
    best = fmin(fn=learners, space=fspace, algo=tpe.suggest, max_evals=max_eval,
                trials=trials)

    return best, trials

def truncate_features(Y, t_max):
    id_list = list(np.unique(Y['id'].values))
    n_samples = len(id_list)
    for i in range(n_samples):
        times_i = Y[(Y["id"] == id_list[i])].id.values
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
    seed = 100
    running_time = []
    score = []
    for idx in range(n_run):
        start_time = time()
        data, data_lights, Y_rep, time_dep_feat, time_indep_feat = load_data(simu=simulation, seed=seed)
        id_list = data_lights["id"]
        nb_test_sample = int(test_size * len(id_list))
        id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
        data_lights_train = data_lights[~data_lights.id.isin(id_test)]
        data_lights_test = data_lights[data_lights.id.isin(id_test)]
        X_lights_train, Y_lights_train, T_train, delta_train = \
            extract_lights_feat(data_lights_train, time_indep_feat,
                                time_dep_feat)
        X_lights_test, Y_lights_test, T_test, delta_test = \
            extract_lights_feat(data_lights_test, time_indep_feat,
                                time_dep_feat)

        Y_rep_train = Y_rep[~Y_rep.id.isin(id_test)]
        Y_rep_test = Y_rep[Y_rep.id.isin(id_test)]

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
            best_param, trials = cross_validate(X_lights_train, Y_lights_train, T_train,
                                                delta_train, Y_rep_train,
                                                fc_parameters,
                                                fixed_effect_time_order,
                                                zeta_gamma_max=zeta_gamma_max,
                                                n_folds=n_folds,
                                                max_eval=max_eval)
            l_pen_EN, l_pen_SGL = best_param.values()
            learner = prox_QNEM(fixed_effect_time_order=fixed_effect_time_order,
                                max_iter=5, initialize=True, print_every=1,
                                l_pen_SGL=l_pen_SGL, eta_sp_gp_l1=.9, l_pen_EN=l_pen_EN,
                                fc_parameters=fc_parameters)
            learner.fit(X_lights_train, Y_lights_train, T_train, delta_train,
                        Y_rep_train)

            c_index = compute_Cindex(learner, X_lights_test, Y_lights_test,
                                        T_test, delta_test, Y_rep_test)
        exe_time = time() - start_time
        running_time.append(exe_time)
        score.append(c_index)
        seed += 1

    return score, running_time