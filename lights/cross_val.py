import numpy as np
from sklearn.model_selection import KFold
from lights.inference import prox_QNMCEM
import itertools
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def cross_validate(X, Y, T, delta, S_k, simu=True, n_folds=10,
                   adaptative_grid_el=True, grid_size=30, shuffle=True,
                   verbose=True, metric='C-index', tol=1e-5, warm_start=True,
                   eta_elastic_net=.1, eta_sp_gp_l1=.1,
                   zeta_gamma_max = None, zeta_xi_max = None,
                   max_iter=100, max_iter_lbfgs=50, max_iter_proxg=50,
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

    grid_size : `int`, default=30
        Grid size if adaptative_grid_el=`True`

    shuffle : `bool`, default=True
        Whether to shuffle the data before splitting into batches

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    metric : 'log_lik', 'C-index', default='C-index'
        Either computes log-likelihood or C-index

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

    eta_sp_gp_l1: `float`, default=0.1
        The Sparse Group L1 mixing parameter, with 0 <= eta_sp_gp_l1 <= 1
        For eta_sp_gp_l1 = 1 this is Group L1

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
            learner = prox_QNMCEM(verbose=False, tol=tol,
                                   warm_start=warm_start, simu=simu,
                                   fixed_effect_time_order=1, max_iter=max_iter,
                                   max_iter_lbfgs=max_iter_lbfgs,
                                   max_iter_proxg=max_iter_proxg, S_k=S_k)
            X_train, X_test = X[idx_train], X[idx_test]
            T_train, T_test = T[idx_train], T[idx_test]
            Y_train, Y_test = Y.iloc[idx_train, :], Y.iloc[idx_test, :]
            delta_train, delta_test = delta[idx_train], delta[idx_test]
            learner.l_pen_EN = params['l_pen_EN']
            learner.l_pen_SGL = params['l_pen_SGL']
            try:
                learner.fit(X_train, Y_train, T_train, delta_train)
            except ValueError:
                scores = np.array([1])
                break
            else:
                scores.append(learner.score(X_test, Y_test, T_test, delta_test))
        return {'loss': -np.mean(scores), 'status': STATUS_OK}

    fspace = {
        'l_pen_EN': hp.uniform('l_pen_EN', zeta_xi_max * 1e-4, zeta_xi_max),
        'l_pen_SGL': hp.uniform('l_pen_SGL', zeta_gamma_max * 1e-4, zeta_gamma_max)
    }

    trials = Trials()
    best = fmin(fn=learners, space=fspace, algo=tpe.suggest, max_evals=max_eval,
                trials=trials)

    return best
