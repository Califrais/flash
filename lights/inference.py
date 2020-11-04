# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score
from lights.base.base import Learner, extract_features, normalize, \
    get_vect_from_ext, get_xi_from_xi_ext, logistic_grad
from lights.init.mlmm import MLMM
from lights.init.cox import initialize_asso_params
from lights.model.e_step_functions import EstepFunctions
from lights.model.m_step_functions import MstepFunctions
from lights.model.regularizations import Penalties


class QNMCEM(Learner):
    """QNMCEM Algorithm for the lights model inference

    Parameters
    ----------
    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model for the time independant
        features

    l_pen : `float`, default=0
        Level of penalization for the ElasticNet and the Sparse Group l1

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 0 this is ridge (L2) regularization
        For eta_elastic_net = 1 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2

    eta_sp_gp_l1: `float`, default=0.1
        The Sparse Group l1 mixing parameter, with 0 <= eta_sp_gp_l1 <= 1

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    warm_start : `bool`, default=True
        If true, learning will start from the last reached solution

    fixed_effect_time_order : `int`, default=5
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    asso_functions : `list` or `str`='all', default='all'
        List of association functions wanted or string 'all' to select all
        defined association functions. The available functions are :
            - 'lp' : linear predictor
            - 're' : random effects
            - 'tps' : time dependent slope
            - 'ce' : cumulative effects

    initialize : `bool`, default=True
        If `True`, we initialize the parameters using MLMM model, otherwise we
        use arbitrarily chosen fixed initialization
    """
    def __init__(self, fit_intercept=False, l_pen=0., eta_elastic_net=.1,
                 eta_sp_gp_l1=.1, max_iter=100, verbose=True, print_every=10,
                 tol=1e-5, warm_start=True, fixed_effect_time_order=5,
                 asso_functions='all', initialize=True):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.initialize = initialize
        self.l_pen = l_pen
        self.eta_elastic_net = eta_elastic_net
        self.eta_sp_gp_l1 = eta_sp_gp_l1
        self.pen = Penalties(fit_intercept, l_pen, eta_elastic_net,
                             eta_sp_gp_l1)

        # Attributes that will be instantiated afterwards
        self.n_samples = None
        self.n_time_indep_features = None
        self.n_long_features = None
        self.theta = {
            "beta_0": None,
            "beta_1": None,
            "long_cov": None,
            "phi": None,
            "xi": None,
            "baseline_hazard": None,
            "gamma_0": None,
            "gamma_1": None
        }

    @property
    def asso_functions(self):
        return self._asso_functions

    @asso_functions.setter
    def asso_functions(self, val):
        if not (val == 'all' or set(val).issubset({'lp', 're', 'tps', 'ce'})):
            raise ValueError("``asso_functions`` must be either 'all', or a "
                             "`list` in ['lp', 're', 'tps', 'ce']")
        self._asso_functions = val

    def _log_lik(self, X, Y, T, delta):
        """Computes the likelihood of the lights model

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        Returns
        -------
        output : `float`
            The log-likelihood computed on the given data
        """
        prb = 1
        # TODO
        return np.mean(np.log(prb))

    def _func_obj(self, X, Y, T, delta, xi_ext):
        """The global objective to be minimized by the QNMCEM algorithm
        (including penalization)

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        log_lik = self._log_lik(X, Y, T, delta)
        pen = self.pen.elastic_net(xi_ext)
        # TODO : add sparse group l1 penalties
        return -log_lik + pen

    def get_proba(self, X, xi_ext):
        """Probability estimates for being on the high-risk group given
        time-independent features

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the probability of the sample for being on the high-risk
            group given time-independent features
        """
        fit_intercept = self.fit_intercept
        xi_0, xi = get_xi_from_xi_ext(xi_ext, fit_intercept)
        u = xi_0 + X.dot(xi)
        return logistic_grad(u)

    def get_post_proba(self, pi_xi, Lambda_1):
        """Posterior probability estimates for being on the high-risk group
        given all observed data

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Comes from get_proba function

        Lambda_1 : `np.ndarray`, shape=(n_samples, 2)
            blabla

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the posterior probability of the sample for being on the
            high-risk group given all observed data
        """
        # TODO
        n_samples = self.n_samples
        return np.ones((n_samples, 2))

    def predict_marker(self, X, Y):
        """Marker rule of the lights model for being on the high-risk group

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the marker rule of the sample for being on the high-risk
            group
        """
        marker = None
        # TODO (only if self.fitted = True, else raise error)
        return marker

    def update_theta(self, **kwargs):
        """Update class attributes corresponding to lights model parameters
        """
        for key, value in kwargs.items():
            if key in ["beta_0", "beta_1", "gamma_0", "gamma_1"]:
                self.theta[key] = get_vect_from_ext(value)
            elif key in ["long_cov", "phi", "baseline_hazard"]:
                self.theta[key] = value
            elif key in ["xi"]:
                _, self.theta[key] = get_xi_from_xi_ext(value,
                                                        self.fit_intercept)
            else:
                raise NameError('Parameter {} has not defined'.format(key))

    def fit(self, X, Y, T, delta):
        """Fit the lights model

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator
        """
        self._start_solve()
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        warm_start = self.warm_start
        fit_intercept = self.fit_intercept
        fixed_effect_time_order = self.fixed_effect_time_order

        n_samples, n_time_indep_features = X.shape
        n_long_features = Y.shape[1]

        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        self.n_long_features = n_long_features
        q_l = fixed_effect_time_order + 1
        r_l = 2  # Linear time-varying features, so all r_l=2
        if fit_intercept:
            n_time_indep_features += 1

        if self.asso_functions == 'all':
            self.asso_functions = ['lp', 're', 'tps', 'ce']
        asso_functions = self.asso_functions
        nb_asso_param = len(asso_functions)
        if 're' in asso_functions:
            nb_asso_param += 1
        nb_asso_features = n_long_features * nb_asso_param + n_time_indep_features
        N = 5  # Number of initial Monte Carlo sample for S

        # Normalize time-independent features
        X = normalize(X)

        # Features extraction
        extracted_features = extract_features(Y, fixed_effect_time_order)

        # Initialization
        xi_ext = np.zeros(2 * n_time_indep_features)

        # The J unique censored times of the event of interest
        T_u = np.unique(T)
        J = T_u.shape[0]

        # Create indicator matrices to compare event times
        # TODO: use indicator to update f_data_given_latent
        tmp = np.broadcast_to(T, (n_samples, n_samples))
        indicator = (tmp < tmp.T) * 1 + np.eye(n_samples)
        indicator_1 = T.reshape(-1, 1) == T_u
        indicator_2 = T.reshape(-1, 1) >= T_u

        # Initialize longitudinal submodels
        if self.initialize:
            mlmm = MLMM(max_iter=max_iter, verbose=verbose,
                        print_every=print_every, tol=tol,
                        fixed_effect_time_order=fixed_effect_time_order)
            mlmm.fit(extracted_features)
            beta = mlmm.fixed_effect_coeffs
            D = mlmm.long_cov
            phi = mlmm.phi
            est = initialize_asso_params(X, T, delta)
            time_indep_cox_coeffs, baseline_hazard = est
        else:
            # Fixed initialization
            q = q_l * n_long_features
            r = r_l * n_long_features
            beta = np.zeros((q, 1))
            D = np.diag(np.ones(r))
            phi = np.ones((n_long_features, 1))
            time_indep_cox_coeffs = np.zeros(n_time_indep_features)
            baseline_hazard = pd.Series(data=np.zeros(J), index=T_u)

        gamma = np.zeros(nb_asso_features)
        gamma[:n_time_indep_features] = time_indep_cox_coeffs
        gamma_0_ext = np.concatenate((gamma, -gamma))
        gamma_0_ext[gamma_0_ext < 0] = 0
        gamma_1_ext = gamma_0_ext.copy()

        beta_0_ext = np.concatenate((beta, -beta))
        beta_0_ext[beta_0_ext < 0] = 0
        beta_1_ext = beta_0_ext.copy()

        self.update_theta(beta_0=beta_0_ext, beta_1=beta_1_ext,
                          xi=xi_ext, gamma_0=gamma_0_ext,
                          gamma_1=gamma_1_ext, long_cov=D, phi=phi,
                          baseline_hazard=baseline_hazard)
        func_obj = self._func_obj

        obj = func_obj(X, Y, T, delta, xi_ext)
        # Store init values
        self.history.update(n_iter=0, obj=obj, rel_obj=np.inf, theta=self.theta)
        if verbose:
            self.history.print_history()

        # Stopping criteria and bounds vector for the L-BGFS-B algorithms
        maxiter, pgtol = 60, 1e-5
        bounds_xi = [(0, None)] * 2 * n_time_indep_features
        bounds_beta = [(0, None)] * 2 * n_long_features * \
                      (fixed_effect_time_order + 1)
        bounds_gamma = [(0, None)] * 2 * nb_asso_features

        # Instanciates the E-step and M-step functions
        E_func = EstepFunctions(X, T, delta, extracted_features,
                                n_long_features, n_time_indep_features,
                                fixed_effect_time_order, N, asso_functions)
        F_func = MstepFunctions(fit_intercept, X, T, delta, n_long_features,
                                n_time_indep_features, self.l_pen,
                                self.eta_elastic_net, self.eta_sp_gp_l1)

        for n_iter in range(1, max_iter + 1):

            pi_xi = self.get_proba(X, xi_ext)

            # E-Step
            E_func.theta = self.theta
            S = E_func.construct_MC_samples()
            f = E_func.f_data_given_latent(S)
            Lambda_1 = E_func.Lambda_g(np.ones(shape=(n_samples, 2, 2 * N)), f)
            pi_est = self.get_post_proba(pi_xi, Lambda_1)

            g0 = g0(S)
            g0 = np.broadcast_to(g0, (n_samples, 2) + g0.shape)
            Lambda_g0 = Lambda_g(g0, f)
            E_g0 = Eg(pi_xi, Lambda_1, Lambda_g0)

            g1 = g1(X, T_u, S, theta, asso_functions, n_long_features, fixed_effect_time_order)
            g1 = np.broadcast_to(g1[..., None], g1.shape + (2,)).swapaxes(1, 4)
            Lambda_g1 = Lambda_g(g1, f).swapaxes(1, 3)
            E_g1 = Eg(pi_xi, Lambda_1, Lambda_g1)

            g2 = g2(T, S, theta, n_time_indep_features, asso_functions, n_long_features, fixed_effect_time_order).swapaxes(0, 1)
            g2 = np.broadcast_to(g2[..., None], g2.shape + (2,)).swapaxes(1, 3)
            Lambda_g2 = Lambda_g(g2, f).swapaxes(1, 2)
            E_g2 = Eg(pi_xi, Lambda_1, Lambda_g2)

            g5 = g5(T, S, theta, asso_functions, n_long_features, fixed_effect_time_order)
            g5 = np.broadcast_to(g5[..., None], g5.shape + (2,)).swapaxes(0, 5)
            Lambda_g5 = Lambda_g(g5.swapaxes(0, 2).swapaxes(1, 2),
                                       f).swapaxes(1, 4)
            E_g5 = Eg(pi_xi, Lambda_1, Lambda_g5)

            g6 = g6(X, T_u, S, theta, asso_functions, n_long_features, fixed_effect_time_order)
            g6 = np.broadcast_to(g6[..., None], g6.shape + (2,)).swapaxes(1, 6)
            Lambda_g6 = Lambda_g(g6, f).swapaxes(1, 5)
            E_g6 = Eg(pi_xi, Lambda_1, Lambda_g6)

            g8 = g8(extracted_features, S, theta, n_long_features, n_samples)
            g8 = np.broadcast_to(g8[..., None], g8.shape + (2,)).swapaxes(1, 3)
            Lambda_g8 = Lambda_g(g8, f).swapaxes(1, 2)
            E_g8 = Eg(pi_xi, Lambda_1, Lambda_g8)

            # if False: # to be defined
            #     N *= 10
            #     fctr *= .1

            # M-Step

            # Update D
            D = E_g0.sum(axis=0) / n_samples

            if warm_start:
                xi_0 = xi_ext
                beta_0_0, beta_1_0 = beta_0_ext, beta_1_ext
                gamma_0_0, gamma_1_0 = gamma_0_ext, gamma_1_ext
            else:
                xi_0 = np.zeros(2 * n_time_indep_features)
                beta_0_0 = np.zeros(2 * n_long_features *
                                    (fixed_effect_time_order + 1))
                beta_1_0 = beta_0_0.copy()
                gamma_0_0 = np.zeros(2 * nb_asso_features)
                gamma_1_0 = gamma_0_0.copy()

            # Update xi
            xi_ext = fmin_l_bfgs_b(
                func=lambda xi_ext_: F_func.P_func(X, pi_est, xi_ext_), x0=xi_0,
                fprime=lambda xi_ext_: F_func.grad_P(X, pi_est, xi_ext_),
                disp=False, bounds=bounds_xi, maxiter=maxiter, pgtol=pgtol)[0]

            # Update beta_0
            beta_0_ext = fmin_l_bfgs_b(
                func=lambda beta_ext_: F_func.R_func(beta_ext_, pi_est, E_g1, E_g2, E_g8,
                            baseline_hazard, delta, indicator_2), x0=beta_0_0,
                fprime=lambda beta_ext_: F_func.grad_R(beta_ext_), disp=False,
                bounds=bounds_beta, maxiter=maxiter, pgtol=pgtol)[0]

            # Update beta_1
            beta_1_ext = fmin_l_bfgs_b(
                func=lambda beta_ext_: F_func.R_func(beta_ext_, pi_est, E_g1, E_g2, E_g8,
                            baseline_hazard, delta, indicator_2), x0=beta_1_0,
                fprime=lambda beta_ext_: F_func.grad_R(beta_ext_), disp=False,
                bounds=bounds_beta, maxiter=maxiter, pgtol=pgtol)[0]

            self.update_theta(beta_0=beta_0_ext, beta_1=beta_1_ext)

            # g1_Q = self._g1(X, T_u, S)
            # g1_Q = np.broadcast_to(g1_Q[..., None], g1_Q.shape + (2,)).swapaxes(1, 4)
            # Lambda_g1_Q = self._Lambda_g(g1_Q, f).swapaxes(1, 3)
            # E_g1_Q = self._Eg(pi_xi, Lambda_1, Lambda_g1_Q)
            #
            # log_g1_Q = np.log(g1_Q)
            # Lambda_log_g1_Q = self._Lambda_g(log_g1_Q, f).swapaxes(1, 3)
            # E_log_g1_Q = (self._Eg(pi_xi, Lambda_1, Lambda_log_g1_Q).T * (
            #             indicator_1 * 1).T).sum(axis=1).T

            # TODO
            g1 = g1(X, T_u, S)
            E_g1 = Eg(g1)
            E_log_g1 = Eg(np.log(g1))

            # Update gamma_0
            gamma_0_ext = fmin_l_bfgs_b(
                func=lambda gamma_ext_: Q_func(gamma_ext_, pi_est, E_log_g1, E_g1,
                            baseline_hazard, delta, indicator_2), x0=gamma_0_0,
                fprime=lambda gamma_ext_: grad_Q(gamma_ext_), disp=False,
                bounds=bounds_gamma, maxiter=maxiter, pgtol=pgtol)[0]

            # Update gamma_1
            gamma_1_ext = fmin_l_bfgs_b(
                func=lambda gamma_ext_: Q_func(gamma_ext_, pi_est, E_log_g1, E_g1,
                            baseline_hazard, delta, indicator_2), x0=gamma_1_0,
                fprime=lambda gamma_ext_: grad_Q(gamma_ext_), disp=False,
                bounds=bounds_gamma, maxiter=maxiter, pgtol=pgtol)[0]

            self.update_theta(gamma_0=gamma_0_ext, gamma_1=gamma_1_ext)

            # Update baseline hazard
            g1 = g1(X, T_u, S)
            E_g1 = Eg(g1)
            baseline_hazard = ((indicator_1 * 1).T * delta).sum(axis=1) / \
                              ((E_g1.T * pi_est.T).T.swapaxes(0, 1)[:,
                               indicator_1].sum(axis=0)
                               * (indicator_2 * 1).T).sum(axis=1)

            self.update_theta(phi=phi, baseline_hazard=baseline_hazard,
                              long_cov=D)
            prev_obj = obj
            obj = func_obj(X, Y, T, delta, xi_ext)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)

            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj,
                                    theta=self.theta)
                if verbose:
                    self.history.print_history()
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        self._end_solve()

    def score(self, X, Y, T, delta, metric):
        """Computes the score with the trained parameters on the given data,
        either log-likelihood or C-index

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        metric : 'log_lik', 'C-index'
            Either computes log-likelihood or C-index

        Returns
        -------
        output : `float`
            The score computed on the given data
        """
        if metric == 'log_lik':
            return self._log_lik(X, Y, T, delta)
        elif metric == 'C-index':
            return c_index_score(T, self.predict_marker(X, Y), delta)
        else:
            raise ValueError("``metric`` must be 'log_lik' or 'C-index', got "
                             "%s instead" % metric)
