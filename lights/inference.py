# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score
from lights.base.base import Learner, extract_features, normalize, block_diag, \
    get_vect_from_ext, get_xi_from_xi_ext, logistic_grad, get_times_infos
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
        If `True`, include an intercept in the model for the time independent
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
        self._fitted = False

        # Attributes that will be instantiated afterwards
        self.n_samples = None
        self.n_time_indep_features = None
        self.n_long_features = None
        self.theta = {
            "beta_0": np.empty(1),
            "beta_1": np.empty(1),
            "long_cov": np.empty(1),
            "phi": np.empty(1),
            "xi": np.empty(1),
            "baseline_hazard": np.empty(1),
            "gamma_0": np.empty(1),
            "gamma_1": np.empty(1)
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

    @property
    def fitted(self):
        return self._fitted

    @staticmethod
    def _log_lik(pi_xi, f):
        """Computes the approximation of the likelihood of the lights model

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of f(Y, T, delta| S, G ; theta)

        Returns
        -------
        prb : `float`
            The approximated log-likelihood computed on the given data
        """
        pi_xi_ = np.vstack((1 - pi_xi, pi_xi)).T
        prb = np.log((pi_xi_ * f.mean(axis=-1)).sum(axis=-1)).mean()
        return prb

    def _func_obj(self, pi_xi, f):
        """The global objective to be minimized by the QNMCEM algorithm
        (including penalization)

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of f(Y, T, delta| S, G ; theta)

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        p, L = self.n_time_indep_features, self.n_long_features
        theta = self.theta
        log_lik = self._log_lik(pi_xi, f)
        # xi elastic net penalty
        xi = theta["xi"]
        xi_pen = self.pen.elastic_net(xi)
        # beta sparse group l1 penalty
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        beta_0_pen = self.pen.sparse_group_l1(beta_0, L)
        beta_1_pen = self.pen.sparse_group_l1(beta_1, L)
        # gamma sparse group l1 penalty
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_0_indep = gamma_0[:p]
        gamma_0_dep = gamma_0[p:]
        gamma_0_pen = self.pen.elastic_net(gamma_0_indep)
        gamma_0_pen += self.pen.sparse_group_l1(gamma_0_dep, L)
        gamma_1_indep = gamma_1[:p]
        gamma_1_dep = gamma_1[p:]
        gamma_1_pen = self.pen.elastic_net(gamma_1_indep)
        gamma_1_pen += self.pen.sparse_group_l1(gamma_1_dep, L)
        pen = xi_pen + beta_0_pen + beta_1_pen + gamma_0_pen + gamma_1_pen
        return -log_lik + pen

    def _get_proba(self, X, xi_ext):
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

    @staticmethod
    def _get_post_proba(pi_xi, Lambda_1):
        """Posterior probability estimates for being on the high-risk group
        given all observed data

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Comes from get_proba function

        Lambda_1 : `np.ndarray`, shape=(n_samples, K)
            Approximated integral (see (15) in the lights paper) with
            \tilde(g)=1

        Returns
        -------
        pi_est : `np.ndarray`, shape=(n_samples,)
            Returns the posterior probability of the sample for being on the
            high-risk group given all observed data
        """
        tmp = Lambda_1 * np.vstack((1 - pi_xi, pi_xi)).T
        pi_est = tmp[:, 1] / tmp.sum(axis=1)
        return pi_est

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
        if self._fitted:
            marker = None
            return marker
        else:
            raise RuntimeError('You must fit the model first')

    def _update_theta(self, **kwargs):
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
        """Fits the lights model

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
        alpha = self.fixed_effect_time_order
        n_samples, p = X.shape
        L = Y.shape[1]
        self.n_samples = n_samples
        self.n_time_indep_features = p
        self.n_long_features = L
        q_l = alpha + 1
        r_l = 2  # Affine random effects
        if fit_intercept:
            p += 1

        if self.asso_functions == 'all':
            self.asso_functions = ['lp', 're', 'tps', 'ce']
        asso_functions = self.asso_functions
        nb_asso_param = len(asso_functions)
        if 're' in asso_functions:
            nb_asso_param += 1
        nb_asso_feat = L * nb_asso_param + p
        N = 5  # Number of initial Monte Carlo sample for S

        X = normalize(X)  # Normalize time-independent features
        ext_feat = extract_features(Y, alpha)  # Features extraction
        T_u, J, ind_1, ind_2 = get_times_infos(T)

        # Initialization
        xi_ext = np.zeros(2 * p)

        if self.initialize:
            # Initialize longitudinal submodels
            mlmm = MLMM(max_iter=max_iter, verbose=verbose, tol=tol,
                        print_every=print_every, fixed_effect_time_order=alpha)
            mlmm.fit(ext_feat)
            beta = mlmm.fixed_effect_coeffs
            D = mlmm.long_cov
            phi = mlmm.phi
            est = initialize_asso_params(X, T, delta)
            time_indep_cox_coeffs, baseline_hazard = est
        else:
            # Fixed initialization
            q = q_l * L
            r = r_l * L
            beta = np.zeros((q, 1))
            D = np.diag(np.ones(r))
            phi = np.ones((L, 1))
            time_indep_cox_coeffs = np.zeros(p)
            baseline_hazard = pd.Series(data=np.zeros(J), index=T_u)

        gamma = np.zeros(nb_asso_feat)
        gamma[:p] = time_indep_cox_coeffs
        gamma_0_ext = np.concatenate((gamma, -gamma)).reshape(-1, 1)
        gamma_0_ext[gamma_0_ext < 0] = 0
        gamma_1_ext = gamma_0_ext.copy()

        beta_0_ext = np.concatenate((beta, -beta)).reshape(-1, 1)
        beta_0_ext[beta_0_ext < 0] = 0
        beta_1_ext = beta_0_ext.copy()

        self._update_theta(beta_0=beta_0_ext, beta_1=beta_1_ext, xi=xi_ext,
                          gamma_0=gamma_0_ext, gamma_1=gamma_1_ext, long_cov=D,
                          phi=phi, baseline_hazard=baseline_hazard)

        # Stopping criteria and bounds vector for the L-BGFS-B algorithms
        maxiter, pgtol = 60, 1e-5
        bounds_xi = [(0, None)] * 2 * p
        bounds_beta = [(0, None)] * 2 * L * q_l
        bounds_gamma = [(0, None)] * 2 * nb_asso_feat

        # Instanciates E-step and M-step functions
        E_func = EstepFunctions(X, T, delta, ext_feat, L, p, alpha,
                                asso_functions, self.theta)
        F_func = MstepFunctions(fit_intercept, X, T, delta, L, p, self.l_pen,
                                self.eta_elastic_net, self.eta_sp_gp_l1,
                                nb_asso_feat, alpha)

        S = E_func.construct_MC_samples(N)
        f = E_func.f_data_given_latent(S, ind_1, ind_2)
        Lambda_1 = E_func.Lambda_g(np.ones(shape=(n_samples, 2, 2 * N)), f)
        pi_xi = self._get_proba(X, xi_ext)
        obj = self._func_obj(pi_xi, f)

        # Store init values
        self.history.update(n_iter=0, obj=obj, rel_obj=np.inf, theta=self.theta)
        if verbose:
            self.history.print_history()

        for n_iter in range(1, max_iter + 1):

            # E-Step
            pi_est = self._get_post_proba(pi_xi, Lambda_1)
            E_g0 = E_func.Eg(E_func.g0(S), Lambda_1, pi_xi, f)
            E_g0_l = E_func.Eg(E_func.g0_l(S), Lambda_1, pi_xi, f)
            E_gS = E_func.Eg(E_func.gS(S), Lambda_1, pi_xi, f)
            E_g1 = E_func.Eg(E_func.g1(S), Lambda_1, pi_xi, f)
            E_g2 = E_func.Eg(E_func.g2(S, ind_1), Lambda_1, pi_xi, f)
            E_g5 = E_func.Eg(E_func.g5(S, ind_1), Lambda_1, pi_xi, f)
            E_g6 = E_func.Eg(E_func.g6(S), Lambda_1, pi_xi, f)
            E_g9 = E_func.Eg(E_func.g9(S), Lambda_1, pi_xi, f)

            if False:  # TODO: condition to be defined
                N *= 1.1
                fctr *= .1

            # M-Step
            D = E_g0.sum(axis=0) / n_samples  # D update

            if warm_start:
                xi_init = xi_ext
                beta_init = [beta_0_ext, beta_1_ext]
                gamma_init = [gamma_0_ext, gamma_1_ext]
            else:
                xi_init = np.zeros(2 * p)
                beta_init = np.zeros(2 * L * q_l)
                beta_init = [beta_init, beta_init.copy()]
                gamma_init = np.zeros(2 * nb_asso_feat).reshape(-1, 1)
                gamma_init = [gamma_init, gamma_init.copy()]

            # xi update
            xi_ext = fmin_l_bfgs_b(
                func=lambda xi_ext_: F_func.P_func(pi_est, xi_ext_), x0=xi_init,
                fprime=lambda xi_ext_: F_func.grad_P(pi_est, xi_ext_),
                disp=False, bounds=bounds_xi, maxiter=maxiter, pgtol=pgtol)[0]

            # beta update
            pi_est_K = [1 - pi_est, pi_est]
            [beta_0_ext, beta_1_ext] = [fmin_l_bfgs_b(
                func=lambda beta_ext:
                F_func.R_func(beta_ext, pi_est_K[k], E_g1.T[k].T, E_g2.T[k].T,
                              E_g9.T[k].T, baseline_hazard, ind_2),
                x0=beta_init[k],
                fprime=lambda beta_ext:
                F_func.grad_R(beta_ext, gamma_0_ext, pi_est_K[k], E_g5.T[k].T,
                              E_g6.T[k].T, E_gS, baseline_hazard, ind_2,
                              ext_feat, phi),
                disp=False, bounds=bounds_beta, maxiter=maxiter,
                pgtol=pgtol)[0].reshape(-1, 1)
                                        for k in [0, 1]]

            # beta needs to be updated before gamma
            self._update_theta(beta_0=beta_0_ext, beta_1=beta_1_ext)
            E_func.theta = self.theta
            g1 = E_func.g1(S)
            E_g1 = E_func.Eg(g1, Lambda_1, pi_xi, f)
            E_log_g1 = E_func.Eg(np.log(g1), Lambda_1, pi_xi, f)
            E_g7 = E_func.Eg(E_func.g7(S), Lambda_1, pi_xi, f)
            E_g8 = E_func.Eg(E_func.g8(S), Lambda_1, pi_xi, f)

            # gamma update
            [gamma_0_ext, gamma_1_ext] = [fmin_l_bfgs_b(
                func=lambda gamma_ext:
                F_func.Q_func(gamma_ext, pi_est[k], E_log_g1.T[k].T, E_g1.T[k].T,
                              baseline_hazard, ind_1, ind_2),
                x0=gamma_init[k],
                fprime=lambda gamma_ext:
                F_func.grad_Q(gamma_ext, pi_est[k], E_g1.T[k].T, E_g7.T[k].T,
                              E_g8.T[k].T, baseline_hazard, ind_1, ind_2),
                disp=False, bounds=bounds_gamma, maxiter=maxiter,
                pgtol=pgtol)[0].reshape(-1, 1)
                                          for k in [0, 1]]

            # gamma needs to be updated before the baseline
            self._update_theta(gamma_0=gamma_0_ext, gamma_1=gamma_1_ext)
            E_func.theta = self.theta
            E_g1 = E_func.Eg(E_func.g1(S), Lambda_1, pi_xi, f)

            # baseline hazard update
            baseline_hazard = pd.Series(
                data=((ind_1 * 1).T * delta).sum(axis=1) / (
                        (E_g1.T * (ind_2 * 1).T).swapaxes(0, 1)
                        * pi_est.T).sum(axis=2).sum(axis=1),
                index=T_u)

            # phi update
            (U_L, V_L, y_L, N_L) = ext_feat[1]
            E_gS_ = E_gS.reshape(n_samples, L, q_l)
            for l in range(L):
                pi_est_ = np.concatenate([[pi_est[i]] * N_L[l][i]
                                          for i in range(n_samples)])
                pi_est_ = np.vstack((1 - pi_est_, pi_est_)).T  # K = 2
                N_l, y_l, U_l, V_l = sum(N_L[l]), y_L[l], U_L[l], V_L[l]
                beta_l = beta[q_l * l: q_l * (l + 1)]
                E_b_l = E_gS_[:, l].reshape(-1, 1)
                E_bb_l = block_diag(E_g0_l[:, l])
                tmp = y_l - U_l.dot(beta_l)
                phi_l = pi_est_ * (tmp.T.dot(tmp - 2 * (V_l.dot(E_b_l))) +
                                   np.trace((V_l.T.dot(V_l).dot(E_bb_l))))
                phi[l] = phi_l.sum() / N_l

            self._update_theta(phi=phi, baseline_hazard=baseline_hazard,
                              long_cov=D, xi=xi_ext)
            pi_xi = self._get_proba(X, xi_ext)
            E_func.theta = self.theta
            S = E_func.construct_MC_samples(N)
            f = E_func.f_data_given_latent(S, ind_1, ind_2)

            prev_obj = obj
            obj = self._func_obj(pi_xi, f)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj,
                                    theta=self.theta)
                if verbose:
                    self.history.print_history()

            if (n_iter > max_iter) or (rel_obj < tol):
                self._fitted = True
                break
            else:
                # Update for next iteration
                Lambda_1 = E_func.Lambda_g(np.ones((n_samples, 2, 2 * N)), f)

        self._end_solve()

    def score(self, X, Y, T, delta):
        """Computes the C-index score with the trained parameters on the given
        data

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
            The C-index score computed on the given data
        """
        if self._fitted:
            return c_index_score(T, self.predict_marker(X, Y), delta)
        else:
            raise ValueError('You must fit the model first')
