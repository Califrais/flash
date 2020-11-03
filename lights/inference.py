# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from lights.base.base import Learner, extract_features, normalize
from lights.init.mlmm import MLMM
from lights.association import AssociationFunctions
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score
from sklearn.model_selection import KFold
from lights.init.cox import initialize_asso_params
import pandas as pd


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
        self.l_pen = l_pen
        self.eta_elastic_net = eta_elastic_net
        self.eta_sp_gp_l1 = eta_sp_gp_l1
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.initialize = initialize

        # Attributes that will be instantiated afterwards
        self.n_time_indep_features = None
        self.n_samples = None
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
        self.avg_scores = None
        self.scores = None
        self.l_pen_best = None
        self.l_pen_chosen = None
        self.grid_elastic_net = None
        self.adaptative_grid_el = None
        self.grid_size = None

    @property
    def l_pen(self):
        return self._l_pen

    @l_pen.setter
    def l_pen(self, val):
        if not val >= 0:
            raise ValueError("``l_pen`` must be non negative")
        self._l_pen = val

    @property
    def eta_elastic_net(self):
        return self._eta_elastic_net

    @eta_elastic_net.setter
    def eta_elastic_net(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta_elastic_net`` must be in (0, 1)")
        self._eta_elastic_net = val

    @property
    def eta_sp_gp_l1(self):
        return self._eta_sp_gp_l1

    @eta_sp_gp_l1.setter
    def eta_sp_gp_l1(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta_sp_gp_l1`` must be in (0, 1)")
        self._eta_sp_gp_l1 = val

    @property
    def asso_functions(self):
        return self._asso_functions

    @asso_functions.setter
    def asso_functions(self, val):
        if not (val == 'all' or set(val).issubset({'lp', 're', 'tps', 'ce'})):
            raise ValueError("``asso_functions`` must be either 'all', or a "
                             "`list` in ['lp', 're', 'tps', 'ce']")
        self._asso_functions = val

    @staticmethod
    def logistic_grad(z):
        """Overflow proof computation of 1 / (1 + exp(-z)))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = 1. / (1. + np.exp(-z[idx_pos]))
        res[idx_neg] = 1 - 1. / (1. + np.exp(z[idx_neg]))
        return res

    @staticmethod
    def logistic_loss(z):
        """Overflow proof computation of log(1 + exp(-z))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = np.log(1. + np.exp(-z[idx_pos]))
        z_neg = z[idx_neg]
        res[idx_neg] = -z_neg + np.log(1. + np.exp(z_neg))
        return res

    @staticmethod
    def get_vect_from_ext(v_ext):
        """Obtain the signed coefficient vector from its extension on positive
        and negative parts
        """
        dim = len(v_ext)
        if dim % 2 != 0:
            raise ValueError("``v_ext`` dimension cannot be odd, got %s" % dim)
        v = v_ext[:dim // 2] - v_ext[dim // 2:]
        return v

    def _get_xi_from_xi_ext(self, xi_ext):
        """Get the time-independent coefficient vector from its extension on
        positive and negative parts

        Parameters
        ----------
        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        xi_0 : `float`
            The intercept term

        xi : `np.ndarray`, shape=(n_time_indep_features,)
            The time-independent coefficient vector
        """
        n_time_indep_features = self.n_time_indep_features
        if self.fit_intercept:
            xi = xi_ext[:n_time_indep_features + 1] - \
                 xi_ext[n_time_indep_features + 1:]
            xi_0 = xi[0]
            xi = xi[1:]
        else:
            xi_0 = 0
            xi = xi_ext[:n_time_indep_features] - xi_ext[n_time_indep_features:]
        return xi_0, xi

    def _clean_xi_ext(self, xi_ext):
        """Removes potential intercept coefficients in the time-independent
        coefficient vector decomposed on positive and negative parts
        """
        if self.fit_intercept:
            n_time_indep_features = self.n_time_indep_features
            xi_ext = np.delete(xi_ext, [0, n_time_indep_features + 1])
        return xi_ext

    def _elastic_net_pen(self, xi_ext):
        """Computes the elasticNet penalization of vector xi

        Parameters
        ----------
        xi_ext: `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the elasticNet penalization part of vector xi
        """
        l_pen = self.l_pen
        eta = self.eta_elastic_net
        xi = self._get_xi_from_xi_ext(xi_ext)[1]
        xi_ext = self._clean_xi_ext(xi_ext)
        return l_pen * ((1. - eta) * xi_ext.sum() +
                        0.5 * eta * np.linalg.norm(xi) ** 2)

    def _grad_elastic_net_pen(self, xi):
        """Computes the gradient of the elasticNet penalization of vector xi

        Parameters
        ----------
        xi : `np.ndarray`, shape=(n_time_indep_features,)
            The time-independent coefficient vector

        Returns
        -------
        output : `float`
            The gradient of the elasticNet penalization part of vector xi
        """
        l_pen = self.l_pen
        eta = self.eta_elastic_net
        n_time_indep_features = self.n_time_indep_features
        grad = np.zeros(2 * n_time_indep_features)
        # Gradient of lasso penalization
        grad += l_pen * (1 - eta)
        # Gradient of ridge penalization
        grad_pos = (l_pen * eta) * xi
        grad[:n_time_indep_features] += grad_pos
        grad[n_time_indep_features:] -= grad_pos
        return grad

    def _sparse_group_l1_pen(self, v_ext):
        """Computes the sparse group l1 penalization of vector v

        Parameters
        ----------
        v_ext: `np.ndarray`
            A vector decomposed on positive and negative parts

        Returns
        -------
        output : `float`
            The value of the sparse group l1 penalization of vector v
        """
        l_pen = self.l_pen
        eta = self.eta_sp_gp_l1
        v = self.get_vect_from_ext(v_ext)
        return l_pen * ((1. - eta) * v_ext.sum() + eta * np.linalg.norm(v))

    def _grad_sparse_group_l1_pen(self, v):
        """Computes the gradient of the sparse group l1 penalization of a
        vector v

        Parameters
        ----------
        v : `np.ndarray`
            A coefficient vector

        Returns
        -------
        output : `float`
            The gradient of the sparse group l1 penalization of vector v
        """
        l_pen = self.l_pen
        eta = self.eta_sp_gp_l1
        L = self.n_long_features
        dim = len(v)
        grad = np.zeros(2 * dim)
        # Gradient of lasso penalization
        grad += l_pen * (1 - eta)
        # Gradient of sparse group l1 penalization
        # TODO Van Tuan : to be verified
        tmp = np.array([np.repeat(np.linalg.norm(v_l), dim // L)
                        for v_l in np.array_split(v, L)]).flatten()
        grad_pos = (l_pen * eta) * v / tmp
        grad[:dim] += grad_pos
        grad[dim:] -= grad_pos
        return grad

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
        pen = self._elastic_net_pen(xi_ext)
        return -log_lik + pen

    def _P_func(self, X, pi_est, xi_ext):
        """Computes the sub objective function denoted P in the lights paper,
        to be minimized at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the P sub objective to be minimized at each QNMCEM step
        """
        xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
        pen = self._elastic_net_pen(xi_ext)
        u = xi_0 + X.dot(xi)
        sub_obj = (pi_est * u + self.logistic_loss(u)).mean()
        return sub_obj + pen

    def _grad_P(self, X, pi_est, xi_ext):
        """Computes the gradient of the sub objective P

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the P sub objective gradient
        """
        n_time_indep_features = self.n_time_indep_features
        n_samples = self.n_samples
        xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
        grad_pen = self._grad_elastic_net_pen(xi)
        u = xi_0 + X.dot(xi)
        if self.fit_intercept:
            X = np.concatenate((np.ones(n_samples).reshape(1, n_samples).T, X),
                               axis=1)
            grad_pen = np.concatenate([[0], grad_pen[:n_time_indep_features],
                                       [0], grad_pen[n_time_indep_features:]])
        grad = (X * (pi_est - self.logistic_grad(-u)).reshape(
            n_samples, 1)).mean(axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def _R_func(self, beta_ext, pi_est, E_g1, E_g2, E_g8, baseline_hazard,
                delta, indicator):
        """Computes the sub objective function denoted R in the lights paper,
        to be minimized at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
        beta_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        E_g1 : `np.ndarray`, shape=(n_samples, J, 2)
            The approximated expectations of function g1

        E_g2 : `np.ndarray`, shape=(n_samples, 2)
            The approximated expectations of function g2

        E_g8 : `np.ndarray`, shape=(n_samples, 2)
            The approximated expectations of function g8

        baseline_hazard : `np.ndarray`, shape=(n_samples,)
            The baseline hazard function evaluated at each censored time

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        Returns
        -------
        output : `float`
            The value of the R sub objective to be minimized at each QNMCEM step
        """
        n_samples = delta.shape[0]
        pen = self._sparse_group_l1_pen(beta_ext)
        E_g1_ = E_g1.swapaxes(1, 2).swapaxes(0, 1)
        baseline_val = baseline_hazard.values.flatten()
        ind_ = indicator * 1
        sub_obj = E_g2 * delta.reshape(-1, 1) + E_g8 - np.sum(
            E_g1_ * baseline_val * ind_, axis=2).T
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples + pen

    def _grad_R(self, beta_ext, E_g5, E_g6, delta, baseline_hazard, indicator):
        """Computes the gradient of the sub objective R

        Parameters
        ----------
        # TODO Van Tuan

        Returns
        -------
        output : `float`
            The value of the R sub objective gradient
        """
        beta = self.get_vect_from_ext(beta_ext)
        grad_pen = self._grad_sparse_group_l1_pen(beta)
        # TODO: handle gamma
        tmp1 = (E_g5.T * delta).T - np.sum(
            (E_g6.T * baseline_hazard.values * (indicator * 1).T).T, axis=1)
        tmp2 = 0
        grad = tmp1 + tmp2
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def _Q_func(self, gamma_ext, pi_est, E_log_g1, E_g1, baseline_hazard, delta,
                indicator):
        """Computes the sub objective function denoted Q in the lights paper,
        to be minimized at each QNMCEM iteration using fmin_l_bfgs_b

        Parameters
        ----------
        gamma_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        E_log_g1 : `np.ndarray`, shape=(n_samples, J, 2)
            The approximated expectations of function logarithm of g1

        E_g1 : `np.ndarray`, shape=(n_samples, J, 2)
            The approximated expectations of function g1

        baseline_hazard : `np.ndarray`, shape=(n_samples,)
            The baseline hazard function evaluated at each censored time

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        Returns
        -------
        output : `float`
            The value of the Q sub objective to be minimized at each QNMCEM step
        """
        n_samples = delta.shape[0]
        pen = self._sparse_group_l1_pen(gamma_ext)

        E_g1_ = E_g1.swapaxes(1, 2).swapaxes(0, 1)
        baseline_val = baseline_hazard.values.flatten()
        ind_ = indicator * 1
        sub_obj = E_log_g1 * delta.reshape(-1, 1) - np.sum(
            E_g1_ * baseline_val * ind_, axis=2).T
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples + pen

    def _grad_Q(self, gamma_ext):
        """Computes the gradient of the sub objective Q

        Parameters
        ----------
        # TODO Van Tuan

        Returns
        -------
        output : `float`
            The value of the Q sub objective gradient
        """
        gamma = self.get_vect_from_ext(gamma_ext)
        grad_pen = self._grad_sparse_group_l1_pen(gamma)
        # TODO Van Tuan
        grad = 0
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

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
        xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
        u = xi_0 + X.dot(xi)
        return QNMCEM.logistic_grad(u)

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

    def get_asso_func(self, T, S, derivative=False):
        """Computes association functions or derivatives association ones

        Parameters
        ----------
        T : `np.ndarray`, shape=(J,)
            The J unique censored times of the event of interest

        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        derivative : `bool`, default=False
        If `False`, returns the association functions, otherwise returns the
        derivative versions

        Returns
        -------
        asso_func_stack : `np.ndarray`, , shape=(2, n_samples*2*N, dim)
            Stack version of association functions or derivatives for all
            subjects, all groups and all Monte Carlo samples. `dim` is the
            total dimension of returned association functions.
        """
        fixed_effect_coeffs = np.array([self.theta["beta_0"],
                                        self.theta["beta_1"]])
        fixed_effect_time_order = self.fixed_effect_time_order
        n_long_features = self.n_long_features
        J = T.shape[0]
        asso_functions = self.asso_functions
        q_l = fixed_effect_time_order + 1

        N = S.shape[0] // 2
        asso_func = AssociationFunctions(T, S, fixed_effect_coeffs,
                                         fixed_effect_time_order,
                                         n_long_features)

        if derivative:
            asso_func_stack = np.empty(shape=(2, 2 * N, J, n_long_features, 0))
        else:
            asso_func_stack = np.empty(shape=(2, J * 2 * N, 0))

        for func_name in asso_functions:
            if derivative:
                func = asso_func.assoc_func_dict["d_" + func_name]
                func_r = func.reshape(2, 2 * N, J, n_long_features, q_l)
            else:
                func = asso_func.assoc_func_dict[func_name]
                dim = n_long_features
                if func_name == 're':
                    dim *= 2
                func_r = func.swapaxes(0, 1).swapaxes(2, 3).reshape(
                    2, J * 2 * N, dim)
            asso_func_stack = np.concatenate((asso_func_stack, func_r), axis=-1)

        return asso_func_stack

    def f_data_given_latent(self, X, extracted_features, T, delta, S):
        """Computes f(Y, T, delta| S, G, theta)

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        T : `np.ndarray`, shape=(n_samples,)
            The censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            The censoring indicator

        S: `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, 2, N)
            The value of the f(Y, T, delta| S, G, theta)
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        beta_0, beta_1 = self.theta["beta_0"], self.theta["beta_1"]
        baseline_hazard = self.theta["baseline_hazard"]
        phi = self.theta["phi"]
        T_u = np.unique(T)
        (U_list, V_list, y_list, N_list) = extracted_features[0]
        N = S.shape[0] // 2
        g1 = self._g1(X, T, S)

        f = np.ones(shape=(n_samples, 2, N * 2))
        # TODO LATER : to be optimized
        for i in range(n_samples):
            t_i = T[i]
            baseline_hazard_t_i = baseline_hazard.loc[[t_i]].values
            tmp = g1[i].swapaxes(2, 1).swapaxes(1, 0)
            op1 = (baseline_hazard_t_i * tmp[T_u == t_i]) ** delta[i]
            op2 = np.sum(tmp[T_u <= t_i] * baseline_hazard.loc[
                T_u[T_u <= t_i]].values.reshape(-1, 1, 1), axis=0)

            # Compute f(y|b)
            beta_stack = np.hstack((beta_0, beta_1))
            U_i = U_list[i]
            V_i = V_list[i]
            n_i = sum(N_list[i])
            y_i = y_list[i]
            Phi_i = [[phi[l, 0]] * N_list[i][l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            M_iS = U_i.dot(beta_stack).T.reshape(2, -1, 1) + V_i.dot(S.T)
            f_y = 1 / np.sqrt((2 * np.pi) ** n_i * np.prod(Phi_i) * np.exp(
                np.sum(((y_i - M_iS) ** 2) / Phi_i, axis=1)))

            f[i] = op1 * np.exp(-op2) * f_y

        return f

    def construct_MC_samples(self, N):
        """Constructs the set of samples used for Monte Carlo approximation

        Parameters
        ----------
        N : `int`
            Number of constructed samples

        Returns
        -------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples
        """
        D = self.theta["long_cov"]
        C = np.linalg.cholesky(D)
        r = D.shape[0]
        Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N)
        b = Omega.dot(C.T)
        S = np.vstack((b, -b))
        return S

    @staticmethod
    def _g0(S):
        """Computes g0
        """
        g0 = np.array([s.reshape(-1, 1).dot(s.reshape(-1, 1).T) for s in S])
        return g0

    def _g1(self, X, T, S):
        """Computes g1

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        T : `np.ndarray`, shape=(n_samples,)
            The censored times of the event of interest

        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g1 : `np.ndarray`, shape=(n_samples, 2, 2*N, J)
            The values of g1 function
        """
        #TODO Simon: pass directly the T_u
        T_u = np.unique(T)
        n_samples = self.n_samples
        N = S.shape[0] // 2
        J = T_u.shape[0]
        p = self.n_time_indep_features
        gamma_0, gamma_1 = self.theta["gamma_0"], self.theta["gamma_1"]
        gamma_indep_stack = np.vstack((gamma_0[:p], gamma_1[:p])).T
        g2 = self._g2(T_u, S)
        tmp = X.dot(gamma_indep_stack)
        g1 = np.exp(
            tmp.T.reshape(2, n_samples, 1, 1) + g2.reshape(2, 1, J, 2 * N))
        g1 = g1.swapaxes(0, 1).swapaxes(2, 3)
        return g1

    def _g2(self, T_u, S):
        """Computes g2

        Parameters
        ----------
        T_u : `np.ndarray`, shape=(J,)
            The J unique censored times of the event of interest

        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g2 : `np.ndarray`, shape=(2, J, 2*N)
            The values of g2 function
        """
        N = S.shape[0] // 2
        p = self.n_time_indep_features
        gamma_0, gamma_1 = self.theta["gamma_0"], self.theta["gamma_1"]
        asso_func = self.get_asso_func(T_u, S)
        J = T_u.shape[0]
        gamma_time_depend_stack = np.vstack((gamma_0[p:], gamma_1[p:])).reshape(
            (2, 1, -1))
        g2 = np.sum(asso_func * gamma_time_depend_stack, axis=2).reshape(
            (2, J, 2 * N))
        return g2

    def _g5(self, T, S):
        """Computes g5

        Parameters
        ----------
        T : `np.ndarray`, shape=(n_samples,)
            The censored times of the event of interest

        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g5 : `np.ndarray`, shape=(2, 2 * N, J, n_long_features, q_l)
            The values of g5 function
        """
        g5 = self.get_asso_func(T, S, derivative=True)
        return g5

    def _g6(self, X, T, S):
        """Computes g6

        Parameters
        ----------
        T : `np.ndarray`, shape=(n_samples,)
            The censored times of the event of interest

        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g6 : `np.ndarray`, shape=(n_samples, n_long_features, 2, 2 * N * J, q_l)
            The values of g6 function
        """
        T_u = np.unique(T)
        n_samples = T.shape[0]
        g5 = self._g5(T_u, S)
        g5 = np.broadcast_to(g5, (n_samples,) + g5.shape)
        g1 = self._g1(X, T, S)
        g6 = (g1.T * g5.T).T
        return g6

    def _g8(self, extracted_features, S):
        """Computes g8

        Parameters
        ----------
        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g8 : `np.ndarray`, shape=(n_samples, 2, 2 * N)
            The values of g8 function
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        beta_0, beta_1 = self.theta["beta_0"], self.theta["beta_1"]
        phi = self.theta["phi"]
        (U_list, V_list, y_list, N_list) = extracted_features[0]

        g8 = np.zeros(shape=(n_samples, 2, S.shape[0]))
        for i in range(n_samples):
            beta_stack = np.hstack((beta_0, beta_1))
            U_i = U_list[i]
            V_i = V_list[i]
            y_i = y_list[i]
            Phi_i = [[phi[l, 0]] * N_list[i][l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            M_iS = U_i.dot(beta_stack).T.reshape(2, -1, 1) + V_i.dot(S.T)
            g8[i] = np.sum(M_iS * y_i * Phi_i + (M_iS ** 2) * Phi_i, axis=1)

        return g8

    @staticmethod
    def _Lambda_g(g, f):
        """Approximated integral (see (15) in the lights paper)

        Parameters
        ----------
        g : `np.ndarray`, shape=(n_samples, 2, N)
            Values of g function for all subjects, all groups and all Monte
            Carlo samples. Each element could be real or matrices depending on
            Im(\tilde{g}_i)

        f: `np.ndarray`, shape=(n_samples, 2, N)
            Values of the density of the observed data given the latent ones and
            the current estimate of the parameters, computed for all subjects,
            all groups and all Monte Carlo samples

        Returns
        -------
        Lambda_g : `np.array`, shape=(n_samples, 2)
            The approximated integral computed for all subjects, all groups and
            all Monte Carlo samples. Each element could be real or matrices
            depending on Im(\tilde{g}_i)
        """
        Lambda_g = np.mean((g.T * f.T).T, axis=2)
        return Lambda_g

    @staticmethod
    def _Eg(pi_xi, Lambda_1, Lambda_g):
        """Computes approximated expectations of different functions g taking
        random effects as input, conditional on the observed data and the
        current estimate of the parameters. See (14) in the lights paper

        Parameters
        ----------
        pi_xi : `np.array`, shape=(n_samples,)
            The value of g function for all samples

        Lambda_1: `np.ndarray`, shape=(n_samples, 2)
            Approximated integral (see (15) in the lights paper) with
            \tilde(g)=1

        Lambda_g: `np.ndarray`, shape=(n_samples, 2)
             Approximated integral (see (15) in the lights paper)

        Returns
        -------
        Eg : `np.ndarray`, shape=(n_samples,)
            The approximated expectations for g
        """
        Eg = ((Lambda_g[:, 0].T * (1 - pi_xi) + Lambda_g[:, 1].T * pi_xi)
              / (Lambda_1[:, 0] * (1 - pi_xi) + Lambda_1[:, 1] * pi_xi)).T
        return Eg

    def update_theta(self, **kwargs):
        """Update class attributes corresponding to lights model parameters

        Parameters
        ----------

        """
        for key, value in kwargs.items():
            if key in ["beta_0", "beta_1", "gamma_0", "gamma_1"]:
                self.theta[key] = self.get_vect_from_ext(value)
            elif key in ["long_cov", "phi", "baseline_hazard"]:
                self.theta[key] = value
            elif key in ["xi"]:
                self.theta[key] = self._get_xi_from_xi_ext(value)[1]
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
        r_l = 2  # linear time-varying features, so all r_l=2
        if fit_intercept:
            n_time_indep_features += 1

        if self.asso_functions == 'all':
            self.asso_functions = ['lp', 're', 'tps', 'ce']
        asso_functions = self.asso_functions
        nb_asso_param = len(asso_functions)
        if 're' in asso_functions:
            nb_asso_param += 1
        nb_asso_features = n_long_features * nb_asso_param + n_time_indep_features
        N = 5  # number of initial Monte Carlo sample for S

        # normalize time-independent features
        X = normalize(X)

        # features extraction
        extracted_features = extract_features(Y, fixed_effect_time_order)

        # initialization
        xi_ext = np.zeros(2 * n_time_indep_features)

        # the J unique censored times of the event of interest
        T_u = np.unique(T)
        J = T_u.shape[0]

        # create indicator matrices to compare event times
        # TODO: use indicator to update f_data_given_latent
        tmp = np.broadcast_to(T, (n_samples, n_samples))
        indicator = (tmp < tmp.T) * 1 + np.eye(n_samples)
        indicator_1 = T.reshape(-1, 1) == T_u
        indicator_2 = T.reshape(-1, 1) >= T_u

        # initialize longitudinal submodels
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
            # fixed initialization
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
        P_func, grad_P = self._P_func, self._grad_P
        R_func, grad_R = self._R_func, self._grad_R
        Q_func, grad_Q = self._Q_func, self._grad_Q

        obj = func_obj(X, Y, T, delta, xi_ext)
        # store init values
        self.history.update(n_iter=0, obj=obj, rel_obj=np.inf, theta=self.theta)
        if verbose:
            self.history.print_history()

        # stopping criteria and bounds vector for the L-BGFS-B algorithms
        maxiter, pgtol = 60, 1e-5
        bounds_xi = [(0, None)] * 2 * n_time_indep_features
        bounds_beta = [(0, None)] * 2 * n_long_features * \
                      (fixed_effect_time_order + 1)
        bounds_gamma = [(0, None)] * 2 * nb_asso_features

        # TODO : E_g1 = None
        for n_iter in range(1, max_iter + 1):

            pi_xi = self.get_proba(X, xi_ext)

            # E-Step
            S = self.construct_MC_samples(N)
            f = self.f_data_given_latent(X, extracted_features, T, delta, S)
            Lambda_1 = self._Lambda_g(np.ones(shape=(n_samples, 2, 2 * N)), f)
            pi_est = self.get_post_proba(pi_xi, Lambda_1)

            g0 = self._g0(S)
            g0 = np.broadcast_to(g0, (n_samples, 2) + g0.shape)
            Lambda_g0 = self._Lambda_g(g0, f)
            E_g0 = self._Eg(pi_xi, Lambda_1, Lambda_g0)

            g1 = self._g1(X, T, S)
            g1 = np.broadcast_to(g1[..., None], g1.shape + (2,)).swapaxes(1, 4)
            Lambda_g1 = self._Lambda_g(g1, f).swapaxes(1, 3)
            E_g1 = self._Eg(pi_xi, Lambda_1, Lambda_g1)

            g2 = self._g2(T, S).swapaxes(0, 1)
            g2 = np.broadcast_to(g2[..., None], g2.shape + (2,)).swapaxes(1, 3)
            Lambda_g2 = self._Lambda_g(g2, f).swapaxes(1, 2)
            E_g2 = self._Eg(pi_xi, Lambda_1, Lambda_g2)

            g5 = self._g5(T, S)
            g5 = np.broadcast_to(g5[..., None], g5.shape + (2,)).swapaxes(0, 5)
            Lambda_g5 = self._Lambda_g(g5.swapaxes(0, 2).swapaxes(1, 2),
                                       f).swapaxes(1, 4)
            E_g5 = self._Eg(pi_xi, Lambda_1, Lambda_g5)

            g6 = self._g6(X, T, S)
            g6 = np.broadcast_to(g6[..., None], g6.shape + (2,)).swapaxes(1, 6)
            Lambda_g6 = self._Lambda_g(g6, f).swapaxes(1, 5)
            E_g6 = self._Eg(pi_xi, Lambda_1, Lambda_g6)

            g8 = self._g8(extracted_features, S)
            g8 = np.broadcast_to(g8[..., None], g8.shape + (2,)).swapaxes(1, 3)
            Lambda_g8 = self._Lambda_g(g8, f).swapaxes(1, 2)
            E_g8 = self._Eg(pi_xi, Lambda_1, Lambda_g8)

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
                func=lambda xi_ext_: P_func(X, pi_est, xi_ext_), x0=xi_0,
                fprime=lambda xi_ext_: grad_P(X, pi_est, xi_ext_),
                disp=False, bounds=bounds_xi, maxiter=maxiter, pgtol=pgtol)[0]

            # Update beta_0
            beta_0_ext = fmin_l_bfgs_b(
                func=lambda beta_ext_: R_func(beta_ext_, pi_est, E_g1, E_g2, E_g8,
                            baseline_hazard, delta, indicator_2), x0=beta_0_0,
                fprime=lambda beta_ext_: grad_R(beta_ext_), disp=False,
                bounds=bounds_beta, maxiter=maxiter, pgtol=pgtol)[0]

            # Update beta_1
            beta_1_ext = fmin_l_bfgs_b(
                func=lambda beta_ext_: R_func(beta_ext_, pi_est, E_g1, E_g2, E_g8,
                            baseline_hazard, delta, indicator_2), x0=beta_1_0,
                fprime=lambda beta_ext_: grad_R(beta_ext_), disp=False,
                bounds=bounds_beta, maxiter=maxiter, pgtol=pgtol)[0]

            self.update_theta(beta_0=beta_0_ext, beta_1=beta_1_ext)

            g1_Q = self._g1(X, T, S)
            g1_Q = np.broadcast_to(g1_Q[..., None], g1_Q.shape + (2,)).swapaxes(1, 4)
            Lambda_g1_Q = self._Lambda_g(g1_Q, f).swapaxes(1, 3)
            E_g1_Q = self._Eg(pi_xi, Lambda_1, Lambda_g1_Q)

            log_g1_Q = np.log(g1_Q)
            Lambda_log_g1_Q = self._Lambda_g(log_g1_Q, f).swapaxes(1, 3)
            E_log_g1_Q = (self._Eg(pi_xi, Lambda_1, Lambda_log_g1_Q).T * (
                        indicator_1 * 1).T).sum(axis=1).T

            # Update gamma_0
            gamma_0_ext = fmin_l_bfgs_b(
                func=lambda gamma_ext_: Q_func(gamma_ext_, pi_est, E_log_g1_Q, E_g1_Q,
                            baseline_hazard, delta, indicator_2), x0=gamma_0_0,
                fprime=lambda gamma_ext_: grad_Q(gamma_ext_), disp=False,
                bounds=bounds_gamma, maxiter=maxiter, pgtol=pgtol)[0]

            # Update gamma_1
            gamma_1_ext = fmin_l_bfgs_b(
                func=lambda gamma_ext_: Q_func(gamma_ext_, pi_est, E_log_g1_Q, E_g1_Q,
                            baseline_hazard, delta, indicator_2), x0=gamma_1_0,
                fprime=lambda gamma_ext_: grad_Q(gamma_ext_), disp=False,
                bounds=bounds_gamma, maxiter=maxiter, pgtol=pgtol)[0]

            self.update_theta(gamma_0=gamma_0_ext, gamma_1=gamma_1_ext)

            # Update baseline hazard
            E_g1 = self._Eg(pi_xi, Lambda_1, Lambda_g1)
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

    def cross_validate(self, X, Y, T, delta, n_folds=10, eta=0.1,
                       adaptative_grid_el=True, grid_size=30,
                       grid_elastic_net=np.array([0]), shuffle=True,
                       verbose=True, metric='C-index'):
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

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        n_folds : `int`, default=10
            Number of folds. Must be at least 2.

        eta : `float`, default=0.1
            The ElasticNet mixing parameter, with 0 <= eta <= 1.
            For eta = 0 this is ridge (L2) regularization
            For eta = 1 this is lasso (L1) regularization
            For 0 < eta < 1, the regularization is a linear combination
            of L1 and L2

        adaptative_grid_el : `bool`, default=True
            If `True`, adapt the ElasticNet strength parameter grid using the
            KKT conditions

        grid_size : `int`, default=30
            Grid size if adaptative_grid_el=`True`

        grid_elastic_net : `np.ndarray`, default=np.array([0])
            Grid of ElasticNet strength parameters to be run through, if
            adaptative_grid_el=`False`

        shuffle : `bool`, default=True
            Whether to shuffle the data before splitting into batches

        verbose : `bool`, default=True
            If `True`, we verbose things, otherwise the solver does not
            print anything (but records information in history anyway)

        metric : 'log_lik', 'C-index', default='C-index'
            Either computes log-likelihood or C-index
        """
        n_samples = T.shape[0]
        cv = KFold(n_splits=n_folds, shuffle=shuffle)
        self.grid_elastic_net = grid_elastic_net
        self.adaptative_grid_el = adaptative_grid_el
        self.grid_size = grid_size
        tol = self.tol
        warm_start = self.warm_start

        if adaptative_grid_el:
            # from KKT conditions
            gamma_max = 1. / np.log(10.) * np.log(
                1. / (1. - eta) * (.5 / n_samples)
                * np.absolute(X).sum(axis=0).max())
            grid_elastic_net = np.logspace(gamma_max - 4, gamma_max, grid_size)

        learners = [
            QNMCEM(verbose=False, tol=tol, eta=eta, warm_start=warm_start,
                   fit_intercept=self.fit_intercept)
            for _ in range(n_folds)
        ]

        # TODO Sim: adapt to randomized search

        n_grid_elastic_net = grid_elastic_net.shape[0]
        scores = np.empty((n_grid_elastic_net, n_folds))
        if verbose is not None:
            verbose = self.verbose
        for idx_elasticNet, l_pen in enumerate(grid_elastic_net):
            if verbose:
                print("Testing l_pen=%.2e" % l_pen, "on fold ",
                      end="")
            for n_fold, (idx_train, idx_test) in enumerate(cv.split(X)):
                if verbose:
                    print(" " + str(n_fold), end="")
                X_train, X_test = X[idx_train], X[idx_test]
                T_train, T_test = Y[idx_train], T[idx_test]
                delta_train, delta_test = delta[idx_train], delta[idx_test]
                learner = learners[n_fold]
                learner.l_pen = l_pen
                learner.fit(X_train, T_train, delta_train)
                scores[idx_elasticNet, n_fold] = learner.score(
                    X_test, T_test, delta_test, metric)
            if verbose:
                print(": avg_score=%.2e" % scores[idx_elasticNet, :].mean())

        avg_scores = scores.mean(1)
        std_scores = scores.std(1)
        idx_best = avg_scores.argmax()
        l_pen_best = grid_elastic_net[idx_best]
        idx_chosen = max([i for i, j in enumerate(
            list(avg_scores >= avg_scores.max() - std_scores[idx_best])) if j])
        l_pen_chosen = grid_elastic_net[idx_chosen]

        self.grid_elastic_net = grid_elastic_net
        self.l_pen_best = l_pen_best
        self.l_pen_chosen = l_pen_chosen
        self.scores = scores
        self.avg_scores = avg_scores
