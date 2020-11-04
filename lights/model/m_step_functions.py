# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
from lights.base.base import get_vect_from_ext, logistic_loss, logistic_grad, \
    get_xi_from_xi_ext
from lights.model.regularizations import Penalties


class MstepFunctions:
    """A class to define functions relative to the M-step of the QNMCEM

    Parameters
    ----------
    fit_intercept : `bool`
        If `True`, include an intercept in the model for the time independant
        features

    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

    T : `np.ndarray`, shape=(n_samples,)
        The censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        The censoring indicator

    n_time_indep_features : `int`
        Number of time-independent features

    n_long_features : `int`
        Number of longitudinal features

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
    """
    def __init__(self, fit_intercept, X, T, delta, n_long_features,
                 n_time_indep_features, l_pen, eta_elastic_net, eta_sp_gp_l1):
        self.fit_intercept = fit_intercept
        self.X = X
        self.T = T
        self.delta = delta
        self.n_long_features = n_long_features
        self.n_time_indep_features = n_time_indep_features
        n_samples = len(T)
        self.n_samples = n_samples
        self.pen = Penalties(fit_intercept, l_pen, eta_elastic_net,
                             eta_sp_gp_l1)

        # TODO: MstepFunctions class does not depend on theta : careful !

    def P_func(self, pi_est, xi_ext):
        """Computes the sub objective function denoted P in the lights paper,
        to be minimized at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
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
        xi_0, xi = get_xi_from_xi_ext(xi_ext, self.fit_intercept)
        pen = self.pen.elastic_net(xi_ext)
        u = xi_0 + self.X.dot(xi)
        sub_obj = (pi_est * u + logistic_loss(u)).mean()
        return sub_obj + pen

    def grad_P(self, pi_est, xi_ext):
        """Computes the gradient of the sub objective P

        Parameters
        ----------
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
        fit_intercept = self.fit_intercept
        X, n_samples = self.X, self.n_samples
        n_time_indep_features = self.n_time_indep_features
        xi_0, xi = get_xi_from_xi_ext(xi_ext, fit_intercept)
        grad_pen = self.pen.grad_elastic_net(xi)
        u = xi_0 + X.dot(xi)
        if fit_intercept:
            X = np.concatenate((np.ones(n_samples).reshape(1, n_samples).T, X),
                               axis=1)
            grad_pen = np.concatenate([[0], grad_pen[:n_time_indep_features],
                                       [0], grad_pen[n_time_indep_features:]])
        grad = (X * (pi_est - logistic_grad(-u)).reshape(
            n_samples, 1)).mean(axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def R_func(self, beta_ext, pi_est, E_g1, E_g2, E_g8, baseline_hazard,
                indicator):
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

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        Returns
        -------
        output : `float`
            The value of the R sub objective to be minimized at each QNMCEM step
        """
        n_samples = self.n_samples
        pen = self.pen.sparse_group_l1(beta_ext)
        E_g1_ = E_g1.swapaxes(1, 2).swapaxes(0, 1)
        baseline_val = baseline_hazard.values.flatten()
        ind_ = indicator * 1
        sub_obj = E_g2 * self.delta.reshape(-1, 1) + E_g8 - np.sum(
            E_g1_ * baseline_val * ind_, axis=2).T
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples + pen

    def grad_R(self, beta_ext, E_g5, E_g6, baseline_hazard, indicator):
        """Computes the gradient of the sub objective R

        Parameters
        ----------
        # TODO Van Tuan

        Returns
        -------
        output : `float`
            The value of the R sub objective gradient
        """
        beta = get_vect_from_ext(beta_ext)
        grad_pen = self.pen.grad_sparse_group_l1(beta, self.n_long_features)
        # TODO: handle gamma
        tmp1 = (E_g5.T * self.delta).T - np.sum(
            (E_g6.T * baseline_hazard.values * (indicator * 1).T).T, axis=1)
        tmp2 = 0
        grad = tmp1 + tmp2
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def Q_func(self, gamma_ext, pi_est, E_log_g1, E_g1, baseline_hazard,
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

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        Returns
        -------
        output : `float`
            The value of the Q sub objective to be minimized at each QNMCEM step
        """
        n_samples, delta = self.n_samples, self.delta
        pen = self.pen.sparse_group_l1(gamma_ext)
        E_g1_ = E_g1.swapaxes(1, 2).swapaxes(0, 1)
        baseline_val = baseline_hazard.values.flatten()
        ind_ = indicator * 1
        sub_obj = E_log_g1 * delta.reshape(-1, 1) - np.sum(
            E_g1_ * baseline_val * ind_, axis=2).T
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples + pen

    def grad_Q(self, gamma_ext):
        """Computes the gradient of the sub objective Q

        Parameters
        ----------
        # TODO Van Tuan

        Returns
        -------
        output : `float`
            The value of the Q sub objective gradient
        """
        gamma = get_vect_from_ext(gamma_ext)
        grad_pen = self.pen.grad_sparse_group_l1(gamma, self.n_long_features)
        # TODO Van Tuan
        grad = 0
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen
