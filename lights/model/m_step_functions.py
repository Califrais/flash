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
        If `True`, include an intercept in the model for the time independent
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
                 n_time_indep_features, l_pen, eta_elastic_net, eta_sp_gp_l1, nb_asso_features,
                 fixed_effect_time_order):
        self.fit_intercept = fit_intercept
        self.X = X
        self.T = T
        self.delta = delta
        self.n_long_features = n_long_features
        self.n_time_indep_features = n_time_indep_features
        n_samples = len(T)
        self.n_samples = n_samples
        self.nb_asso_features = nb_asso_features
        self.fixed_effect_time_order = fixed_effect_time_order
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
        sub_obj = (pi_est * logistic_loss(u)).mean()
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

        grad = (X * (pi_est * np.exp(-logistic_loss(-u))).reshape(-1, 1)).mean(axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def R_func(self, beta_ext, pi_est, E_g1, E_g2, E_g8, baseline_hazard,
               indicator, idx):
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

        idx : `int`
            Index of latent groups

        Returns
        -------
        output : `float`
            The value of the R sub objective to be minimized at each QNMCEM step
        """
        n_samples = self.n_samples
        pen = self.pen.sparse_group_l1(beta_ext)
        E_g1_, E_g2_, E_g8_ = E_g1[:, :, idx], E_g2[:, idx], E_g8[:, idx]
        delta_ = self.delta
        baseline_val = baseline_hazard.values.flatten()
        ind_ = indicator * 1
        sub_obj =  E_g2_ * delta_ + E_g8_ - (E_g1_ * baseline_val * ind_).sum(axis=1)
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples + pen

    def grad_R(self, beta_ext, gamma_ext, pi_est, E_g5, E_g6, E_gS, baseline_hazard,
               indicator, extracted_features, phi, idx):
        """Computes the gradient of the sub objective R

        Parameters
        ----------
        gamma_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        beta_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        E_g5 : `np.ndarray`, shape=(n, J, n_long_features, dim, 2)
            The approximated expectations of function logarithm of g5

        E_g6 : `np.ndarray`, shape=(n_samples, J, n_long_features, dim, 2)
            The approximated expectations of function g6

        baseline_hazard : `np.ndarray`, shape=(n_samples,)
            The baseline hazard function evaluated at each censored time

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        phi : `np.ndarray`, shape=(n_long_features,)
            Variance vector for the error term of the longitudinal processes

        idx: `int`
            Index of latent groups

        Returns
        -------
        output : `float`
            The value of the R sub objective gradient
        """
        n_time_indep_features = self.n_time_indep_features
        n_long_features = self.n_long_features
        n_samples = self.n_samples
        q_l =  self.fixed_effect_time_order + 1

        beta = get_vect_from_ext(beta_ext)
        gamma = get_vect_from_ext(gamma_ext)[n_time_indep_features:].reshape(n_long_features, -1)
        # To match the derivative of association functions over the beta
        gamma_ = np.repeat(gamma, q_l, axis=1)
        grad_pen = self.pen.grad_sparse_group_l1(beta, n_long_features)

        E_g5_, E_g6_ = E_g5.T[idx].T, E_g6.T[idx].T
        ind_ = indicator * 1
        baseline_val = baseline_hazard.values
        tmp1 = (E_g5_.T * self.delta).T - (E_g6_.T * (ind_ * baseline_val).T).T.sum(axis=1)
        # split and sum over each l-th beta
        tmp1 = (tmp1 * gamma_).reshape(n_samples, n_long_features, -1, q_l).sum(axis=2)

        (U_list, V_list, y_list, N_list) = extracted_features[0]
        tmp2 = np.zeros((n_samples, n_long_features * q_l))
        for i in range(n_samples):
            U_i, V_i, n_i, y_i = U_list[i], V_list[i], N_list[i], y_list[i].flatten()
            Phi_i = [[phi[l, 0]] * n_i[l] for l in range(n_long_features)]
            Phi_i = np.diag(np.concatenate(Phi_i))
            tmp2[i] = U_i.T.dot(Phi_i.dot(y_i - U_i.dot(beta) - V_i.dot(E_gS[i])))

        grad = ((tmp1.reshape(n_samples, -1) + tmp2).T * pi_est).sum(axis=1)
        grad_sub_obj = np.concatenate([grad, -grad])
        return -grad_sub_obj / n_samples + grad_pen

    def Q_func(self, gamma_ext, pi_est, E_log_g1, E_g1, baseline_hazard,
               indicator_1, indicator_2, idx):
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

        indicator_1 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        indicator_2 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        idx: `int`
            Index of latent groups

        Returns
        -------
        output : `float`
            The value of the Q sub objective to be minimized at each QNMCEM step
        """
        n_samples, delta = self.n_samples, self.delta
        n_time_indep_features = self.n_time_indep_features
        n_long_features = self.n_long_features

        # split into 2 latent groups
        gamma_ext_ = gamma_ext.reshape(-1, 2)
        pen = self.pen.sparse_group_l1(gamma_ext_[n_time_indep_features:].flatten())\
              + self.pen.lasso(gamma_ext_[:n_time_indep_features].flatten())
        baseline_val = baseline_hazard.values.flatten()
        ind_1 = indicator_1 * 1
        ind_2 = indicator_2 * 1
        E_g1_, E_log_g1_ = E_g1[:, :, idx], E_log_g1[:, :, idx]
        sub_obj = (E_log_g1_ * ind_1).sum(axis=1) * delta - (E_g1_ * ind_2 * baseline_val).sum(axis=1)
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples + pen

    def grad_Q(self, gamma_ext, pi_est, E_g1, E_g7, E_g8, baseline_hazard, indicator_1, indicator_2, idx):
        """Computes the gradient of the sub objective Q

        Parameters
        ----------
        gamma_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        E_g1 : `np.ndarray`, shape=(n_samples, J, 2)
            The approximated expectations of function g1

        E_g7 : `np.ndarray`, shape=(n_samples, J, dim, 2)
            The approximated expectations of function g7

        E_g8 : `np.ndarray`, shape=(n_samples, J, dim, 2)
            The approximated expectations of function g8

        baseline_hazard : `np.ndarray`, shape=(n_samples,)
            The baseline hazard function evaluated at each censored time

        indicator_1 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        indicator_2 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        idx: `int`
            Index of latent groups

        Returns
        -------
        output : `float`
            The value of the Q sub objective gradient
        """
        n_time_indep_features = self.n_time_indep_features
        nb_asso_features = self.nb_asso_features
        n_samples, delta = self.n_samples, self.delta
        n_long_features = self.n_long_features
        gamma = get_vect_from_ext(gamma_ext)
        gamma_indep, gamma_dep = gamma[:n_time_indep_features], gamma[n_time_indep_features:]
        baseline_val = baseline_hazard.values.flatten()
        ind_1, ind_2 = indicator_1 * 1, indicator_2 * 1
        E_g1_, E_g7_, E_g8_ = E_g1.T[idx].T, E_g7.T[idx].T, E_g8.T[idx].T.swapaxes(0, 1)
        X_ = self.X.flatten()

        grad_pen_indep = self.pen.grad_lasso(gamma_indep).reshape(-1, 2)
        grad_pen_dep = self.pen.grad_sparse_group_l1(gamma_dep, n_long_features).reshape(-1, 2)
        grad_pen = np.vstack((grad_pen_indep, grad_pen_dep))
        grad = np.zeros(nb_asso_features)
        grad[:n_time_indep_features] = (pi_est * (delta - (E_g1_ * baseline_val * ind_2).sum(axis=1)) * X_).sum()
        tmp = (E_g7_.T * delta * ind_1.T).T.sum(axis=1) - (E_g8_.T * baseline_val * ind_2).sum(axis=-1).T
        grad[n_time_indep_features:] = (tmp.swapaxes(0, 1) * pi_est).sum(axis=1)
        grad_sub_obj = np.vstack((grad, -grad)).T
        return (-grad_sub_obj / n_samples + grad_pen).flatten()
