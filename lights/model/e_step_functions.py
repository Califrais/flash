# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
from scipy.stats import multivariate_normal
from lights.model.associations import get_asso_func


class EstepFunctions:
    """A class to define functions relative to the E-step of the QNMCEM

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

    T : `np.ndarray`, shape=(n_samples,)
        The censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        The censoring indicator

    extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

    n_time_indep_features : `int`
        Number of time-independent features

    n_long_features : `int`
        Number of longitudinal features

    fixed_effect_time_order : `int`
            Order of the higher time monomial considered for the representations
             of the time-varying features corresponding to the fixed effect. The
            dimension of the corresponding design matrix is then equal to
            fixed_effect_time_order + 1

    asso_functions : `list` or `str`='all'
            List of association functions wanted or string 'all' to select all
            defined association functions. The available functions are :
                - 'lp' : linear predictor
                - 're' : random effects
                - 'tps' : time dependent slope
                - 'ce' : cumulative effects

    theta : `dict`, default=None
        Vector that concatenates all parameters to be inferred in the lights
        model
    """

    def __init__(self, X, T, delta, extracted_features, n_long_features,
                 n_time_indep_features, fixed_effect_time_order,
                 asso_functions, theta=None):
        self.K = 2  # 2 latent groups
        self.X, self.T, self.delta = X, T, delta
        self.T_u, self.n_samples = np.unique(T), len(T)
        self.extracted_features, self.theta = extracted_features, theta
        self.n_long_features = n_long_features
        self.n_time_indep_features = n_time_indep_features
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions

    def construct_MC_samples(self, N):
        """Constructs the set of samples used for Monte Carlo approximation

        Returns
        -------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples
        """
        D = self.theta["long_cov"]
        C = np.linalg.cholesky(D)
        r = D.shape[0]
        Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N)
        b = Omega.dot(C.T)
        S = np.vstack((b, -b))
        return S

    def f_data_given_latent(self, S, indicator_1, indicator_2):
        """Computes f(Y, T, delta| S, G, theta)

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        indicator_1 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        indicator_2 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of the f(Y, T, delta| S, G, theta)
        """
        def intensity():
            return (tmp * ind_1).sum(axis=-1)

        def survival():
            return np.exp(-(tmp * ind_2).sum(axis=-1).T)

        def f_y_given_latent():
            f_y = np.ones(shape=(n_samples, K, N_MC))
            g3 = self.g3(S)
            for i in range(n_samples):
                n_i, y_i, M_iS = sum(N_list[i]), y_list[i], g3[i]
                inv_Phi_i = [[phi[l, 0]] * N_list[i][l] for l in
                             range(n_long_features)]
                inv_Phi_i = np.concatenate(inv_Phi_i).reshape(-1, 1)
                f_y[i] = (1 / (np.sqrt(((2 * np.pi) ** n_i) *
                        np.prod(inv_Phi_i))) * np.exp(np.sum(-0.5 *
                        ((y_i - M_iS) ** 2) / inv_Phi_i, axis=1)))
            return f_y

        X, extracted_features = self.X, self.extracted_features
        T, T_u, delta = self.T, self.T_u, self.delta
        theta, K = self.theta, self.K
        n_samples, n_long_features = self.n_samples, self.n_long_features
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        (U_list, V_list, y_list, N_list) = extracted_features[0]
        N_MC = S.shape[0]
        g1 = self.g1(S, False)
        ind_1, ind_2 = indicator_1 * 1, indicator_2 * 1

        baseline_val = baseline_hazard.values.flatten()
        tmp = g1.swapaxes(0, 2) * baseline_val
        op1 = (intensity() ** delta).T
        op2 = survival()
        f_y = f_y_given_latent()
        f = op1 * op2 * f_y
        return f

    def g0(self, S):
        """Computes g0

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g0 : `np.ndarray`, shape=(n_samples, K, N_MC,)
            The values of g0 function
        """
        tmp = np.array([s.reshape(-1, 1).dot(s.reshape(-1, 1).T) for s in S])
        g0 = np.broadcast_to(tmp, (self.n_samples, self.K) + tmp.shape)
        return g0

    def g0_l(self, S):
        """Computes g0_tide

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g0_l : `np.ndarray`,
            shape=(n_samples, K, N_MC, n_long_features, r_l, r_l)
            The values of g0_l function
        """
        N_MC = S.shape[0]
        n_long_features = self.n_long_features
        S_ = S.reshape(N_MC, n_long_features, -1)
        tmp = []
        for s in S_:
            tmp.append(np.array([s_.reshape(-1, 1).dot(s_.reshape(-1, 1).T)
                                 for s_ in s]))
        tmp = np.array(tmp)
        g0_l = np.broadcast_to(tmp, (self.n_samples, self.K) + tmp.shape)
        return g0_l

    def gS(self, S):
        """Computes gS

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        gS : `np.ndarray`, shape=(n_samples, K, N_MC, r)
            The values of gS function
        """
        gS = np.broadcast_to(S, (self.n_samples, self.K) + S.shape)
        return gS

    def g1(self, S, broadcast=True):
        """Computes g1

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        broadcast : `boolean`
            Indicates to expand the dimension of g1 or not

        Returns
        -------
        g1 : `np.ndarray`, shape=(n_samples, K, N_MC, J, K)
            The values of g1 function
        """
        n_samples, K, theta = self.n_samples, self.K, self.theta
        p = self.n_time_indep_features
        X, T_u = self.X, self.T_u
        N_MC, J = S.shape[0], T_u.shape[0]
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_indep = np.hstack((gamma_0[:p], gamma_1[:p]))
        g2_ = self.g2(S, broadcast=False).reshape(K, 1, J, N_MC)
        tmp = X.dot(gamma_indep).T.reshape(K, n_samples, 1, 1)
        g1 = np.exp(tmp + g2_).swapaxes(0, 1).swapaxes(2, 3)
        if broadcast:
            g1 = np.broadcast_to(g1[..., None], g1.shape + (2,)).swapaxes(1, -1)
        return g1

    def g2(self, S, indicator=None, broadcast=True):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        broadcast : `boolean`
            Indicate to expand the dimension of g2 or not

        Returns
        -------
        g2 : `np.ndarray`, shape=(K, J, N_MC)
            The values of g2 function
        """
        T_u, p, K = self.T_u, self.n_time_indep_features, self.K
        asso_functions, L = self.asso_functions, self.n_long_features
        alpha = self.fixed_effect_time_order
        theta = self.theta
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_dep = np.vstack((gamma_0[p:], gamma_1[p:])).reshape((K, 1, 1, -1))
        asso_func = get_asso_func(T_u, S, theta, asso_functions, L, alpha)
        g2 = np.sum(asso_func * gamma_dep, axis=-1)
        if broadcast:
            g2 = g2.swapaxes(0, 1)
            g2 = np.sum(np.broadcast_to(g2, (self.n_samples,) + g2.shape).T *
                        (indicator * 1).T, axis=2).T
            g2 = np.broadcast_to(g2[..., None], g2.shape + (2,)).swapaxes(1, 3)
        return g2

    def g3(self, S):
        """Computes g3

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
                Set of constructed Monte Carlo samples

        Returns
        -------
        g3 : list, size=(n_samples, np.array)
                The values of g3 function
        """
        n_samples, n_long_features = self.n_samples, self.n_long_features
        extracted_features = self.extracted_features
        beta_0, beta_1 = self.theta["beta_0"], self.theta["beta_1"]
        phi, K, N_MC = self.theta["phi"], self.K, S.shape[0]
        (U_list, V_list, y_list, N_list) = extracted_features[0]
        g3 = []
        beta_stack = np.hstack((beta_0, beta_1))
        for i in range(n_samples):
            U_i, V_i, y_i = U_list[i], V_list[i], y_list[i]
            M_iS = U_i.dot(beta_stack).T.reshape(K, -1, 1) + V_i.dot(S.T)
            g3.append(M_iS)
        return g3

    def g5(self, S, indicator=None, broadcast=True):
        """Computes g5

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        indicator : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times

        broadcast : `boolean`
            Indicate to expand the dimension of g1 or not

        Returns
        -------
        g5 : `np.ndarray`, shape=(K, N_MC, J, n_long_features, A*r_l)
            The values of g5 function
        """
        g5 = get_asso_func(self.T_u, S, self.theta, self.asso_functions,
                           self.n_long_features, self.fixed_effect_time_order,
                           derivative=True)

        if broadcast:
            g5 = g5.swapaxes(0, 2)
            g5 = np.sum(np.broadcast_to(g5, (self.n_samples,) + g5.shape).T *
                        (indicator * 1).T, axis=-2).T
            g5 = np.broadcast_to(g5[..., None], g5.shape + (2,))\
                .swapaxes(2, 5).swapaxes(1, 2)

        return g5

    def g6(self, S):
        """Computes g6

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g6 : `np.ndarray`, shape=(n_samples, K, N_MC, J, n_long_features, A*r_l)
            The values of g6 function
        """
        n_samples = self.n_samples
        g1, g5 = self.g1(S, False), self.g5(S, broadcast=False)
        g5 = np.broadcast_to(g5, (n_samples,) + g5.shape)
        g6 = (g1.T * g5.T).T
        g6 = np.broadcast_to(g6[..., None], g6.shape + (2,)).swapaxes(1, -1)
        return g6

    def g7(self, S, broadcast=True):
        """Computes g7

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        broadcast : `boolean`
            Indicate to expand the dimension of g1 or not

        Returns
        -------
        g7 : `np.ndarray`, shape=(K, N_MC, J, dim, K)
            The values of g7 function
        """
        T_u, theta = self.T_u, self.theta
        asso_func, L = self.asso_functions, self.n_long_features
        alpha = self.fixed_effect_time_order
        g7 = get_asso_func(T_u, S, theta, asso_func, L, alpha).swapaxes(1, 2)
        if broadcast:
            g7 = np.broadcast_to(g7, (self.n_samples,) + g7.shape)
            g7 = np.broadcast_to(g7[..., None], g7.shape + (2,)).swapaxes(1, -1)
        return g7

    def g8(self, S):
        """Computes g8
        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g8 : `np.ndarray`, shape=(n_samples, K, N_MC, J, dim, K)
            The values of g8 function
        """

        g7 = self.g7(S, False)
        g1 = self.g1(S, False)
        g8 = g1[..., np.newaxis] * g7
        g8 = np.broadcast_to(g8[..., None], g8.shape + (2,)).swapaxes(1, -1)
        return g8

    def g9(self, S):
        """Computes g9

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g9 : `np.ndarray`, shape=(n_samples, K, N_MC, K)
            The values of g9 function
        """
        n_samples, n_long_features = self.n_samples, self.n_long_features
        extracted_features = self.extracted_features
        beta_0, beta_1 = self.theta["beta_0"], self.theta["beta_1"]
        phi, K, N_MC = self.theta["phi"], self.K, S.shape[0]
        (U_list, V_list, y_list, N_list) = extracted_features[0]

        g3 = self.g3(S)
        g9 = np.zeros(shape=(n_samples, K, N_MC))
        for i in range(n_samples):
            y_i, M_iS = y_list[i], g3[i]
            Phi_i = [[1 / phi[l, 0]] * N_list[i][l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            g9[i] = np.sum(M_iS * y_i * Phi_i - .5 * (M_iS ** 2) * Phi_i, axis=1)

        g9 = np.broadcast_to(g9[..., None], g9.shape + (2,)).swapaxes(1, -1)

        return g9

    @staticmethod
    def Lambda_g(g, f):
        """Approximated integral (see (15) in the lights paper)

        Parameters
        ----------
        g : `np.ndarray`, shape=(n_samples, K, N_MC, ...)
            Values of g function for all subjects, all groups and all Monte
            Carlo samples. Each element could be real or matrices depending on
            Im(\tilde{g}_i)

        f: `np.ndarray`, shape=(n_samples, K, N_MC)
            Values of the density of the observed data given the latent ones and
            the current estimate of the parameters, computed for all subjects,
            all groups and all Monte Carlo samples

        Returns
        -------
        Lambda_g : `np.array`, shape=(n_samples, K)
            The approximated integral computed for all subjects, all groups and
            all Monte Carlo samples. Each element could be real or matrices
            depending on Im(\tilde{g}_i)
        """
        Lambda_g = np.mean((g.T * f.T).T, axis=2)
        return Lambda_g

    def Eg(self, g, Lambda_1, pi_xi, f):
        """Computes approximated expectations of different functions g taking
        random effects as input, conditional on the observed data and the
        current estimate of the parameters. See (14) in the lights paper

        Parameters
        ----------
        g : `np.array`, shape=(n_samples, g.shape)
            The value of g function for all samples

        Lambda_1: `np.ndarray`, shape=(n_samples, K)
            Approximated integral (see (15) in the lights paper) with
            \tilde(g)=1

        pi_xi: `np.ndarray`, shape=(n_samples,)
            The posterior probability of the sample for being on the
            high-risk group given all observed data

        f: `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of the f(Y, T, delta| S, G, theta)

        Returns
        -------
        Eg : `np.ndarray`, shape=(n_samples, g.shape)
            The approximated expectations for g
        """
        Lambda_g = self.Lambda_g(g, f)
        Eg = (Lambda_g[:, 0].T * (1 - pi_xi) + Lambda_g[:, 1].T * pi_xi) / (
                    Lambda_1[:, 0] * (1 - pi_xi) + Lambda_1[:, 1] * pi_xi)
        return Eg.T
