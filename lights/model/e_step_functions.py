# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
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
        self.X = X
        self.T = T
        self.T_u = np.unique(T)
        self.delta = delta
        self.extracted_features = extracted_features
        self.theta = theta
        self.n_long_features = n_long_features
        self.n_time_indep_features = n_time_indep_features
        self.fixed_effect_time_order = fixed_effect_time_order
        n_samples = len(T)
        self.n_samples = n_samples
        self.asso_functions = asso_functions

    def construct_MC_samples(self, N):
        """Constructs the set of samples used for Monte Carlo approximation

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

    def f_data_given_latent(self, S):
        """Computes f(Y, T, delta| S, G, theta)

        Parameters
        ----------
        S: `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, 2, N)
            The value of the f(Y, T, delta| S, G, theta)
        """
        X, extracted_features = self.X, self.extracted_features
        T, T_u, delta = self.T, self.T_u, self.delta
        theta = self.theta
        n_samples, n_long_features = self.n_samples, self.n_long_features
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        (U_list, V_list, y_list, N_list) = extracted_features[0]
        N = S.shape[0] // 2
        g1 = self._g1(S)

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

    @staticmethod
    def _g0(S):
        """Computes g0

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g0 : `np.ndarray`, shape=(2*N,)
            The values of g0 function
        """
        g0 = np.array([s.reshape(-1, 1).dot(s.reshape(-1, 1).T) for s in S])
        return g0

    def _g0_t(self, S):
        """Computes g0_tide

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g0 : `np.ndarray`, shape=(2*N, n_long_features, q_l, q_l)
            The values of g0_t function
        """
        N = S.shape[0] // 2
        n_long_features = self.n_long_features
        S_ = S.reshape(2*N, n_long_features, -1)
        g0_t = []
        for s in S_:
            g0_t.append(np.array([s_.reshape(-1, 1).dot(s_.reshape(-1, 1).T) for s_ in s]))
        return np.array(g0_t)


    def _g1(self, S):
        """Computes g1

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g1 : `np.ndarray`, shape=(n_samples, 2, 2*N, J)
            The values of g1 function
        """
        n_samples = self.n_samples
        n_time_indep_features = self.n_time_indep_features
        theta = self.theta
        X, T_u = self.X, self.T_u
        N = S.shape[0] // 2
        J = T_u.shape[0]
        p = n_time_indep_features
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_indep_stack = np.vstack((gamma_0[:p], gamma_1[:p])).T
        g2 = self._g2(S)
        tmp = X.dot(gamma_indep_stack)
        g1 = np.exp(
            tmp.T.reshape(2, n_samples, 1, 1) + g2.reshape(2, 1, J, 2 * N))
        g1 = g1.swapaxes(0, 1).swapaxes(2, 3)
        return g1

    def _g2(self, S):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g2 : `np.ndarray`, shape=(2, J, 2*N)
            The values of g2 function
        """
        T_u, theta = self.T_u, self.theta
        p = self.n_time_indep_features
        N = S.shape[0] // 2
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        asso_func = get_asso_func(T_u, S, theta, self.asso_functions,
                                  self.n_long_features,
                                  self.fixed_effect_time_order)
        gamma_time_depend_stack = np.vstack((gamma_0[p:],
                                             gamma_1[p:])).reshape((2, 1, -1))
        J = T_u.shape[0]
        g2 = np.sum(asso_func * gamma_time_depend_stack,
                    axis=2).reshape((2, J, 2 * N))
        return g2

    def _g5(self, S):
        """Computes g5

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g5 : `np.ndarray`, shape=(2, 2 * N, J, n_long_features, q_l)
            The values of g5 function
        """
        g5 = get_asso_func(self.T_u, S, self.theta, self.asso_functions,
                           self.n_long_features, self.fixed_effect_time_order,
                           derivative=True)
        return g5

    def _g6(self, S):
        """Computes g6

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g6 : `np.ndarray`, shape=(n_samples, n_long_features, 2, 2 * N * J, q_l)
            The values of g6 function
        """
        n_samples = self.n_samples
        g1, g5 = self._g1(S), self._g5(S)
        g5 = np.broadcast_to(g5, (n_samples,) + g5.shape)
        g6 = (g1.T * g5.T).T
        return g6

    def _g7(self, S):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g7 : `np.ndarray`, shape=(2, 2*N, J, dim)
            The values of g7 function
        """
        T_u, theta = self.T_u, self.theta
        J = T_u.shape[0]
        N = S.shape[0] // 2
        g7 = get_asso_func(T_u, S, theta, self.asso_functions,
                                  self.n_long_features,
                                  self.fixed_effect_time_order)\
            .reshape(2, J, 2 * N, -1).swapaxes(1, 2)
        return g7


    def _g8(self, S):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g8 : `np.ndarray`, shape=(n_samples, 2, 2*N, J, dim)
            The values of g8 function
        """
        T_u, theta = self.T_u, self.theta
        J = T_u.shape[0]
        N = S.shape[0] // 2
        asso_func = get_asso_func(T_u, S, theta, self.asso_functions,
                                  self.n_long_features,
                                  self.fixed_effect_time_order)\
            .reshape(2, J, 2 * N, -1).swapaxes(1, 2)

        g7 = self._g7(S)
        g1 = self._g1(S)
        g8 = g1.reshape(g1.shape + (1,)) * g7
        return g8

    def _g9(self, S):
        """Computes g8

        Parameters
        ----------
        S : `np.ndarray`, shape=(2*N, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g9 : `np.ndarray`, shape=(n_samples, 2, 2 * N)
            The values of g9 function
        """
        n_samples, n_long_features = self.n_samples, self.n_long_features
        extracted_features = self.extracted_features
        theta = self.theta
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        phi = theta["phi"]
        (U_list, V_list, y_list, N_list) = extracted_features[0]

        g9 = np.zeros(shape=(n_samples, 2, S.shape[0]))
        for i in range(n_samples):
            beta_stack = np.hstack((beta_0, beta_1))
            U_i = U_list[i]
            V_i = V_list[i]
            y_i = y_list[i]
            Phi_i = [[phi[l, 0]] * N_list[i][l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            M_iS = U_i.dot(beta_stack).T.reshape(2, -1, 1) + V_i.dot(S.T)
            g9[i] = np.sum(M_iS * y_i * Phi_i + (M_iS ** 2) * Phi_i, axis=1)

        return g9

    @staticmethod
    def Lambda_g(g, f):
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

    def _Eg(self, g, Lambda_1, pi_xi, f):
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

        pi_xi:

        f:

        Returns
        -------
        Eg : `np.ndarray`, shape=(n_samples,)
            The approximated expectations for g
        """
        Lambda_g = self.Lambda_g(g, f)
        Eg = (Lambda_g[:, 0].T * (1 - pi_xi) + Lambda_g[:, 1].T * pi_xi) / (
                    Lambda_1[:, 0] * (1 - pi_xi) + Lambda_1[:, 1] * pi_xi)
        return Eg.T
