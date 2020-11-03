
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