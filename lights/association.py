import numpy as np


class AssociationFunctions:
    """A class to define all the association functions

    Parameters
    ----------
    T : `np.ndarray`, shape=(n_samples,)
        Censored times of the event of interest

    S : `np.ndarray`, shape=(2*N, r)
        Set of samples used for Monte Carlo approximation

    fixed_effect_coeffs : `np.ndarray`,
        shape=((fixed_effect_time_order+1)*n_long_features,)
        Fixed effect coefficient vectors

    fixed_effect_time_order: `int`, default=5
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    n_long_features: `int`, default=5
        Number of longitudinal features
    """
    def __init__(self, T, S, fixed_effect_coeffs, fixed_effect_time_order=5,
                 n_long_features=5):
        self.S = S
        self.fixed_effect_coeffs = fixed_effect_coeffs
        self.n_long_features = n_long_features
        n_samples = len(T)
        self.n_samples = n_samples
        self.N = len(S) // 2
        self.r_l = 2  # linear time-varying features, so all r_l=2
        self.q_l = fixed_effect_time_order + 1

        U_l = np.ones(n_samples)
        # integral over U
        iU_l = T
        # derivative of U
        dU_l = np.zeros(n_samples)
        for t in range(1, self.q_l):
            U_l = np.c_[U_l, T ** t]
            iU_l = np.c_[iU_l, (T ** (t + 1)) / (t + 1)]
            dU_l = np.c_[dU_l, t * T ** (t - 1)]

        V_l = np.c_[np.ones(n_samples), T]
        iV_l = np.c_[T, (T ** 2) / 2]
        dV_l = np.c_[np.zeros(n_samples), np.ones(n_samples)]

        self.U_l, self.iU_l, self.dU_l = U_l, iU_l, dU_l
        self.V_l, self.iV_l, self.dV_l = V_l, iV_l, dV_l

        self.assoc_func_dict = {"lp": self.linear_predictor(),
                                "re": self.random_effects(),
                                "tps": self.time_dependent_slope(),
                                "ce": self.cumulative_effects(),
                                "d_lp": self.derivative_linear_predictor(),
                                "d_re": self.derivative_random_effects(),
                                "d_tps": self.derivative_time_dependent_slope(),
                                "d_ce": self.derivative_cumulative_effects()
                                }

    def _linear_association(self, U, V):
        """ Computes the linear association function U*beta + V*b

        Parameters
        ----------
        U : `np.ndarray`, shape=(n_samples, q_l)
            Fixed-effect design features

        V : `np.ndarray`, , shape=(n_samples, r_l)
            Random-effect design features

        Returns
        -------
        phi : `np.ndarray`, shape=(n_samples, 2, n_long_features, 2*N)
            The value of linear association function

        """
        beta = self.fixed_effect_coeffs
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        S, N, r_l, q_l = self.S, self.N, self.r_l, self.q_l
        phi = np.zeros(shape=(n_samples, 2, n_long_features, 2 * N))

        for l in range(n_long_features):
            tmp = V.dot(S[:, r_l * l: r_l * (l + 1)].T)
            beta_0l = beta[0, q_l * l: q_l * (l + 1)]
            beta_1l = beta[1, q_l * l: q_l * (l + 1)]
            phi[:, 0, l, :] = U.dot(beta_0l) + tmp
            phi[:, 1, l, :] = U.dot(beta_1l) + tmp

        return phi

    def linear_predictor(self):
        """Computes the linear predictor function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_samples, 2, n_long_features, 2*N)
            The value of linear predictor function
        """
        U_l, V_l = self.U_l, self.V_l
        phi = self._linear_association(U_l, V_l)
        return phi

    def random_effects(self):
        """ Computes the random effects function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_samples, 2, r_l*n_long_features, 2*N)
            The value of random effects function
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        S, N, r_l = self.S, self.N, self.r_l
        phi = np.broadcast_to(S.T, (n_samples, 2, r_l * n_long_features, 2 * N))
        return phi

    def time_dependent_slope(self):
        """Computes the time-dependent slope function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_samples, 2, n_long_features, 2*N)
            The value of time-dependent slope function
        """
        dU_l, dV_l = self.dU_l, self.dV_l
        phi = self._linear_association(dU_l, dV_l)
        return phi

    def cumulative_effects(self):
        """Computes the cumulative effects function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_samples, 2, n_long_features, 2*N)
            The value of cumulative effects function
        """
        iU_l, iV_l = self.iU_l, self.iV_l
        phi = self._linear_association(iU_l, iV_l)
        return phi

    def derivative_linear_predictor(self):
        """Computes the derivative of linear predictor function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_long_features, 2, 2*N, n_samples, q_l)
            The value of linear predictor function
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        N, q_l = self.N, self.q_l
        U_l = self.U_l
        U = np.broadcast_to(U_l, (n_long_features,) + U_l.shape).swapaxes(0,1)
        d_phi = np.broadcast_to(U, (2, 2 * N) + U.shape)
        return d_phi

    def derivative_random_effects(self):
        """ Computes the derivative of random effects function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_long_features, 2, 2*N, n_samples, q_l)
            The value of random effects function
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        N, q_l = self.N, self.q_l
        d_phi = np.zeros(shape = (2, 2 * N, n_samples, n_long_features, q_l))
        return d_phi

    def derivative_time_dependent_slope(self):
        """Computes the derivative of time-dependent slope function

        Returns
        -------
        phi : `np.ndarray`, shape= (2, 2*N, n_samples, n_l, q_l)
            The value of time-dependent slope function
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        N, q_l = self.N, self.q_l
        dU_l= self.dU_l
        dU = np.broadcast_to(dU_l, (n_long_features, ) + dU_l.shape).swapaxes(0,1)
        d_phi = np.broadcast_to(dU, (2, 2 * N) + dU.shape)
        return d_phi

    def derivative_cumulative_effects(self):
        """Computes the derivative of cumulative effects function

        Returns
        -------
        phi : `np.ndarray`, shape=(n_long_features, 2, 2*N, n_samples, q_l)
            The value of cumulative effects function
        """
        n_samples = self.n_samples
        n_long_features = self.n_long_features
        N, q_l = self.N, self.q_l
        iU_l = self.iU_l
        iU = np.broadcast_to(iU_l, (n_long_features,) + iU_l.shape).swapaxes(0,1)
        d_phi = np.broadcast_to(iU, (2, 2 * N) + iU.shape)
        return d_phi