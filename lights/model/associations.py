import numpy as np


def get_asso_func(T_u, S, theta, asso_functions, n_long_features,
                  fixed_effect_time_order, derivative=False):
    """Computes association functions or derivatives association ones

    Parameters
    ----------
    T_u : `np.ndarray`, shape=(J,)
        The J unique censored times of the event of interest

    S : `np.ndarray`, shape=(2*N, r)
        Set of constructed Monte Carlo samples

    theta : `dict`
        Vector that concatenates all parameters to be inferred in the lights
        model

    asso_functions : `list` or `str`='all'
        List of association functions wanted or string 'all' to select all
        defined association functions. The available functions are :
            - 'lp' : linear predictor
            - 're' : random effects
            - 'tps' : time dependent slope
            - 'ce' : cumulative effects

    n_long_features : `int`
        Number of longitudinal features

    fixed_effect_time_order :
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    derivative : `bool`, default=False
    If `False`, returns the association functions, otherwise returns the
    derivative versions

    Returns
    -------
    asso_func_stack : `np.ndarray`, shape=(K, J, N_MC, dim)
        Stack version of association functions or derivatives for all
        subjects, all groups and all Monte Carlo samples. `dim` is the
        total dimension of returned association functions.
    """
    fixed_effect_coeffs = np.array([theta["beta_0"], theta["beta_1"]])
    J, N_MC = T_u.shape[0], S.shape[0]
    K = 2 # 2 latent groups
    asso_func = AssociationFunctions(T_u, S, fixed_effect_coeffs,
                                     fixed_effect_time_order, n_long_features)

    if derivative:
        asso_func_stack = np.empty(shape=(K, N_MC, J, n_long_features, 0))
    else:
        asso_func_stack = np.empty(shape=(K, J, N_MC, 0))

    for func_name in asso_functions:
        if derivative: func_name = "d_" + func_name
        func = asso_func.assoc_func_dict[func_name]
        asso_func_stack = np.concatenate((asso_func_stack, func), axis=-1)

    return asso_func_stack


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
    def __init__(self, T_u, S, fixed_effect_coeffs, fixed_effect_time_order=5,
                 n_long_features=5):
        self.K = 2  # 2 latent groups
        self.S, self.J = S, len(T_u)
        self.fixed_effect_coeffs = fixed_effect_coeffs
        self.n_long_features = n_long_features
        self.N_MC = len(S)
        self.q_l = fixed_effect_time_order + 1
        J = self.J

        # U, integral over U, derivative of U
        U_l, iU_l, dU_l = np.ones(J), T_u, np.zeros(J)
        for t in range(1, self.q_l):
            U_l = np.c_[U_l, T_u ** t]
            iU_l = np.c_[iU_l, (T_u ** (t + 1)) / (t + 1)]
            dU_l = np.c_[dU_l, t * T_u ** (t - 1)]

        V_l = np.c_[np.ones(J), T_u]
        iV_l = np.c_[T_u, (T_u ** 2) / 2]
        dV_l = np.c_[np.zeros(J), np.ones(J)]

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
        U : `np.ndarray`, shape=(J, q_l)
            Fixed-effect design features

        V : `np.ndarray`, , shape=(J, r_l)
            Random-effect design features

        Returns
        -------
        phi : `np.ndarray`, shape=(K, J, N_MC, n_long_features)
            The value of linear association function
        """
        beta = self.fixed_effect_coeffs
        J = self.J
        n_long_features = self.n_long_features
        S, N_MC, q_l, K = self.S, self.N_MC, self.q_l, self.K
        r_l = 2  # affine random effects
        phi = np.zeros(shape=(K, J, n_long_features, N_MC))

        for l in range(n_long_features):
            tmp = V.dot(S[:, r_l * l: r_l * (l + 1)].T)
            beta_l = beta[:, q_l * l: q_l * (l + 1)]
            phi[:, :, l] = U.dot(beta_l).T.reshape(K, -1, 1) + tmp
        phi = phi.swapaxes(2, 3)

        return phi

    def linear_predictor(self):
        """Computes the linear predictor function

        Returns
        -------
        phi : `np.ndarray`, shape=(K, J, N_MC, n_long_features)
            The value of linear predictor function
        """
        U_l, V_l = self.U_l, self.V_l
        phi = self._linear_association(U_l, V_l)
        return phi

    def random_effects(self):
        """ Computes the random effects function

        Returns
        -------
        phi : `np.ndarray`, shape=(K, J, N_MC, r)
            The value of random effects function
        """
        S, K, J = self.S, self.K, self.J
        phi = np.broadcast_to(S, ((K, J) + S.shape))
        return phi

    def time_dependent_slope(self):
        """Computes the time-dependent slope function

        Returns
        -------
        phi : `np.ndarray`, shape=(K, J, N_MC, n_long_features)
            The value of time-dependent slope function
        """
        dU_l, dV_l = self.dU_l, self.dV_l
        phi = self._linear_association(dU_l, dV_l)
        return phi

    def cumulative_effects(self):
        """Computes the cumulative effects function

        Returns
        -------
        phi : `np.ndarray`, shape=(K, J, N_MC, n_long_features)
            The value of cumulative effects function
        """
        iU_l, iV_l = self.iU_l, self.iV_l
        phi = self._linear_association(iU_l, iV_l)
        return phi

    def derivative_random_effects(self):
        """ Computes the derivative of the random effects function

        Returns
        -------
        d_phi : `np.ndarray`, shape=(K, N_MC, J, n_long_features, 2 * q_l)
            The value of the derivative of the random effects function
        """
        J = self.J
        n_long_features = self.n_long_features
        N_MC, q_l, K = self.N_MC, self.q_l, self.K
        d_phi = np.zeros(shape=(K, N_MC, J, n_long_features, 2 * q_l))
        return d_phi

    def _get_derivative(self, val):
        """Formats the derivative based on its value

        Parameters
        ----------
        val : `np.ndarray`
            Value of the derivative

        Returns
        -------
        d_phi : `np.ndarray`, shape=(K, N_MC, J, n_long_features, q_l)
            The derivative broadcast to the right shape
        """
        n_long_features = self.n_long_features
        N_MC,  K = self.N_MC, self.K
        U = np.broadcast_to(val, (n_long_features,) + val.shape).swapaxes(0, 1)
        d_phi = np.broadcast_to(U, (K, N_MC) + U.shape)
        return d_phi

    def derivative_linear_predictor(self):
        """Computes the derivative of the linear predictor function

        Returns
        -------
        output : `np.ndarray`, shape=(K, N_MC, J, n_long_features, q_l)
            The value of derivative of the linear predictor function
        """
        return self._get_derivative(self.U_l)

    def derivative_time_dependent_slope(self):
        """Computes the derivative of the time-dependent slope function

        Returns
        -------
        output : `np.ndarray`, shape=(K, N_MC, J, n_long_features, q_l)
            The value of the derivative of the time-dependent slope function
        """
        return self._get_derivative(self.dU_l)

    def derivative_cumulative_effects(self):
        """Computes the derivative of the cumulative effects function

        Returns
        -------
        output : `np.ndarray`, shape=(K, N_MC, J, n_long_features, q_l)
            The value of the derivative of the cumulative effects function
        """
        return self._get_derivative(self.iU_l)
