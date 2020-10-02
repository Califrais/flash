import numpy as np

class AssociationFunctions:
    """A class to define all the association functions

    Parameters
    ----------
    T : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

    S : `np.ndarray`, , shape=(2*N, r)
            Set of constructed samples

    beta: `np.ndarray`, shape=(2, (fixed_effect_time_order+1)*n_long_features,)
            Fixed effect coefficient vectors

    fixed_effect_time_order: `int`

    n_long_features: `int`
    """

    def __init__(self, T, S, beta, fixed_effect_time_order, n_long_features):
        self.T = T
        self.S = S
        self.beta = beta
        self.fixed_effect_time_order = fixed_effect_time_order
        self.n_long_features = n_long_features
        self.q_l = self.fixed_effect_time_order + 1
        self.r_l = 2  # linear time-varying features, so all r_l=2
        self.n_samples = len(self.T)
        self.N = len(self.S)

    def phi_1(self):
        """Compute the linear predictor function (first function in Table1)
        """

        phi = np.array(shape=(self.n_samples, 2, self.n_long_features, self.N))

        U_l = np.ones(self.n_samples)
        for t in range(1, self.q_l):
            U_l = np.c_[U_l, self.T ** t]
        # linear time-varying features
        V_l = np.c_[np.ones(self.n_samples), self.T]

        for l in range(self.n_long_features):
            tmp = V_l.dot(self.S[:, self.r_l * l : self.r_l * (l + 1)].T)
            beta_0l = self.beta[0, self.q_l * l: self.q_l * (l + 1)]
            beta_1l = self.beta[1, self.q_l * l: self.q_l * (l + 1)]
            phi[:, 0, l, :] = U_l.dot(beta_0l) + tmp
            phi[:, 1, l, :] = U_l.dot(beta_1l) + tmp

        return phi

    def phi_2(self):
        """ Compute the random effects function (second function in Table1)
        """

        phi = np.broadcast_to(self.S.T, (self.n_samples, 2,
                                         self.r_l*self.n_long_features, self.N))

        return phi

    def phi_3(self):
        """Compute the time-dependent slope function (third function in Table1)
        """

        phi = np.array(shape=(self.n_samples, 2, self.n_long_features, self.N))

        # derivative of U
        dU_l = np.zeros(self.n_samples)
        for t in range(1, self.q_l):
            dU_l = np.c_[dU_l, t*self.T ** (t-1)]

        dV_l = np.c_[np.zeros(self.n_samples), np.ones(self.n_samples)]

        for l in range(self.n_long_features):
            tmp = dV_l.dot(self.S[:, self.r_l * l: self.r_l * (l + 1)].T)
            beta_0l = self.beta[0, self.q_l * l: self.q_l * (l + 1)]
            beta_1l = self.beta[1, self.q_l * l: self.q_l * (l + 1)]
            phi[:, 0, l, :] = dU_l.dot(beta_0l) + tmp
            phi[:, 1, l, :] = dU_l.dot(beta_1l) + tmp

        return phi

    def phi_4(self):
        """Compute the cumulative effect function (forth function in Table1)
        """

        phi = np.array(shape=(self.n_samples, 2, self.n_long_features, self.N))

        # integral over U
        iU_l = self.T
        for t in range(1, self.q_l):
            iU_l = np.c_[iU_l, (self.T ** (t + 1)) / (t + 1)]

        # linear time-varying features
        iV_l = np.c_[self.T, (self.T ** 2) / 2]
        for l in range(self.n_long_features):
            tmp = iV_l.dot(self.S[:, self.r_l * l: self.r_l * (l + 1)].T)
            beta_0l = self.beta[0, self.q_l * l: self.q_l * (l + 1)]
            beta_1l = self.beta[1, self.q_l * l: self.q_l * (l + 1)]
            phi[:, 0, l, :] = iU_l.dot(beta_0l) + tmp
            phi[:, 1, l, :] = iU_l.dot(beta_1l) + tmp

        return phi