import numpy as np
from lights.model.associations import AssociationFunctionFeatures
from lights.base.base import get_times_infos


class EstepFunctions:
    """A class to define functions relative to the E-step of the QNMCEM

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

    T : `np.ndarray`, shape=(n_samples,)
        The censored times of the event of interest

    T_u : `np.ndarray`, shape=(J,)
        The J unique training censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        The censoring indicator

    extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

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

    theta : `dict`
        Vector that concatenates all parameters to be inferred in the lights
        model

    MC_sep: `bool`, default=False
        If `False`, we use the same set of MC samples for all subject,
        otherwise we sample a seperate set of MC samples for each subject
    """

    def __init__(self, X, T, T_u, delta, extracted_features,
                 fixed_effect_time_order, asso_functions_list, theta, MC_sep):
        self.K = 2  # 2 latent groups
        self.X, self.T, self.delta = X, T, delta
        self.T_u, self.n_samples = T_u, len(T)
        self.J, self.ind_1, _ = get_times_infos(T, T_u)
        self.extracted_features, self.theta = extracted_features, theta
        self.n_long_features = len(extracted_features[1][0])
        self.n_time_indep_features = X.shape[1]
        self.fixed_effect_time_order = fixed_effect_time_order
        alpha, L = self.fixed_effect_time_order, self.n_long_features
        self.F_f, self.F_r = AssociationFunctionFeatures(asso_functions_list,
                                                T_u, alpha, L).get_asso_feat()
        self.MC_sep = MC_sep
        self.g3_, self.g4_, self.g9_ = None, None, None
        self.asso_funcs = None

    def compute_AssociationFunctions(self, S):
        """
        Compute the value of association functions

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r) or (n_samples, K, N_MC, r)
            Set of constructed Monte Carlo samples, with N_MC = 2 * N

        """
        beta = np.hstack((self.theta["beta_0"], self.theta["beta_1"])).T
        if self.MC_sep:
            self.asso_funcs = (self.F_f.dot(beta.T)[:, :, :, None, None]
                   + (self.F_r[:, :, :, None, None, None] * S.T).sum(
                        axis=2).swapaxes(2, 3)).swapaxes(2, 3).swapaxes(1, 4)
        else:
            self.asso_funcs = (self.F_f.dot(beta.T)[:, :, :, None] +
                  self.F_r.dot(S.T)[:, :, None, :]).swapaxes(1, 3)

    def construct_MC_samples(self, N_MC):
        """Constructs the set of samples used for Monte Carlo approximation

        Parameters
        ----------
        N_MC : `int`
            Number of Monte Carlo samples

        MC_sep: `bool`, default=False
            If `False`, we use the same set of MC samples for all subject,
            otherwise we sample a seperate set of MC samples for each subject

        Returns
        -------
        S : `np.ndarray`, shape=(N_MC, r) or (n_samples, K, N_MC, r)
            Set of constructed Monte Carlo samples, with N_MC = 2 * N
        """
        D = self.theta["long_cov"]
        if self.MC_sep:
            n_samples = self.n_samples
            (U, V, y, N) = self.extracted_features[0]
            n_long_features = self.n_long_features
            phi = self.theta["phi"]
            beta = np.hstack((self.theta["beta_0"], self.theta["beta_1"]))
            r = D.shape[0]
            Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N_MC)
            S = np.zeros((n_samples, 2, 2 * N_MC, r))

            for i in range(n_samples):
                U_i, V_i, y_i, N_i = U[i], V[i], y[i], N[i]

                # compute Sigma_i
                Phi_i = [[1 / phi[l, 0]] * N_i[l]
                         for l in range(n_long_features)]
                Sigma_i = np.diag(np.concatenate(Phi_i))

                # compute Omega_i
                D_inv = np.linalg.inv(D)
                A_i = np.linalg.inv(
                    V_i.transpose().dot(Sigma_i).dot(V_i) + D_inv)
                # compute mu_i
                mu_i = (A_i.dot(V_i.transpose()).dot(Sigma_i).dot(
                    y_i - U_i.dot(beta))).T[..., np.newaxis]
                C_i = np.linalg.cholesky(A_i)
                tmp = C_i.dot(Omega.T)
                S[i] = (mu_i + np.hstack((tmp, -tmp))).swapaxes(1, 2)
        else:
            C = np.linalg.cholesky(D)
            r = D.shape[0]
            Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N_MC)
            b = Omega.dot(C.T)
            S = np.vstack((b, -b))

        return S

    def g1(self, S, gamma_0, gamma_1, broadcast=True):
        """Computes g1

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for high-risk group

        broadcast : `boolean`, default=True
            Indicates to expand the dimension of g1 or not

        Returns
        -------
        g1 : `np.ndarray`, shape=(n_samples, K, N_MC, J)
                            or (n_samples, K, N_MC, J, K)
            The values of g1 function
        """
        n_samples, K = self.n_samples, self.K
        X, T_u, J = self.X, self.T_u, self.J
        g2 = self.g2(gamma_0, gamma_1)
        if self.MC_sep:
            g1 = np.exp(g2).swapaxes(0, 1).swapaxes(2, 3)
        else:
            tmp = self.g2(gamma_0, gamma_1)
            g2 = np.broadcast_to(tmp, (n_samples, ) + tmp.shape)
            g1 = np.exp(g2).swapaxes(2, 3)
        if broadcast:
            g1 = np.broadcast_to(g1[..., None], g1.shape + (2,)).swapaxes(1, -1)
        return g1

    def g2(self, gamma_0, gamma_1):
        """Computes g2

        Parameters
        ----------
        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for high-risk group

        Returns
        -------
        g2 : `np.ndarray`, shape=(K, J, N_MC) or (K, n_samples, J, N_MC)
            The values of g2 function
        """
        gamma = np.hstack((gamma_0, gamma_1)).T
        g2 = (self.asso_funcs * gamma).sum(axis=-1)
        if self.MC_sep:
            g2 = g2.swapaxes(0, 2).swapaxes(1, 2).T
        else:
            g2 = g2.swapaxes(0, 1).T
        return g2

    def g3(self, S, beta_0, beta_1):
        """Computes g3

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        beta_0 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for low-risk group

        beta_1 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for high-risk group
        """
        n_samples, n_long_features = self.n_samples, self.n_long_features
        U_list, V_list, y_list, N_list = self.extracted_features[0]
        K = self.K
        beta_stack = np.hstack((beta_0, beta_1))
        g3 = []
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], N_list[i]
            if self.MC_sep:
                M_iS = U_i.dot(beta_stack).T.reshape(K, -1, 1) \
                       + S[i].dot(V_i.T).swapaxes(1, 2)
            else:
                M_iS = U_i.dot(beta_stack).T.reshape(K, -1, 1) + V_i.dot(S.T)
            g3.append(M_iS)
        return g3

    def g4(self, S):
        """Computes g4

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g4 : `np.ndarray`, shape=(n_samples, K, N_MC, r, r)
            The values of g4 function
        """
        if self.MC_sep:
            n_samples = self.n_samples
            K, r = self.K, self.n_long_features * 2
            N_MC = S.shape[2]
            g4 = np.zeros((n_samples, K, N_MC, r, r))
            for i in range(n_samples):
                for k in range(K):
                    g4[i, k] = np.array(
                        [s.reshape(-1, 1).dot(s.reshape(-1, 1).T)
                         for s in S[i, k]])
        else:
            tmp = np.array([s.reshape(-1, 1).dot(s.reshape(-1, 1).T) for s in S])
            g4 = np.broadcast_to(tmp, (self.n_samples, self.K) + tmp.shape)
        return g4

    def g5(self, S):
        """Computes g5

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g5 : `np.ndarray`, shape=(n_samples, K, N_MC, r)
            The values of gS function
        """
        if self.MC_sep:
            g5 = S
        else:
            g5 = np.broadcast_to(S, (self.n_samples, self.K) + S.shape)
        return g5

    def g6(self, S, gamma_0, gamma_1):
        """Computes g6

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for high-risk group

        Returns
        -------
        g6 : `np.ndarray`, shape=(n_samples, K, N_MC, r, J, K)
            The values of g6 function
        """
        if self.MC_sep:
            g1 = self.g1(S, gamma_0, gamma_1)
            g6 = g1.swapaxes(0, -2)[..., np.newaxis] \
                 * S.swapaxes(0, 2).swapaxes(1, 2)
            return g6.T.swapaxes(0, 2).swapaxes(2, 3).swapaxes(4, 5)
        else:
            g1 = self.g1(S, gamma_0, gamma_1, broadcast=True)
            g6 = g1.swapaxes(2, -1)[..., np.newaxis] * S
            return g6.swapaxes(2, -1).swapaxes(3, 4).swapaxes(2, 3)

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
            The value of the f(Y, T, delta| S, G ; theta)

        Returns
        -------
        Eg : `np.ndarray`, shape=(n_samples, g.shape)
            The approximated expectations for g
        """
        Lambda_g = self.Lambda_g(g, f)
        Eg = (Lambda_g[:, 0].T * (1 - pi_xi) + Lambda_g[:, 1].T * pi_xi) / (
                Lambda_1[:, 0] * (1 - pi_xi) + Lambda_1[:, 1] * pi_xi)
        return Eg.T
