import numpy as np
from lights.model.associations import AssociationFunctions
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
    """

    def __init__(self, X, T, T_u, delta, extracted_features,
                 fixed_effect_time_order, asso_functions, theta):
        self.K = 2  # 2 latent groups
        self.X, self.T, self.delta = X, T, delta
        self.T_u, self.n_samples = T_u, len(T)
        self.J, self.ind_1, _ = get_times_infos(T, T_u)
        self.extracted_features, self.theta = extracted_features, theta
        self.n_long_features = len(extracted_features[1][0])
        self.n_time_indep_features = X.shape[1]
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.g3_, self.g4_, self.g9_ = None, None, None

    def construct_MC_samples(self, N):
        """Constructs the set of samples used for Monte Carlo approximation

        Parameters
        ----------
        N : `int`
            Number of Monte Carlo samples

        Returns
        -------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples, with N_MC = 2 * N
        """
        D = self.theta["long_cov"]
        C = np.linalg.cholesky(D)
        r = D.shape[0]
        Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N)
        b = Omega.dot(C.T)
        S = np.vstack((b, -b))
        return S

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

    def g1(self, S, gamma_0, beta_0, gamma_1, beta_1, broadcast=True):
        """Computes g1

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param + p,)
            Association parameters for low-risk group

        beta_0 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param + p,)
            Association parameters for high-risk group

        beta_1 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for high-risk group

        broadcast : `boolean`, default=True
            Indicates to expand the dimension of g1 or not

        Returns
        -------
        g1 : `np.ndarray`, shape=(n_samples, K, N_MC, J)
                            or (n_samples, K, N_MC, J, K)
            The values of g1 function
        """
        n_samples, K = self.n_samples, self.K
        p = self.n_time_indep_features
        X, T_u, J = self.X, self.T_u, self.J
        N_MC = S.shape[0]
        gamma_indep = np.hstack((gamma_0[:p], gamma_1[:p]))
        g2_ = self.g2(S, gamma_0, beta_0, gamma_1, beta_1).reshape(K, 1, J, N_MC)
        tmp = X.dot(gamma_indep).T.reshape(K, n_samples, 1, 1)
        g1 = np.exp(tmp + g2_).swapaxes(0, 1).swapaxes(2, 3)
        if broadcast:
            g1 = np.broadcast_to(g1[..., None], g1.shape + (2,)).swapaxes(1, -1)
        return g1

    def g2(self, S, gamma_0, beta_0, gamma_1, beta_1):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param + p,)
            Association parameters for low-risk group

        beta_0 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param + p,)
            Association parameters for high-risk group

        beta_1 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for high-risk group

        broadcast : `boolean`, default=True
            Indicate to expand the dimension of g2 or not

        Returns
        -------
        g2 : `np.ndarray`, shape=(K, J, N_MC)
            The values of g2 function
        """
        T_u, p, K = self.T_u, self.n_time_indep_features, self.K
        N_MC, J = S.shape[0], self.J
        asso_functions, L = self.asso_functions, self.n_long_features
        alpha = self.fixed_effect_time_order
        gamma_dep = np.vstack((gamma_0[p:], gamma_1[p:])).reshape(K, -1)
        beta = np.vstack((beta_0, beta_1)).reshape(K, -1)
        fixed_feat_assoc, rand_feat_assoc = AssociationFunctions(T_u, alpha, L)._get_assoc_feat()
        g2 = ((fixed_feat_assoc.dot(beta.T)[:, :, :, None] +
              rand_feat_assoc.dot(S.T)[:, :, None, :]).swapaxes(1,3) * gamma_dep).sum(axis=-1)
        g2 = g2.swapaxes(0, 1).T
        return g2

    def g3(self, S, beta_0, beta_1):
        """Computes g3, g4, g9

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
        theta, K, N_MC = self.theta, self.K, S.shape[0]
        phi = theta["phi"]
        beta_stack = np.hstack((beta_0, beta_1))
        g3 = []
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], N_list[i]
            M_iS = U_i.dot(beta_stack).T.reshape(K, -1, 1) + V_i.dot(S.T)
            Phi_i = [[1 / phi[l, 0]] * n_i[l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            g3.append(M_iS)
        return g3

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
