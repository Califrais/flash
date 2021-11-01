
import numpy as np
from lights.model.associations import AssociationFunctionFeatures
from sklearn.preprocessing import StandardScaler
import numba as nb
from llvmlite import binding
binding.set_option('SVML', '-vector-library=SVML')
import matplotlib.pyplot as plt
from lights.simulation import features_normal_cov_toeplitz


class EstepFunctions:
    """A class to define functions relative to the E-step of the prox_QNMCEM

    Parameters
    ----------
    T_u : `np.ndarray`, shape=(J,)
        The J unique training censored times of the event of interest

    n_long_features :  `int`,
            Number of longitudinal features

    fixed_effect_time_order : `int`
        Order of the higher time monomial considered for the representations
        of the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    asso_functions_list : `list` or `str`='all'
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

    def __init__(self, T_u, n_long_features, fixed_effect_time_order,
                 asso_functions_list, theta):
        self.theta = theta
        self.n_long_features = n_long_features
        self.J = len(T_u)
        self.nb_total_asso_features = len(theta["gamma_0"])
        self.fixed_effect_time_order = fixed_effect_time_order
        self.q_l, r_l = fixed_effect_time_order + 1, 2
        alpha, L = self.fixed_effect_time_order, self.n_long_features
        self.F_f, self.F_r = AssociationFunctionFeatures(asso_functions_list,
                                                T_u, alpha, L).get_asso_feat()
        self.asso_funcs = None

    def compute_AssociationFunctions(self, S, simu, S_k=None):
        """
        Compute the value of association functions

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples, with N_MC = 2 * N

        simu : `bool`, defaut=True
            If `True` we comute the asso feat with simulated data.

        S_k : `list`
            Set of nonactive group for 2 classes (will be useful in case of
            simulated data)

        """
        N_MC = S.shape[0]
        K = 2
        L = self.n_long_features
        J, nb_total_asso_features = self.J, self.nb_total_asso_features
        q_l, r_l = self.q_l, 2
        nb_total_asso_param = nb_total_asso_features // L
        nb_asso_param = self.F_f.shape[1]
        nb_noise_param = nb_total_asso_param - nb_asso_param
        self.asso_funcs = np.zeros((J, N_MC, K, nb_total_asso_features))
        beta = np.hstack((self.theta["beta_0"], self.theta["beta_1"])).T
        #TODO: Hardcode
        # Correlation coefficient of the toeplitz correlation matrix
        rho = .05

        # TODO: Refactor
        if simu:
            for k in range(K):
                for l in range(L):
                    start_idx = nb_total_asso_param * l
                    stop_idx = nb_total_asso_param * l + nb_asso_param
                    if l not in S_k[k]:
                        beta_tmp = beta[k, r_l * l: r_l * (l + 1)]
                        S_tmp = S[:, q_l * l: q_l * (l + 1)].T
                        tmp = (self.F_f.dot(beta_tmp).T +
                               self.F_r.dot(S_tmp).T).T.swapaxes(1, -1)
                        self.asso_funcs[:, :, k, start_idx: stop_idx] = tmp
                        self.asso_funcs[:, :, k, nb_asso_param + nb_total_asso_param * l
                        : nb_total_asso_param * (l + 1)] = features_normal_cov_toeplitz(
                            J * N_MC,nb_noise_param, rho, .1)[0].reshape(J, N_MC, -1)

                    else:
                        self.asso_funcs[:, :, k, nb_total_asso_param * l : nb_total_asso_param * (l + 1)] = \
                        features_normal_cov_toeplitz(J * N_MC, nb_total_asso_param,
                                rho, .1)[0].reshape(J, N_MC, -1)
                # normalize features
                shape = self.asso_funcs[:, :, k].shape
                tmp_reshaped = self.asso_funcs[:, :, k].copy().reshape(-1, shape[-1])
                self.asso_funcs[:, :, k] = StandardScaler().fit_transform(
                    tmp_reshaped).copy().reshape(shape)
        else:
            for k in range(K):
                for l in range(L):
                    start_idx = nb_total_asso_param * l
                    stop_idx = nb_total_asso_param * l + nb_asso_param
                    beta_tmp = beta[k, r_l * l: r_l * (l + 1)]
                    S_tmp = S[:, q_l * l: q_l * (l + 1)].T
                    tmp = (self.F_f.dot(beta_tmp).T +
                            self.F_r.dot(S_tmp).T).T.swapaxes(1, -1)
                    self.asso_funcs[:, :, k, start_idx: stop_idx] = tmp
            pass

    def construct_MC_samples(self, N_MC):
        """Constructs the set of samples used for Monte Carlo approximation

        Parameters
        ----------
        N_MC : `int`
            Number of Monte Carlo samples

        Returns
        -------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples, with N_MC = 2 * N
        """
        D = self.theta["long_cov"]
        C = np.linalg.cholesky(D)
        r = D.shape[0]
        Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N_MC)
        b = Omega.dot(C.T)
        S = np.vstack((b, -b))

        return S

    def g1(self, S):
        """Computes g1

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g1 : `np.ndarray`, shape=(N_MC, r)
            The values of g1 function
        """

        g1 = S
        return g1

    def g2(self, S):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g2 : `np.ndarray`, shape=(N_MC, r, r)
            The values of g2 function
        """
        g2 = np.array([s.reshape(-1, 1).dot(s.reshape(-1, 1).T) for s in S])
        return g2

    def g3(self):
        """Computes g3
        Returns
        -------
        g3 : `np.ndarray`, shape=(N_MC, J, dim, K)
            The values of g3 function
        """
        g3 = self.asso_funcs.swapaxes(0, 1).swapaxes(2, 3)
        return g3

    def g4(self, gamma_0, gamma_1):
        """Computes g4

        Parameters
        ----------
        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for high-risk group

        Returns
        -------
        g4 : `np.ndarray`, shape=(N_MC, J, K)
            The values of g4 function
        """
        gamma = np.hstack((gamma_0, gamma_1)).T
        g4 = np.exp((self.asso_funcs * gamma).sum(axis=-1).swapaxes(0, 1))
        return g4

    def g5(self, gamma_0, gamma_1):
        """Computes g5
        Parameters
        ----------
        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for high-risk group
        Returns
        -------
        g5 : `np.ndarray`, shape=(N_MC, J, dim, K)
            The values of g5 function
        """
        g3 = self.g3()
        g4 = self.g4(gamma_0, gamma_1)
        g5 = g4[:, :, np.newaxis, :] * g3
        return g5

    @staticmethod
    def Lambda_g(g, f):
        """Approximated integral (see (15) in the lights paper)

        Parameters
        ----------
        g : `np.ndarray`, shape=(N_MC, ...)
            Values of g function for all subjects, all groups and all Monte
            Carlo samples. Each element could be real or matrices depending on
            Im(\tilde{g}_i)

        f: `np.ndarray`, shape=(n_samples, K, N_MC)
            Values of the density of the observed data given the latent ones and
            the current estimate of the parameters, computed for all subjects,
            all groups and all Monte Carlo samples

        Returns
        -------
        Lambda_g : `np.array`, shape=(n_samples, K, shape(g))
            The approximated integral computed for all subjects, all groups and
            all Monte Carlo samples. Each element could be real or matrices
            depending on Im(\tilde{g}_i)
        """
        if len(g.shape) == 1:
            Lambda_g = Lambda_g_1_nb(g, f)
        elif len(g.shape) == 2:
            Lambda_g = Lambda_g_2_nb(g, f)
        elif len(g.shape) == 3:
            Lambda_g = Lambda_g_3_nb(g, f)
        elif len(g.shape) == 4:
            Lambda_g = Lambda_g_4_nb(g, f)
        else:
            raise ValueError("The shape of g function is not supported")
        return Lambda_g

    def Eg(self, g, Lambda_1, pi_xi, f):
        """Computes approximated expectations of different functions g taking
        random effects as input, conditional on the observed data and the
        current estimate of the parameters. See (13) in the lights paper

        Parameters
        ----------
        g : `np.array`
            The value of g function for all samples

        Lambda_1: `np.ndarray`, shape=(n_samples, K)
            Approximated integral (see (17) in the lights paper) with
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


@nb.njit(parallel=True, fastmath=True)
def Lambda_g_1_nb(g, f):
    n_samples, K, N_MC = f.shape
    res = np.zeros((n_samples, K))
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                res[i, k] += (g[j] * f[i, k, j]) / N_MC
    return res


@nb.njit(parallel=True, fastmath=True)
def Lambda_g_2_nb(g, f):
    n_samples, K, N_MC,  = f.shape
    res = np.zeros((n_samples, K,) + g.shape[1:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                for m in nb.prange(g.shape[1]):
                    res[i, k, m] += (g[j, m] * f[i, k, j]) / N_MC
    return res


@nb.njit(parallel=True, fastmath=True)
def Lambda_g_3_nb(g, f):
    n_samples, K, N_MC,  = f.shape
    res = np.zeros((n_samples, K,) + g.shape[1:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                for m in nb.prange(g.shape[1]):
                    for n in nb.prange(g.shape[2]):
                        res[i, k, m, n] += \
                                        (g[j, m, n] * f[i, k, j]) / N_MC
    return res


@nb.njit(parallel=True, fastmath=True)
def Lambda_g_4_nb(g, f):
    n_samples, K, N_MC,  = f.shape
    res = np.zeros((n_samples, K,) + g.shape[1:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                for m in nb.prange(g.shape[1]):
                    for n in nb.prange(g.shape[2]):
                        for t in nb.prange(g.shape[3]):
                            res[i, k, m, n, t] += \
                                (g[j, m, n, t] * f[i, k, j]) / N_MC
    return res