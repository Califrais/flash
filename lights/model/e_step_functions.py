import numpy as np
from lights.model.associations import AssociationFunctionFeatures
from lights.base.base import get_times_infos
import numba as nb
from llvmlite import binding
binding.set_option('SVML', '-vector-library=SVML')
from sklearn.preprocessing import StandardScaler

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
                 fixed_effect_time_order, asso_functions_list, theta):
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
        self.g6_, self.g2_ = None, None
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
        self.asso_funcs = (self.F_f.dot(beta.T)[:, :, :, None] +
              self.F_r.dot(S.T)[:, :, None, :]).swapaxes(1, 3)
        shape = self.asso_funcs.shape
        reshaped_asso_funcs = self.asso_funcs.copy().reshape((-1, shape[-1]))
        self.asso_funcs = StandardScaler().fit_transform(
            reshaped_asso_funcs).reshape(shape)

    def construct_MC_samples(self, N_MC):
        """Constructs the set of samples used for Monte Carlo approximation

        Parameters
        ----------
        N_MC : `int`
            Number of Monte Carlo samples

        Returns
        -------
        S : `np.ndarray`, shape=(N_MC, r) or (n_samples, K, N_MC, r)
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
        g1 : `np.ndarray`, shape=(n_samples, K, N_MC, r)
            The values of g1 function
        """

        g1 = np.broadcast_to(S, (self.n_samples, self.K) + S.shape)
        return g1

    def g2(self, S):
        """Computes g2

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        Returns
        -------
        g2 : `np.ndarray`, shape=(n_samples, K, N_MC, r, r)
            The values of g2 function
        """
        tmp = np.array([s.reshape(-1, 1).dot(s.reshape(-1, 1).T) for s in S])
        g2 = np.broadcast_to(tmp, (self.n_samples, self.K) + tmp.shape)
        return g2

    def g3(self, broadcast=True):
        """Computes g3
        Parameters
        ----------
        broadcast : `boolean`, default=True
            Indicate to expand the dimension or not
        Returns
        -------
        g3 : `np.ndarray`, shape=(K, N_MC, J, dim)
                            or (n_samples, K, N_MC, J, dim, K)
            The values of g3 function
        """
        g3 = self.asso_funcs.swapaxes(0, 2)
        if broadcast:
            g3 = np.broadcast_to(g3, (self.n_samples,) + g3.shape)
            g3 = np.broadcast_to(g3[..., None], g3.shape + (2,)).swapaxes(1, -1)
        return g3

    def g4(self, gamma_0, gamma_1, broadcast=True):
        """Computes g4

        Parameters
        ----------
        gamma_0 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for low-risk group

        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for high-risk group

        broadcast : `boolean`, default=True
            Indicates to expand the dimension of g4 or not

        Returns
        -------
        g4 : `np.ndarray`, shape=(n_samples, K, N_MC, J)
                            or (n_samples, K, N_MC, J, K)
            The values of g4 function
        """
        n_samples, K = self.n_samples, self.K
        gamma = np.hstack((gamma_0, gamma_1)).T
        tmp = (self.asso_funcs * gamma).sum(axis=-1).swapaxes(0, 1).T
        tmp_ = np.broadcast_to(tmp, (n_samples, ) + tmp.shape)
        g4 = np.exp(tmp_).swapaxes(2, 3)
        if broadcast:
            g4 = np.broadcast_to(g4[..., None], g4.shape + (2,)).swapaxes(1, -1)
        return g4

    def g5(self, S, gamma_0, gamma_1):
        """Computes g5
        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples
        gamma_1 : `np.ndarray`, shape=(L * nb_asso_param + p,)
            Association parameters for high-risk group
        beta_1 : `np.ndarray`, shape=(q,)
            Fixed effect parameters for high-risk group
        Returns
        -------
        g5 : `np.ndarray`, shape=(n_samples, K, N_MC, J, dim, K)
            The values of g5 function
        """
        g3 = self.g3(False)
        g4 = self.g4(gamma_0, gamma_1, False)
        g5 = g4[..., np.newaxis] * g3
        g5 = np.broadcast_to(g5[..., None], g5.shape + (2,)).swapaxes(1, -1)
        return g5

    def g6(self, S, beta_0, beta_1):
        """Computes g6

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
        g6 = []
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], N_list[i]
            M_iS = U_i.dot(beta_stack).T.reshape(K, -1, 1) + V_i.dot(S.T)
            g6.append(M_iS)
        return g6

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
        if len(g.shape) == 3:
            Lambda_g = Lambda_g_3_nb(g, f)
        elif len(g.shape) == 4:
            Lambda_g = Lambda_g_4_nb(g, f)
        elif len(g.shape) == 5:
            Lambda_g = Lambda_g_5_nb(g, f)
        elif len(g.shape) == 6:
            Lambda_g = Lambda_g_6_nb(g, f)
        else:
            raise ValueError("The shape of g function is not supported")
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

@nb.njit(parallel=True, fastmath=True)
def Lambda_g_3_nb(g, f):
    n_samples, K, N_MC = f.shape
    res = np.zeros(g.shape[:2] + g.shape[3:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                res[i, k] += (g[i, k, j] * f[i, k, j]) / N_MC
    return res

@nb.njit(parallel=True, fastmath=True)
def Lambda_g_4_nb(g, f):
    n_samples, K, N_MC,  = f.shape
    res = np.zeros(g.shape[:2] + g.shape[3:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                for m in nb.prange(g.shape[3]):
                    res[i, k, m] += (g[i, k, j, m] * f[i, k, j]) / N_MC
    return res

@nb.njit(parallel=True, fastmath=True)
def Lambda_g_5_nb(g, f):
    n_samples, K, N_MC,  = f.shape
    res = np.zeros(g.shape[:2] + g.shape[3:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                for m in nb.prange(g.shape[3]):
                    for n in nb.prange(g.shape[4]):
                        res[i, k, m, n] += \
                                        (g[i, k, j, m, n] * f[i, k, j]) / N_MC
    return res

@nb.njit(parallel=True, fastmath=True)
def Lambda_g_6_nb(g, f):
    n_samples, K, N_MC,  = f.shape
    res = np.zeros(g.shape[:2] + g.shape[3:])
    for i in nb.prange(n_samples):
        for k in nb.prange(K):
            for j in nb.prange(N_MC):
                for m in nb.prange(g.shape[3]):
                    for n in nb.prange(g.shape[4]):
                        for t in nb.prange(g.shape[5]):
                            res[i, k, m, n, t] += \
                                (g[i, k, j, m, n, t] * f[i, k, j]) / N_MC
    return res