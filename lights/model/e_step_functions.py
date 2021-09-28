
import numpy as np
from lights.model.associations import AssociationFunctionFeatures
from sklearn.preprocessing import StandardScaler
import numba as nb
from llvmlite import binding
binding.set_option('SVML', '-vector-library=SVML')


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
        self.fixed_effect_time_order = fixed_effect_time_order
        alpha, L = self.fixed_effect_time_order, self.n_long_features
        self.F_f, self.F_r = AssociationFunctionFeatures(asso_functions_list,
                                                T_u, alpha, L).get_asso_feat()
        self.asso_funcs = None

    def compute_AssociationFunctions(self, S):
        """
        Compute the value of association functions

        Parameters
        ----------
        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples, with N_MC = 2 * N

        """
        beta = np.hstack((self.theta["beta_0"], self.theta["beta_1"])).T
        self.asso_funcs = (self.F_f.dot(beta.T)[:, :, :, None] +
              self.F_r.dot(S.T)[:, :, None, :]).swapaxes(1, 3)
        asso_funcs_shape = self.asso_funcs.shape
        reshaped_asso_funcs = self.asso_funcs.copy().reshape((-1, asso_funcs_shape[-1]))
        nb_rnd_param = self.theta["gamma_0"].shape[0] - asso_funcs_shape[-1]
        # TODO: hardcode
        var_rnd_param = .01
        rnd_asso = np.random.normal(0, var_rnd_param,
                                    size=(reshaped_asso_funcs.shape[0], nb_rnd_param))
        reshaped_asso_funcs = np.hstack((reshaped_asso_funcs, rnd_asso))
        self.asso_funcs = StandardScaler().fit_transform(
            reshaped_asso_funcs).reshape((asso_funcs_shape[:-1] + (-1,)))

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
        Parameters
        ----------
        broadcast : `boolean`, default=True
            Indicate to expand the dimension or not
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