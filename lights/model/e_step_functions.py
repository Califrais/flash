
import numpy as np
from lights.model.associations import AssociationFunctionFeatures
from sklearn.preprocessing import StandardScaler
import numba as nb
from numpy.linalg import multi_dot
from llvmlite import binding
binding.set_option('SVML', '-vector-library=SVML')
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

    def __init__(self, T_u, n_long_features, asso_feats,  fixed_effect_time_order, theta):
        self.theta = theta
        self.n_long_features = n_long_features
        self.J = len(T_u)
        self.nb_total_asso_features = len(theta["gamma_0"])
        self.fixed_effect_time_order = fixed_effect_time_order
        self.q_l, self.r_l = fixed_effect_time_order + 1, 2
        self.asso_feats = asso_feats


    def b_stats(self, extracted_features):
        """
        Compute the statistics of random effect given longitudinal data
        and latent groups.

        Parameters
        ----------
        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        """
        (U_list, V_list, y_list, N), _ = extracted_features
        n_samples, n_long_features = len(U_list), self.n_long_features
        theta = self.theta
        D, phi = theta["long_cov"], theta["phi"]
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        beta_stack = np.hstack((beta_0, beta_1))
        inv_D = np.linalg.inv(D)
        r = n_long_features * self.r_l
        K = 2
        self.Eb = np.zeros((n_samples, K, r))
        self.EbbT = np.zeros((n_samples, K, r, r))
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], sum(N[i])
            Phi_i = [[1 / phi[l, 0]] * N[i][l] for l in range(n_long_features)]
            Sigma_i = np.diag(np.concatenate(Phi_i))
            tmp_1 = np.linalg.inv(multi_dot([V_i.T, Sigma_i, V_i]) + inv_D)
            tmp_2 = y_i - U_i.dot(beta_stack)
            for k in range(K):
                b_mean = multi_dot([tmp_1, V_i.T, Sigma_i, tmp_2[:, k]]).reshape(-1, 1)
                b_cov = tmp_1
                self.Eb[i, k] = b_mean.flatten()
                self.EbbT[i, k] = b_cov + b_mean.dot(b_mean.T)

    @staticmethod
    def Lambda_g(g, f):
        """The calculation of integral (see (15) in the lights paper)

        Parameters
        ----------
        g : `np.ndarray`, shape=(n_samples, K, ...)
            Values of g function for all subjects, all groups and all Monte
            Carlo samples. Each element could be real or matrices depending on
            Im(\tilde{g}_i)

        f: `np.ndarray`, shape=(n_samples, K)
            Values of the density of the observed data given the latent ones and
            the current estimate of the parameters, computed for all subjects,
            all groups.

        Returns
        -------
        Lambda_g : `np.array`, shape=(n_samples, K, shape(g))
            The integral computed for all subjects, all groups and
            all Monte Carlo samples. Each element could be real or matrices
            depending on Im(\tilde{g}_i)
        """
        Lambda_g = (g.T * f.T).T
        return Lambda_g

    def Eg(self, g, Lambda_1, pi_xi, f):
        """Computes expectations of different functions g taking
        random effects as input, conditional on the observed data and the
        current estimate of the parameters. See (13) in the lights paper

        Parameters
        ----------
        g : `np.array`
            The value of g function for all samples

        Lambda_1: `np.ndarray`, shape=(n_samples, K)
            The computed integral (see (15) in the lights paper) with
            \tilde(g)=1

        pi_xi: `np.ndarray`, shape=(n_samples,)
            The posterior probability of the sample for being on the
            high-risk group given all observed data

        f: `np.ndarray`, shape=(n_samples, K)
            The value of the f(Y, T, delta| G ; theta)

        Returns
        -------
        Eg : `np.ndarray`, shape=(n_samples, g.shape)
            The expectations for g
        """
        Lambda_g = self.Lambda_g(g, f)
        Eg = (Lambda_g[:, 0].T * (1 - pi_xi) + Lambda_g[:, 1].T * pi_xi) / (
                Lambda_1[:, 0] * (1 - pi_xi) + Lambda_1[:, 1] * pi_xi)
        return Eg.T