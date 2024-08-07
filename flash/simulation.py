from datetime import datetime
from time import time
from scipy.linalg.special_matrices import toeplitz
from flash.base.base import normalize, logistic_grad, probit
import numpy as np
import pandas as pd


def features_normal_cov_toeplitz(n_samples: int = 200, n_features: int = 10,
                                 rho: float = 0.5, cst: float = .1):
    """Features obtained as samples of a centered Gaussian vector
    with a toeplitz covariance matrix

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=10
        Number of features

    rho : `float`, default=0.5
        Correlation coefficient of the toeplitz correlation matrix

    cst : `float`, default=.1
        Multiplicative constant

    Returns
    -------
    features : `np.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix

    cov : `np.ndarray`, shape=(n_features, n_features)
        The simulated variance-covariance matrix
    """
    cov = toeplitz(rho ** np.arange(0, n_features))
    # The cst is to reduce the variance of covariance matrix
    np.fill_diagonal(cov, (cst ** 2) * np.diagonal(cov))
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    return features, cov


def simulation_method(simulate_method):
    """A decorator for simulation methods.
    It simply calls _start_simulation and _end_simulation methods
    """
    def decorated_simulate_method(self):
        self._start_simulation()
        result = simulate_method(self)
        self._end_simulation()
        self.data = result
        return result

    return decorated_simulate_method


class Simulation:
    """This is an abstract simulation class that inherits from BaseClass
    It does nothing besides printing stuff and verbosing
    """

    def __init__(self, seed=0, verbose=True):
        # Set default parameters
        self.seed = seed
        self.verbose = verbose
        self._set_seed()
        # No data simulated yet
        self.time_indep_features = None
        self.long_features = None
        self.times = None
        self.censoring = None

    def _set_seed(self):
        np.random.seed(self.seed)
        return self

    @staticmethod
    def _get_now():
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def _start_simulation(self):
        self.time_start = Simulation._get_now()
        self._numeric_time_start = time()
        if self.verbose:
            msg = "Launching simulation using {class_}..." \
                .format(class_=self.__class__.__name__)
            print("-" * len(msg))
            print(msg)

    def _end_simulation(self):
        self.time_end = self._get_now()
        t = time()
        self.time_elapsed = t - self._numeric_time_start
        if self.verbose:
            msg = "Done simulating using {class_} in {time:.2e} seconds." \
                .format(class_=self.__class__.__name__,
                        time=self.time_elapsed)
            print(msg)


class SimuJointLongitudinalSurvival(Simulation):
    """Class for the simulation of Joint High-dimensional Longitudinal and
    Survival Data

    Parameters
    ----------
    verbose : `bool`, default=True
        Verbose mode to detail or not ongoing tasks

    seed : `int`, default=None
        The seed of the random number generator, for reproducible simulation. If
        `None` it is not seeded

    n_samples : `int`, default=1000
        Number of samples

    n_time_indep_features : `int`, default=10
        Number of time-independent features

    sparsity : `float`, default=.5
        Proportion of both time-independent and association features active
        coefficients vector. Must be in [0, 1].

    coeff_val_time_indep : `float`, default=.2
        Value of the active coefficients in the time-independent coefficient
        vectors

    coeff_val_asso_low_risk : `float`, default=.1
        Value of the coefficients parameter used in the association coefficient
        vectors of low risk group

    coeff_val_asso_high_risk : `float`, default=.4
        Value of the coefficients parameter used in the association coefficient
        vectors of high risk group

    cov_corr_time_indep : `float`, default=.5
        Correlation to use in the Toeplitz covariance matrix for the
        time-independent features

    high_risk_rate : `float`, default=.4
        Proportion of desired high risk samples rate

    gap : `float`, default=1.
        Gap value to create high/low risk groups in the time-independent
        features

    n_long_features : `int`, default=5
        Number of longitudinal features

    cov_corr_long : `float`, default=.01
        Correlation to use in the Toeplitz covariance matrix for the random
        effects simulation

    fixed_effect_mean_low_risk : `tuple`, default=(-.6, .1)
        Mean vector of the gaussian used to generate the fixed effect parameters
        for the low risk group

    fixed_effect_mean_high_risk : `tuple`, default=(.05, .2)
        Mean vector of the gaussian used to generate the fixed effect parameters
        for the high risk group

    corr_fixed_effect : `float`, default=.01
        Correlation value to use in the diagonal covariance matrix for the
        fixed effect simulated feature

    std_error : `float`, default=.5
        Standard deviation for the error term of the longitudinal processes

    decay : `float`, default=3.0
        Decay of exponential kernels for the multivariate Hawkes processes to
        generate measurement times

    baseline_hawkes_uniform_bounds : `list`, default=(.1, 1.)
        Bounds of the uniform distribution used to generate baselines of
        measurement times intensities

    adjacency_hawkes_uniform_bounds : `list`, default=(.1, .2)
        Bounds of the uniform distribution used to generate sparse adjacency
        matrix for measurement times intensities

    scale : `float`, default=.05
        Scaling parameter of the Gompertz distribution of the baseline

    shape : `float`, default=.1
        Shape parameter of the Gompertz distribution of the baseline

    censoring_factor : `float`, default=2.
        Level of censoring. Increasing censoring_factor leads to less censored
        times and conversely.

    Attributes
    ----------
    time_indep_features : `np.array`, shape=(n_samples, n_time_indep_features)
        Simulated time-independent features

    long_features :
        Simulated longitudinal features

    times : `np.ndarray`, shape=(n_samples,)
        Simulated times of the event of interest

    censoring : `np.ndarray`, shape=(n_samples,)
        Simulated censoring indicator, where ``censoring[i] == 1``
        indicates that the time of the i-th individual is a failure
        time, and where ``censoring[i] == 0`` means that the time of
        the i-th individual is a censoring time

    latent_class : `np.ndarray`, shape=(n_samples,)
        Simulated latent classes

    time_indep_coeffs : `np.ndarray`, shape=(n_time_indep_features,)
        Simulated time-independent coefficient vector

    long_cov : `np.ndarray`, shape=(2*n_long_features, 2*n_long_features)
        Variance-covariance matrix that accounts for dependence between the
        different longitudinal outcome. Here r = 2*n_long_features since
        one choose affine random effects, so all r_l=2

    fixed_effect_coeffs : `list`, [beta_1, beta_2]
        Simulated fixed effect coefficient vectors per group

    asso_coeffs : `list`, [gamma_1, gamma_2]
        Simulated association parameters per group

    iotas : `dict`, {1: [iota_01, iota_11], 2: [iota_02, iota_12]}
        Simulated linear time-varying features per group in the Cox model used
        to simulate event times

    event_times : `np.ndarray`, shape=(n_samples,)
            The simulated times of the event of interest

    gird_time : `bool`, defaut=True
        If `True` we simulate data with the same time measurement for all
        longitudinal features. Otherwise, we use multivariate Hawkes process to
         simulate the time measurement for each longitudinal features.

    hawkes : `list` of `tick.hawkes.simulation.simu_hawkes_exp_kernels`
        Store the multivariate Hawkes processes with exponential kernels
        used to simulate measurement times for intensities plotting purpose

    Notes
    -----
    There is no intercept in this model
    """

    def __init__(self, verbose: bool = True, seed: int = None,
                 n_samples: int = 1000, n_time_indep_features: int = 10,
                 sparsity: float = .5, coeff_val_time_indep: float = .2,
                 coeff_val_asso_low_risk: float = .1,
                 coeff_val_asso_high_risk: float = .4,
                 cov_corr_time_indep: float = .5,
                 high_risk_rate: float = .4, gap: float = 1.,
                 n_long_features: int = 10, cov_corr_long: float = .01,
                 fixed_effect_mean_low_risk: tuple = (-.6, .1),
                 fixed_effect_mean_high_risk: tuple = (.05, .2),
                 corr_fixed_effect: float = .01,
                 std_error: float = .5,
                 shape: float = .1, scale: float = .05,
                 censoring_factor: float = 2., grid_time: bool = True,
                 probit=False):
        Simulation.__init__(self, seed=seed, verbose=verbose)

        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        self.sparsity = sparsity
        self.coeff_val_time_indep = coeff_val_time_indep
        self.coeff_val_asso_low_risk = coeff_val_asso_low_risk
        self.coeff_val_asso_high_risk = coeff_val_asso_high_risk
        self.cov_corr_time_indep = cov_corr_time_indep
        self.high_risk_rate = high_risk_rate
        self.gap = gap
        self.n_long_features = n_long_features
        self.cov_corr_long = cov_corr_long
        self.fixed_effect_mean_low_risk = fixed_effect_mean_low_risk
        self.fixed_effect_mean_high_risk = fixed_effect_mean_high_risk
        self.corr_fixed_effect = corr_fixed_effect
        self.std_error = std_error
        self.shape = shape
        self.scale = scale
        self.censoring_factor = censoring_factor
        self.grid_time = grid_time

        # Attributes that will be instantiated afterwards
        self.latent_class = None
        self.time_indep_coeffs = None
        self.long_cov = None
        self.event_times = None
        self.fixed_effect_coeffs = None
        self.asso_coeffs = None
        self.iotas = None
        self.hawkes = []
        self.probit = probit

    @property
    def sparsity(self):
        return self._sparsity

    @sparsity.setter
    def sparsity(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``sparsity`` must be in (0, 1)")
        self._sparsity = val

    @property
    def high_risk_rate(self):
        return self._high_risk_rate

    @high_risk_rate.setter
    def high_risk_rate(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``high_risk_rate`` must be in (0, 1)")
        self._high_risk_rate = val

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        if val <= 0:
            raise ValueError("``scale`` must be strictly positive")
        self._scale = val

    @simulation_method
    def simulate(self):
        """Launch simulation of the data

        Returns
        -------
        X : `numpy.ndarray`, shape=(n_samples, n_time_indep_features)
            The simulated time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            The simulated censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            The simulated censoring indicator

        """
        seed = self.seed
        n_samples = self.n_samples
        n_time_indep_features = self.n_time_indep_features
        sparsity = self.sparsity
        coeff_val_time_indep = self.coeff_val_time_indep
        coeff_val_asso_low_risk = self.coeff_val_asso_low_risk
        coeff_val_asso_high_risk = self.coeff_val_asso_high_risk
        cov_corr_time_indep = self.cov_corr_time_indep
        high_risk_rate = self.high_risk_rate
        gap = self.gap
        n_long_features = self.n_long_features
        cov_corr_long = self.cov_corr_long
        fixed_effect_mean_low_risk = self.fixed_effect_mean_low_risk
        fixed_effect_mean_high_risk = self.fixed_effect_mean_high_risk
        corr_fixed_effect = self.corr_fixed_effect
        std_error = self.std_error
        shape = self.shape
        scale = self.scale
        censoring_factor = self.censoring_factor

        # Simulation of time-independent coefficient vector
        nb_active_time_indep_features = int(n_time_indep_features * sparsity)
        xi = np.zeros(n_time_indep_features)
        xi[0:nb_active_time_indep_features] = coeff_val_time_indep
        self.time_indep_coeffs = xi

        # Simulation of time-independent features
        X = features_normal_cov_toeplitz(n_samples, n_time_indep_features,
                                         cov_corr_time_indep, cst=1.)[0]
        # Add class relative information on the design matrix
        H = np.random.choice(range(n_samples), replace=False,
                             size=int(high_risk_rate * n_samples))
        H_ = np.delete(range(n_samples), H)
        X[H, :nb_active_time_indep_features] += gap
        X[H_, :nb_active_time_indep_features] -= gap

        # Normalize time-independent features
        X_dot_xi = X.dot(xi)

        # Simulation of latent group
        if self.probit:
            pi_xi = probit(X_dot_xi)
        else:
            pi_xi = logistic_grad(X_dot_xi)
        u = np.random.rand(n_samples)
        G = (u < pi_xi).astype(int) + 1

        X = normalize(X)
        self.time_indep_features = X
        # Simulation of the random effects components
        r_l = 2  # Affine random effects
        r = r_l * n_long_features
        b, D = features_normal_cov_toeplitz(n_samples, r, cov_corr_long)
        self.long_cov = D

        # Simulation of the fixed effect parameters
        q_l = 2
        q = q_l * n_long_features  # linear time-varying features, so all q_l=2
        mean_1 = fixed_effect_mean_low_risk * n_long_features
        beta_1 = np.random.multivariate_normal(mean_1, np.diag(
            corr_fixed_effect * np.ones(q)))
        mean_2 = fixed_effect_mean_high_risk * n_long_features
        beta_2 = np.random.multivariate_normal(mean_2, np.diag(
            corr_fixed_effect * np.ones(q)))

        # Simulation of the fixed effect and association parameters
        nb_asso_param = 3
        nb_asso_features = n_long_features * nb_asso_param

        def simu_sparse_params():

            K = 2
            nb_nonactive_group = n_long_features - int(sparsity * n_long_features)

            gamma = []
            S_k = []
            coeff_val_asso = [coeff_val_asso_low_risk, coeff_val_asso_high_risk]
            active_beta_idx = [[], []]
            active_rdn_effect_idx = [[], []]
            for k in range(K):
                # set of nonactive group
                S_k.append(np.random.choice(n_long_features, nb_nonactive_group,
                                                      replace=False))
                gamma_k = np.zeros(nb_asso_features)
                for l in range(n_long_features):
                    if l not in S_k[k]:
                        active_beta_idx[k] += [1] * q_l
                        active_rdn_effect_idx[k] += [1] * r_l
                        start_idx = nb_asso_param * l
                        stop_idx = nb_asso_param * (l + 1)
                        gamma_k[start_idx : stop_idx] = coeff_val_asso[k]
                    else:
                        active_beta_idx[k] += [0] * q_l
                        active_rdn_effect_idx[k] += [0] * r_l
                gamma.append(gamma_k)
            return gamma, S_k, np.array(active_beta_idx), np.array(active_rdn_effect_idx)
        [gamma_1, gamma_2], S_k, active_beta_idx, active_rdn_effect_idx = \
            simu_sparse_params()
        beta_1 = beta_1 * active_beta_idx[0]
        beta_2 = beta_2 * active_beta_idx[1]
        self.fixed_effect_coeffs = [beta_1.reshape(-1, 1),
                                    beta_2.reshape(-1, 1)]
        b[G == 1] = b[G == 1] * active_rdn_effect_idx[0]
        b[G == 2] = b[G == 2] * active_rdn_effect_idx[1]
        self.asso_coeffs = [gamma_1, gamma_2]
        self.fixed_effect_coeffs = [beta_1.reshape(-1, 1), beta_2.reshape(-1, 1)]

        # Simulation of true times
        idx_2 = np.arange(0, r_l * n_long_features, r_l)
        idx_3 = np.arange(0, r_l * n_long_features, r_l) + 1
        idx_4 = np.arange(0, nb_asso_features, 3)
        idx_12 = np.concatenate(((idx_4 + 1), (idx_4 + 2)))
        idx_12.sort()

        iota_11 = (beta_1[idx_2] + b[G == 1][:, idx_2]).dot(
            gamma_1[idx_4]) + X[G == 1].dot(xi)
        iota_11 += b[G == 1].dot(gamma_1[idx_12])
        iota_12 = (beta_1[idx_3] + b[G == 1][:, idx_3]).dot(
            gamma_1[idx_4])
        iota_21 = (beta_2[idx_2] + b[G == 2][:, idx_2]).dot(
            gamma_2[idx_4]) + X[G == 2].dot(xi)
        iota_21 += b[G == 2].dot(gamma_2[idx_12])
        iota_22 = (beta_2[idx_3] + b[G == 2][:, idx_3]).dot(
            gamma_2[idx_4])
        self.iotas = {1: [iota_11, iota_21], 2: [iota_12, iota_22]}

        T_star = np.zeros(n_samples)
        n_samples_class_2 = np.sum(G == 2)
        n_samples_class_1 = n_samples - n_samples_class_2
        u_1 = np.random.rand(n_samples_class_1)
        u_2 = np.random.rand(n_samples_class_2)
        tmp = iota_12 + shape
        T_star[G == 1] = np.log(1 - tmp * np.log(u_1) /
                                (scale * shape * np.exp(iota_11))) / tmp
        tmp = iota_22 + shape
        T_star[G == 2] = np.log(1 - tmp * np.log(u_2) /
                                (scale * shape * np.exp(iota_21))) / tmp

        m = T_star.mean()
        # Simulation of the censoring
        C = np.random.exponential(scale=censoring_factor * m, size=n_samples)
        # Observed censored time
        T = np.minimum(T_star, C)
        # Impose ties
        T = np.ceil(T)
        self.times = T
        # Censoring indicator: 1 if it is a time of failure, 0 if censoring
        delta = (T_star <= C).astype(np.ushort)
        self.censoring = delta
        id = np.arange(n_samples)
        np.random.seed(seed)
        for i in range(n_samples):
            n_i = np.random.randint(1, 20)
            tmp = np.unique(np.random.uniform(0, T[i], n_i))
            times_i = [tmp] * n_long_features
            y_i = []
            for l in range(n_long_features):
                if G[i] == 1:
                    beta_l = beta_1[2 * l:2 * l + 2]
                else:
                    beta_l = beta_2[2 * l:2 * l + 2]

                b_l = b[i, 2 * l:2 * l + 2]
                n_il = len(times_i[l])
                U_il = np.c_[np.ones(n_il), times_i[l]]
                eps_il = np.random.normal(0, std_error, n_il)
                y_il = U_il.dot(beta_l) + U_il.dot(b_l) + eps_il
                y_i.append(y_il)
            time_dep_feat = ['time_dep_feat%s' % (l + 1)
                             for l in range(n_long_features)]
            Y_i = np.column_stack(
                (np.array([id[i]] * n_il), times_i[0], np.array(y_i).T))
            if i == 0:
                Y_data = Y_i
            else:
                Y_data = np.row_stack((Y_data, Y_i))

        Y = pd.DataFrame(data=Y_data, columns=["id", "T_long"] + time_dep_feat)
        self.event_times = T_star
        self.long_features = Y
        self.latent_class = G
        self.time_dep_feat = time_dep_feat
        self.S_k = S_k
        return X, Y, T, delta