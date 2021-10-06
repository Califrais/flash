from datetime import datetime
from time import time
from scipy.linalg.special_matrices import toeplitz
from tick.hawkes import SimuHawkesExpKernels
from scipy.stats import uniform, beta
from scipy.sparse import random
from lights.base.base import normalize, logistic_grad
import numpy as np
import pandas as pd


def features_normal_cov_toeplitz(n_samples: int = 200, n_features: int = 10,
                                 rho: float = 0.5, cst: float = 1.):
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

    cst : `float`, default=1.
        Multiplicative constant

    Returns
    -------
    features : `np.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix

    cov : `np.ndarray`, shape=(n_features, n_features)
        The simulated variance-covariance matrix
    """
    cov = (cst ** 2) * toeplitz(rho ** np.arange(0, n_features))
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

    def __init__(self, seed=None, verbose=True):
        # Set default parameters
        self.seed = seed
        self.verbose = verbose
        if self.seed is not None:
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

    n_time_indep_features : `int`, default=20
        Number of time-independent features

    sparsity : `float`, default=0.7
        Proportion of both time-independent and association features active
        coefficients vector. Must be in [0, 1].

    coeff_val_time_indep : `float`, default=1.
        Value of the active coefficients in the time-independent coefficient
        vectors

    coeff_val_asso : `float`, default=.05
        Value of the coefficients parameter used in the association coefficient
        vectors

    cov_corr_time_indep : `float`, default=0.5
        Correlation to use in the Toeplitz covariance matrix for the
        time-independent features

    high_risk_rate : `float`, default=0.4
        Proportion of desired high risk samples rate

    gap : `float`, default=0.5
        Gap value to create high/low risk groups in the time-independent
        features

    n_long_features : `int`, default=5
        Number of longitudinal features

    cov_corr_long : `float`, default=0.1
        Correlation to use in the Toeplitz covariance matrix for the random
        effects simulation

    fixed_effect_mean_low_risk : `tuple`, default=(-.8, .2)
        Mean vector of the gaussian used to generate the fixed effect parameters
        for the low risk group

    fixed_effect_mean_high_risk : `tuple`, default=(1.5, .8)
        Mean vector of the gaussian used to generate the fixed effect parameters
        for the high risk group

    corr_fixed_effect : `float`, default=0.01
        Correlation value to use in the diagonal covariance matrix for the
        fixed effect simulated feature

    std_error : `float`, default=0.5
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

    scale : `float`, default=.001
        Scaling parameter of the Gompertz distribution of the baseline

    shape : `float`, default=.1
        Shape parameter of the Gompertz distribution of the baseline

    censoring_factor : `float`, default=2.0
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

    fixed_effect_coeffs : `list`, [beta_0, beta_1]
        Simulated fixed effect coefficient vectors per group

    asso_coeffs : `list`, [gamma_0, gamma_1]
        Simulated association parameters per group

    iotas : `dict`, {1: [iota_01, iota_11], 2: [iota_02, iota_12]}
        Simulated linear time-varying features per group in the Cox model used
        to simulate event times

    event_times : `np.ndarray`, shape=(n_samples,)
            The simulated times of the event of interest

    hawkes : `list` of `tick.hawkes.simulation.simu_hawkes_exp_kernels`
        Store the multivariate Hawkes processes with exponential kernels
        used to simulate measurement times for intensities plotting purpose

    Notes
    -----
    There is no intercept in this model
    """

    def __init__(self, verbose: bool = True, seed: int = None,
                 n_samples: int = 1000, n_time_indep_features: int = 10,
                 sparsity: float = .7, coeff_val_time_indep: float = 1.,
                 coeff_val_asso: float = .1, cov_corr_time_indep: float = .5,
                 high_risk_rate: float = .4, gap: float = .5,
                 n_long_features: int = 10, cov_corr_long: float = .01,
                 fixed_effect_mean_low_risk: tuple = (-.8, .2),
                 fixed_effect_mean_high_risk: tuple = (1.5, .8),
                 corr_fixed_effect: float = .01,
                 std_error: float = .5, decay: float = 3.,
                 baseline_hawkes_uniform_bounds: list = (.1, 1.),
                 adjacency_hawkes_uniform_bounds: list = (.1, .2),
                 shape: float = .1, scale: float = .001,
                 censoring_factor: float = 2):
        Simulation.__init__(self, seed=seed, verbose=verbose)

        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        self.sparsity = sparsity
        self.coeff_val_time_indep = coeff_val_time_indep
        self.coeff_val_asso = coeff_val_asso
        self.cov_corr_time_indep = cov_corr_time_indep
        self.high_risk_rate = high_risk_rate
        self.gap = gap
        self.n_long_features = n_long_features
        self.cov_corr_long = cov_corr_long
        self.fixed_effect_mean_low_risk = fixed_effect_mean_low_risk
        self.fixed_effect_mean_high_risk = fixed_effect_mean_high_risk
        self.corr_fixed_effect = corr_fixed_effect
        self.std_error = std_error
        self.decay = decay
        self.baseline_hawkes_uniform_bounds = baseline_hawkes_uniform_bounds
        self.adjacency_hawkes_uniform_bounds = adjacency_hawkes_uniform_bounds
        self.shape = shape
        self.scale = scale
        self.censoring_factor = censoring_factor

        # Attributes that will be instantiated afterwards
        self.latent_class = None
        self.time_indep_coeffs = None
        self.long_cov = None
        self.event_times = None
        self.fixed_effect_coeffs = None
        self.asso_coeffs = None
        self.iotas = None
        self.G = None
        self.hawkes = []

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
        coeff_val_asso = self.coeff_val_asso
        cov_corr_time_indep = self.cov_corr_time_indep
        high_risk_rate = self.high_risk_rate
        gap = self.gap
        n_long_features = self.n_long_features
        cov_corr_long = self.cov_corr_long
        fixed_effect_mean_low_risk = self.fixed_effect_mean_low_risk
        fixed_effect_mean_high_risk = self.fixed_effect_mean_high_risk
        corr_fixed_effect = self.corr_fixed_effect
        std_error = self.std_error
        decay = self.decay
        baseline_hawkes_uniform_bounds = self.baseline_hawkes_uniform_bounds
        adjacency_hawkes_uniform_bounds = self.adjacency_hawkes_uniform_bounds
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
                                         cov_corr_time_indep)[0]
        # Add class relative information on the design matrix
        H = np.random.choice(range(n_samples), replace=False,
                             size=int(high_risk_rate * n_samples))
        H_ = np.delete(range(n_samples), H)
        X[H, :nb_active_time_indep_features] += gap
        X[H_, :nb_active_time_indep_features] -= gap

        # Normalize time-independent features
        X = normalize(X)
        self.time_indep_features = X
        X_dot_xi = X.dot(xi)

        # Simulation of latent group
        pi_xi = logistic_grad(X_dot_xi)
        u = np.random.rand(n_samples)
        G = (u < pi_xi).astype(int)
        self.G = G

        # Simulation of the random effects components
        r_l = 2  # Affine random effects
        r = r_l * n_long_features
        b, D = features_normal_cov_toeplitz(n_samples, r, cov_corr_long, .1)
        self.long_cov = D

        # Simulation of the fixed effect parameters
        q = 2 * n_long_features  # linear time-varying features, so all q_l=2
        mean_0 = fixed_effect_mean_low_risk * n_long_features
        beta_0 = np.random.multivariate_normal(mean_0, np.diag(
            corr_fixed_effect * np.ones(q)))
        mean_1 = fixed_effect_mean_high_risk * n_long_features
        beta_1 = np.random.multivariate_normal(mean_1, np.diag(
            corr_fixed_effect * np.ones(q)))
        self.fixed_effect_coeffs = [beta_0.reshape(-1, 1),
                                    beta_1.reshape(-1, 1)]

        # Simulation of the fixed effect and association parameters
        nb_asso_param = 4
        nb_asso_features = n_long_features * nb_asso_param
        fixed_effect_mean = [fixed_effect_mean_low_risk,
                             fixed_effect_mean_high_risk]

        def simu_sparse_params():

            K = 2
            nb_nonactive_group = n_long_features - int(sparsity * n_long_features)

            # set of nonactive group for 2 classes
            S_k = np.array_split(np.random.choice(n_long_features,
                                K * nb_nonactive_group, replace=False), K)
            # number of active features in each group
            # nb_active_features = int(sparsity * nb_asso_param)

            #TODO : for testing only group lasso first
            nb_active_features = nb_asso_param

            gamma = []
            beta = []
            q_l = 2
            for k in range(K):
                gamma_k = np.zeros(nb_asso_features)
                beta_k = np.zeros(2 * n_long_features)
                for l in range(n_long_features):
                    if l not in S_k[k]:
                        gamma_k[nb_asso_param * l:
                          nb_asso_param * l + nb_active_features] = coeff_val_asso
                        mean_beta_k_l = fixed_effect_mean[k]
                        cov_beta_k_l = np.diag(active_corr_fixed_effect * np.ones(q_l))
                    else:
                        mean_beta_k_l = np.zeros(q_l)
                        cov_beta_k_l = np.diag(non_active_corr_fixed_effect * np.ones(q_l))
                    beta_k[q_l * l: q_l * (l + 1)] = \
                        np.random.multivariate_normal(mean_beta_k_l, cov_beta_k_l)
                gamma.append(gamma_k)
                beta.append(beta_k)
            return gamma, beta
        [gamma_0, gamma_1], [beta_0, beta_1] = simu_sparse_params()
        self.asso_coeffs = [gamma_0, gamma_1]
        self.fixed_effect_coeffs = [beta_0.reshape(-1, 1), beta_1.reshape(-1, 1)]

        # Simulation of true times
        idx_2 = np.arange(0, nb_asso_features, 2)
        idx_4 = np.arange(0, nb_asso_features, 4)
        idx_34 = np.concatenate((idx_4, (idx_4 - 1)[1:],
                                 [nb_asso_features - 1]))
        idx_34.sort()
        idx_3 = np.arange(0, 2 * n_long_features, 2) + 1

        tmp_0 = np.add.reduceat(gamma_0, idx_2)
        tmp_1 = np.add.reduceat(gamma_1, idx_2)

        iota_01 = b[G == 0].dot(tmp_0)
        iota_01 += gamma_0[idx_34].dot(beta_0)
        iota_02 = (beta_0[idx_3] + b[G == 0][:, idx_3]).dot(gamma_0[idx_4])
        iota_11 = b[G == 1].dot(tmp_1)
        iota_11 += gamma_1[idx_34].dot(beta_1)
        iota_12 = (beta_1[idx_3] + b[G == 1][:, idx_3]).dot(gamma_1[idx_4])
        self.iotas = {1: [iota_01, iota_11], 2: [iota_02, iota_12]}

        T_star = np.zeros(n_samples)
        n_samples_class_1 = np.sum(G)
        n_samples_class_0 = n_samples - n_samples_class_1
        u_0 = np.random.rand(n_samples_class_0)
        u_1 = np.random.rand(n_samples_class_1)

        tmp = iota_02 + shape
        T_star[G == 0] = np.log(1 - tmp * np.log(u_0) /
                                (scale * shape * np.exp(iota_01))) / tmp
        tmp = iota_12 + shape
        T_star[G == 1] = np.log(1 - tmp * np.log(u_1) /
                                (scale * shape * np.exp(iota_11))) / tmp

        m = T_star.mean()
        # Simulation of the censoring
        C = np.random.exponential(scale=censoring_factor * m, size=n_samples)
        # Observed censored time
        T = np.minimum(T_star, C)
        self.times = T
        # Censoring indicator: 1 if it is a time of failure, 0 if censoring
        delta = (T_star <= C).astype(np.ushort)
        self.censoring = delta

        # Simulation of the time up to which one has longitudinal data
        t_max = np.multiply(T, 1 - beta.rvs(2, 5, size=n_samples))

        # Simulation of the longitudinal features
        decays = decay * np.ones((n_long_features, n_long_features))
        Y = pd.DataFrame(columns=['long_feature_%s' % (l + 1)
                                  for l in range(n_long_features)])

        # Simulation of the measurement times of the longitudinal processes
        # using multivariate Hawkes
        a_, b_ = adjacency_hawkes_uniform_bounds
        rvs_adjacency = uniform(a_, b_).rvs
        adjacency = random(n_long_features, n_long_features, density=0.3,
                           data_rvs=rvs_adjacency, random_state=seed).todense()
        np.fill_diagonal(adjacency, rvs_adjacency(size=n_long_features))

        a_, b_ = baseline_hawkes_uniform_bounds
        baseline = uniform(a_, b_).rvs(size=n_long_features, random_state=seed)

        N_il = np.zeros((n_samples, n_long_features))
        # TODO : delete N_il after tests
        for i in range(n_samples):
            hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays,
                                          baseline=baseline, verbose=False,
                                          end_time=t_max[i], seed=seed + i)
            hawkes.simulate()
            self.hawkes += [hawkes]
            times_i = hawkes.timestamps

            y_i = []
            for l in range(n_long_features):
                if G[i] == 0:
                    beta_l = beta_0[2 * l:2 * l + 2]
                else:
                    beta_l = beta_1[2 * l:2 * l + 2]

                b_l = b[i, 2 * l:2 * l + 2]
                if t_max[i] not in times_i[l]:
                    times_i[l] = np.append(times_i[l], t_max[i])
                if len(times_i[l]) > 10:
                    times_i[l] = np.sort(np.random.choice(times_i[l],
                                                          size=10,
                                                          replace=False))
                n_il = len(times_i[l])
                N_il[i, l] = n_il
                U_il = np.c_[np.ones(n_il), times_i[l]]
                eps_il = np.random.normal(0, std_error, n_il)
                y_i += [pd.Series(U_il.dot(beta_l) + U_il.dot(b_l) + eps_il,
                                  index=times_i[l])]

            Y.loc[i] = y_i
        self.event_times = T_star
        self.long_features = Y
        self.latent_class = G
        self.N_il = N_il

        return X, Y, np.ceil(T), delta
