# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from datetime import datetime
from time import time
from scipy.linalg.special_matrices import toeplitz
from tick.hawkes import SimuHawkesExpKernels
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform
from scipy.sparse import random
import numpy as np


def features_normal_cov_toeplitz(n_samples: int = 200, n_features: int = 10,
                                 rho: float = 0.5):
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

    Returns
    -------
    output : `np.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix
    """
    cov = toeplitz(rho ** np.arange(0, n_features))
    return np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)


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
    """This is an abstract simulation class that inherits form BaseClass
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

    coeff_val : `float`, default=1
        Value of the active coefficients in both the time-independent and
        association coefficient vectors

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

    cov_corr_long : `float`, default=0.5
        Correlation to use in the Toeplitz covariance matrix for the random
        effects simulation

    corr_fixed_effect : `float`, default=0.5
        Correlation value to use in the diagonal covariance matrix for the
        fixed effect simulation

    var_error : `float`, default=0.5
        Variance for the error term of the longitudinal process

    decay : `float`, default=3.0
        Decay of exponential kernels for the multivariate Hawkes processes to
        generate measurement times

    baseline_hawkes_uniform_bounds : `list`, default=(.1, .5)
        Bounds of the uniform distribution used to generate baselines of
        measurement times intensities

    adjacency_hawkes_uniform_bounds : `list`, default=(.05, .1)
        Bounds of the uniform distribution used to generate sparse adjacency
        matrix for measurement times intensities

    scale : `float`, default=.5
        Scaling parameter of the Gompertz distribution of the baseline

    shape : `float`, default=.5
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

    hawkes : `tick.hawkes.simulation.simu_hawkes_exp_kernels`
        Multivariate Hawkes process with exponential kernels used to
        simulate measurement times

    Notes
    -----
    There is no intercept in this model
    """

    def __init__(self, verbose: bool = True, seed: int = None,
                 n_samples: int = 1000, n_time_indep_features: int = 20,
                 sparsity: float = 0.7, coeff_val_time_indep: float = 12.,
                 coeff_val_time_dep: float = 1., cov_corr_time_indep: float = 0.5,
                 high_risk_rate: float = .4, gap: float = .5, n_long_features: int = 5,
                 cov_corr_long: float = 0.5, corr_fixed_effect: float = 0.5,
                 var_error: float = 0.5, decay: float = 3.,
                 baseline_hawkes_uniform_bounds: list = (.1, .5),
                 adjacency_hawkes_uniform_bounds: list = (.05, .1),
                 shape: float = .5, scale: float = .5,
                 censoring_factor: float = 2):
        Simulation.__init__(self, seed=seed, verbose=verbose)

        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        self.sparsity = sparsity
        self.coeff_val_time_indep = coeff_val_time_indep
        self.coeff_val_time_dep = coeff_val_time_dep
        self.cov_corr_time_indep = cov_corr_time_indep
        self.high_risk_rate = high_risk_rate
        self.gap = gap
        self.n_long_features = n_long_features
        self.cov_corr_long = cov_corr_long
        self.corr_fixed_effect = corr_fixed_effect
        self.var_error = var_error
        self.decay = decay
        self.baseline_hawkes_uniform_bounds = baseline_hawkes_uniform_bounds
        self.adjacency_hawkes_uniform_bounds = adjacency_hawkes_uniform_bounds
        self.shape = shape
        self.scale = scale
        self.censoring_factor = censoring_factor

        # Attributes that will be instantiated afterwards
        self.latent_class = None
        self.hawkes = None

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

    @staticmethod
    def logistic_grad(z):
        """Overflow proof computation of 1 / (1 + exp(-z)))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = 1. / (1. + np.exp(-z[idx_pos]))
        res[idx_neg] = 1 - 1. / (1. + np.exp(z[idx_neg]))
        return res

    @simulation_method
    def simulate(self):
        """Launch simulation of the data

        Returns
        -------
        X : `numpy.ndarray`, shape=(n_samples, n_time_indep_features)
            The simulated time-independent features matrix

        Y :
            The simulated longitudinal data

        T : `np.ndarray`, shape=(n_samples,)
            The simulated times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            The simulated censoring indicator
        """
        verbose = self.verbose
        seed = self.seed
        n_samples = self.n_samples
        n_time_indep_features = self.n_time_indep_features
        sparsity = self.sparsity
        coeff_val_time_indep = self.coeff_val_time_indep/n_time_indep_features
        cov_corr_time_indep = self.cov_corr_time_indep
        high_risk_rate = self.high_risk_rate
        gap = self.gap
        n_long_features = self.n_long_features
        coeff_val_time_dep = self.coeff_val_time_dep/n_long_features
        cov_corr_long = self.cov_corr_long
        corr_fixed_effect = self.corr_fixed_effect
        var_error = self.var_error
        decay = self.decay
        baseline_hawkes_uniform_bounds = self.baseline_hawkes_uniform_bounds
        adjacency_hawkes_uniform_bounds = self.adjacency_hawkes_uniform_bounds
        shape = self.shape
        scale = self.scale
        censoring_factor = self.censoring_factor

        # Simulation of latent variables
        u = np.random.rand(n_samples)
        G = (u <= high_risk_rate).astype(int)
        self.latent_class = G

        # Simulation of time-independent coefficient vector
        nb_active_time_indep_features = int(n_time_indep_features * sparsity)
        xi = np.zeros(n_time_indep_features)
        xi[0:nb_active_time_indep_features] = coeff_val_time_indep

        # Simulation of time-independent features
        X = features_normal_cov_toeplitz(n_samples, n_time_indep_features,
                                         cov_corr_time_indep)
        # Add class relative information on the design matrix
        X[G == 1, :nb_active_time_indep_features] += gap
        X[G == 0, :nb_active_time_indep_features] -= gap

        scaler = MinMaxScaler(feature_range=(-1, 0))
        X = scaler.fit_transform(X)

        self.time_indep_features = X
        X_dot_xi = X.dot(xi)
        # pi = self.logistic_grad(X_dot_xi)

        # Simulation of the random effects components
        r = 2 * n_long_features  # linear time-varying features, so all r_l=2
        b = 0.2 * features_normal_cov_toeplitz(n_samples, r, cov_corr_long)

        # Simulation of the fixed effect parameters
        q = 2 * n_long_features  # linear time-varying features, so all q_l=2
        beta_0 = -0.2 * np.random.multivariate_normal(np.ones(q), np.diag(
            corr_fixed_effect * np.ones(q)))
        beta_1 = 0.6 * np.random.multivariate_normal(np.ones(q), np.diag(
            corr_fixed_effect * np.ones(q)))

        # Simulation of the association parameters
        nb_asso_features = n_long_features * 4  # 4: nb of asso param

        def simu_sparse_asso_features(k):
            gamma = np.zeros(nb_asso_features)
            low_limit = int((k * sparsity * n_long_features) / 2) + 1
            high_limit = int(((k + 1) * sparsity * n_long_features) / 2)
            S_k = np.arange(low_limit, high_limit + 1)
            for l in range(n_long_features):
                if (l + 1) < n_long_features * sparsity:
                    gamma[4 * l: 4 * (l + 1)] += coeff_val_time_dep
                if (l + 1) in S_k:
                    gamma[4 * l: 4 * (l + 1)] += coeff_val_time_dep
            return gamma

        gamma_0 = simu_sparse_asso_features(0)
        gamma_1 = simu_sparse_asso_features(1)

        # Simulation of true times
        idx_2 = np.arange(0, nb_asso_features, 2)
        idx_4 = np.arange(0, nb_asso_features, 4)
        idx_34 = np.concatenate((idx_4, (idx_4 - 1)[1:], [nb_asso_features - 1]))
        idx_34.sort()
        idx_3 = np.arange(0, 2 * n_long_features, 2) + 1

        tmp_0 = np.add.reduceat(gamma_0, idx_2)
        tmp_1 = np.add.reduceat(gamma_1, idx_2)

        iota_01 = X_dot_xi[G == 0] + b[G == 0].dot(tmp_0) \
                  + gamma_0[idx_34].dot(beta_0)
        iota_02 = (beta_0[idx_3] + b[G == 0][:, idx_3]).dot(gamma_0[idx_4])
        iota_11 = X_dot_xi[G == 1] + b[G == 1].dot(tmp_1) \
                  + gamma_1[idx_34].dot(beta_1)
        iota_12 = (beta_1[idx_3] + b[G == 1][:, idx_3]).dot(gamma_1[idx_4])

        T_star = np.empty(n_samples)
        n_samples_class_1 = np.sum(G)
        n_samples_class_0 = n_samples - n_samples_class_1
        u_0 = np.random.rand(n_samples_class_0)
        u_1 = np.random.rand(n_samples_class_1)

        tmp = iota_02 + shape
        T_star[G == 0] = np.log(1 - tmp * np.log(u_0) /
                                (scale * np.exp(iota_01))) / tmp
        tmp = iota_12 + shape
        T_star[G == 1] = np.log(1 - tmp * np.log(u_1) /
                                (scale * np.exp(iota_11))) / tmp

        m = T_star.mean()
        # Simulation of the censoring
        C = np.random.exponential(scale=censoring_factor * m, size=n_samples)
        # Observed censored time
        T = np.minimum(T_star, C)
        self.times = T
        # Censoring indicator: 1 if it is a time of failure, 0 if censoring
        delta = (T_star <= C).astype(np.ushort)
        self.censoring = delta

        # Simulation of the measurement times using multivariate Hawkes
        a, b = adjacency_hawkes_uniform_bounds
        rvs = uniform(a, b).rvs
        adjacency = random(n_long_features, n_long_features, density=0.3,
                           data_rvs=rvs).todense()
        np.fill_diagonal(adjacency, rvs(size=n_long_features))
        a, b = baseline_hawkes_uniform_bounds
        baseline = uniform(a, b).rvs(size=n_long_features)
        decays = decay * np.ones((n_long_features, n_long_features))
        hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays,
                                      baseline=baseline, verbose=verbose,
                                      end_time=100, seed=seed)

        # TODO: generate T first, then t_i^max, and finally measurment times for long processes with end_time=t_i^max

        if verbose:
            # only useful to plot Hawkes multivariate intensities, but
            # careful: it increases running time!
            dt = 0.01
            hawkes.track_intensity(dt)
        hawkes.simulate()
        self.hawkes = hawkes

        # Simulation of the longitudinal features
        Y = None
        self.long_features = Y

        return X, Y, T, delta
