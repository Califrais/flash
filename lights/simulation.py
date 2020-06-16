# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from datetime import datetime
from time import time
from scipy.linalg.special_matrices import toeplitz
from tick.hawkes import SimuHawkesExpKernels
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

    n_samples : `int`, default=200
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

    low_risk_rate : `float`, default=0.75
        Proportion of desired low risk samples rate

    gap : `float`, default=0.1
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

    decay : `float`, default=3
        Decay of exponential kernels for the multivariate Hawkes processes to
        generate measurement times

    baseline_hawkes_uniform_bounds : `list`, default=(.1, .5)
        Bounds of the uniform distribution used to generate baselines of
        measurement times intensities

    adjacency_hawkes_uniform_bounds : `list`, default=(.05, .1)
        Bounds of the uniform distribution used to generate sparse adjacency
        matrix for measurement times intensities

    scale : `float`, default=2.
        Scaling parameter of the Weibull distribution of the baseline

    shape : `float`, default=.1
        Shape parameter of the Weibull distribution of the baseline

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
                 n_samples: int = 200, n_time_indep_features: int = 20,
                 sparsity: float = 0.7, coeff_val: float = 1.,
                 cov_corr_time_indep: float = 0.5, low_risk_rate: float = .75,
                 gap: float = .1, n_long_features: int = 5,
                 cov_corr_long: float = 0.5, corr_fixed_effect: float = 0.5,
                 var_error: float = 0.5, decay: float = 3,
                 baseline_hawkes_uniform_bounds: list = (.1, .5),
                 adjacency_hawkes_uniform_bounds: list = (.05, .1),
                 shape: float = 1., scale: float = 1.,
                 censoring_factor: float = 2.):
        Simulation.__init__(self, seed=seed, verbose=verbose)

        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        self.sparsity = sparsity
        self.coeff_val = coeff_val
        self.cov_corr_time_indep = cov_corr_time_indep
        self.low_risk_rate = low_risk_rate
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
    def low_risk_rate(self):
        return self._low_risk_rate

    @low_risk_rate.setter
    def low_risk_rate(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``low_risk_rate`` must be in (0, 1)")
        self._low_risk_rate = val

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if val <= 0:
            raise ValueError("``shape`` must be strictly positive")
        self._shape = val

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
        coeff_val = self.coeff_val
        cov_corr_time_indep = self.cov_corr_time_indep
        low_risk_rate = self.low_risk_rate
        gap = self.gap
        n_long_features = self.n_long_features
        cov_corr_long = self.cov_corr_long
        corr_fixed_effect = self.corr_fixed_effect
        var_error = self.var_error
        decay = self.decay
        baseline_hawkes_uniform_bounds = self.baseline_hawkes_uniform_bounds
        adjacency_hawkes_uniform_bounds = self.adjacency_hawkes_uniform_bounds

        # Simulation of time-independent coefficient vector
        nb_active_time_indep_features = int(n_time_indep_features * sparsity)
        xi = np.zeros(n_time_indep_features)
        xi[0:nb_active_time_indep_features] = coeff_val

        # Simulation of time-independent features
        X = features_normal_cov_toeplitz(n_samples, n_time_indep_features,
                                         cov_corr_time_indep)
        # Add class relative information on the design matrix    
        H = np.random.choice(range(n_samples),
                             size=int((1 - low_risk_rate) * n_samples),
                             replace=False)
        H_ = np.delete(range(n_samples), H)
        X[H, :nb_active_time_indep_features] += gap
        X[H_, :nb_active_time_indep_features] -= gap
        self.time_indep_features = X

        # Simulation of latent variables
        pi = self.logistic_grad(-X.dot(xi))
        u = np.random.rand(n_samples)
        G = u <= 1 - pi
        self.latent_class = G

        # Simulation of the random effects components
        r = 2 * n_long_features  # linear time-varying features, so all r_l=2
        b = features_normal_cov_toeplitz(n_samples, r, cov_corr_long)

        # Simulation of the fixed effect parameters
        q = 2 * n_long_features  # linear time-varying features, so all q_l=2
        beta_0 = - np.random.multivariate_normal(np.ones(q), np.diag(
            corr_fixed_effect * np.ones(q)))
        beta_1 = np.random.multivariate_normal(np.ones(q), np.diag(
            corr_fixed_effect * np.ones(q)))

        # Simulation of the association parameters
        nb_asso_features = n_long_features * 4  # 4: nb of asso param
        nb_active_asso_features = int(nb_asso_features * sparsity / 2)  # K=2
        gamma_0 = np.zeros(nb_asso_features)
        gamma_1 = gamma_0.copy()
        gamma_0[0:nb_active_asso_features] = coeff_val
        gamma_1[nb_active_asso_features + 1:
                2 * nb_active_asso_features] = coeff_val

        # Simulation of true times
        T = np.empty(n_samples)
        # n_samples_class_1 = np.sum(G)
        # n_samples_class_0 = n_samples - n_samples_class_1

        # T[G == 0] =
        # T[G == 1] =

        m = T.mean()
        # Simulation of the censoring
        c = self.censoring_factor
        C = np.random.exponential(scale=c * m, size=n_samples)
        # Observed time
        self.times = np.minimum(T, C).astype(int)
        # Censoring indicator: 1 if it is a time of failure, 0 if censoring
        delta = (T <= C).astype(np.ushort)
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
