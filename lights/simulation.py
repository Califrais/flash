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
        self.features = None
        self.labels = None

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

    time_indep_sparsity : `float`, default=0.5
        Percentage of sparsity induced in the time-independent coefficient
        vector. Must be in [0, 1].

    time_indep_coeff : `float`, default=1
        Value of the active coefficients in the time-independent coefficient
        vector.

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


    scale : `float`, default=1.0
        Scaling parameter to use in the distribution of times

    shape : `float`, default=1.0
        Shape parameter to use in the distribution of times

    censoring_factor : `float`, default=2.0
        Level of censoring. Increasing censoring_factor leads
        to less censored times and conversely.

    Attributes
    ----------
    times : `np.ndarray`, shape=(n_samples,)
        The simulated times of the event of interest

    latent_class : `np.ndarray`, shape=(n_samples,)
        The simulated latent classes

    censoring : `np.ndarray`, shape=(n_samples,)
        The simulated censoring indicator, where ``censoring[i] == 1``
        indicates that the time of the i-th individual is a failure
        time, and where ``censoring[i] == 0`` means that the time of
        the i-th individual is a censoring time

    Notes
    -----
    There is no intercept in this model
    """

    def __init__(self, verbose: bool = True, seed: int = None,
                 n_samples: int = 200, n_time_indep_features: int = 20,
                 time_indep_sparsity: float = 0.5, time_indep_coeff: float = 1.,
                 cov_corr_time_indep: float = 0.5, low_risk_rate: float = .75,
                 gap: float = .1, n_long_features: int = 5,
                 cov_corr_long: float = 0.5, corr_fixed_effect: float = 0.5,
                 var_error: float = 0.5, decay: float = 3,
                 baseline_hawkes_uniform_bounds: list = (.1, .5),
                 adjacency_hawkes_uniform_bounds: list = (.05, .1),
                 shape: float = 1.,
                 scale: float = 1.,
                 censoring_factor: float = 2.):
        Simulation.__init__(self, seed=seed, verbose=verbose)

        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        self.time_indep_sparsity = time_indep_sparsity
        self.time_indep_coeff = time_indep_coeff
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
        self.hawkes = None
        self.T = None
        self.G = None
        self.delta = None

    @property
    def time_indep_sparsity(self):
        return self._time_indep_sparsity

    @time_indep_sparsity.setter
    def time_indep_sparsity(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``time_indep_sparsity`` must be in (0, 1)")
        self._time_indep_sparsity = val

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
        time_indep_sparsity = self.time_indep_sparsity
        time_indep_coeff = self.time_indep_coeff
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

        nb_time_indep_active_features = int(
            n_time_indep_features * time_indep_sparsity)
        xi = np.zeros(n_time_indep_features)
        xi[0:nb_time_indep_active_features] = time_indep_coeff

        time_indep_features = features_normal_cov_toeplitz(n_samples,
                                                           n_time_indep_features,
                                                           cov_corr_time_indep)

        # Add class relative information on the design matrix    
        H = np.random.choice(range(n_samples),
                             size=int((1 - low_risk_rate) * n_samples),
                             replace=False)
        H_ = np.delete(range(n_samples), H)
        time_indep_features[H, :nb_time_indep_active_features] += gap
        time_indep_features[H_, :nb_time_indep_active_features] -= gap
        self.features = time_indep_features

        # Simulation of latent variables
        pi = self.logistic_grad(-time_indep_features.dot(xi))
        u = np.random.rand(n_samples)
        self.G = u <= 1 - pi

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

        ##

        # Simulation of true times
        T = np.empty(n_samples)

        m = T.mean()
        # Simulation of the censoring
        c = self.censoring_factor
        C = np.random.exponential(scale=c * m, size=n_samples)
        # Observed time
        self.times = np.minimum(T, C).astype(int)
        # Censoring indicator: 1 if it is a time of failure, 0 if censoring
        censoring = (T <= C).astype(np.ushort)
        self.censoring = censoring
        return self.features, self.times, self.censoring
