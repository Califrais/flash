def _P_func(self, X, pi_est, xi_ext):
    """Computes the sub objective function denoted P in the lights paper,
    to be minimized at each QNMCEM iteration using fmin_l_bfgs_b.

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    pi_est : `np.ndarray`, shape=(n_samples,)
        The estimated posterior probability of the latent class membership
        obtained by the E-step

    xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts

    Returns
    -------
    output : `float`
        The value of the P sub objective to be minimized at each QNMCEM step
    """
    xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
    pen = self._elastic_net_pen(xi_ext)
    u = xi_0 + X.dot(xi)
    sub_obj = (pi_est * u + self.logistic_loss(u)).mean()
    return sub_obj + pen


def _grad_P(self, X, pi_est, xi_ext):
    """Computes the gradient of the sub objective P

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    pi_est : `np.ndarray`, shape=(n_samples,)
        The estimated posterior probability of the latent class membership
        obtained by the E-step

    xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts

    Returns
    -------
    output : `float`
        The value of the P sub objective gradient
    """
    n_time_indep_features = self.n_time_indep_features
    n_samples = self.n_samples
    xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
    grad_pen = self._grad_elastic_net_pen(xi)
    u = xi_0 + X.dot(xi)
    if self.fit_intercept:
        X = np.concatenate((np.ones(n_samples).reshape(1, n_samples).T, X),
                           axis=1)
        grad_pen = np.concatenate([[0], grad_pen[:n_time_indep_features],
                                   [0], grad_pen[n_time_indep_features:]])
    grad = (X * (pi_est - self.logistic_grad(-u)).reshape(
        n_samples, 1)).mean(axis=0)
    grad_sub_obj = np.concatenate([grad, -grad])
    return grad_sub_obj + grad_pen


def _R_func(self, beta_ext, pi_est, E_g1, E_g2, E_g8, baseline_hazard,
            delta, indicator):
    """Computes the sub objective function denoted R in the lights paper,
    to be minimized at each QNMCEM iteration using fmin_l_bfgs_b.

    Parameters
    ----------
    beta_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts

    pi_est : `np.ndarray`, shape=(n_samples,)
        The estimated posterior probability of the latent class membership
        obtained by the E-step

    E_g1 : `np.ndarray`, shape=(n_samples, J, 2)
        The approximated expectations of function g1

    E_g2 : `np.ndarray`, shape=(n_samples, 2)
        The approximated expectations of function g2

    E_g8 : `np.ndarray`, shape=(n_samples, 2)
        The approximated expectations of function g8

    baseline_hazard : `np.ndarray`, shape=(n_samples,)
        The baseline hazard function evaluated at each censored time

    delta : `np.ndarray`, shape=(n_samples,)
        Censoring indicator

    indicator : `np.ndarray`, shape=(n_samples, J)
        The indicator matrix for comparing event times

    Returns
    -------
    output : `float`
        The value of the R sub objective to be minimized at each QNMCEM step
    """
    n_samples = delta.shape[0]
    pen = self._sparse_group_l1_pen(beta_ext)
    E_g1_ = E_g1.swapaxes(1, 2).swapaxes(0, 1)
    baseline_val = baseline_hazard.values.flatten()
    ind_ = indicator * 1
    sub_obj = E_g2 * delta.reshape(-1, 1) + E_g8 - np.sum(
        E_g1_ * baseline_val * ind_, axis=2).T
    sub_obj = (pi_est * sub_obj).sum()
    return -sub_obj / n_samples + pen


def _grad_R(self, beta_ext, E_g5, E_g6, delta, baseline_hazard, indicator):
    """Computes the gradient of the sub objective R

    Parameters
    ----------
    # TODO Van Tuan

    Returns
    -------
    output : `float`
        The value of the R sub objective gradient
    """
    beta = self.get_vect_from_ext(beta_ext)
    grad_pen = self._grad_sparse_group_l1_pen(beta)
    # TODO: handle gamma
    tmp1 = (E_g5.T * delta).T - np.sum(
        (E_g6.T * baseline_hazard.values * (indicator * 1).T).T, axis=1)
    tmp2 = 0
    grad = tmp1 + tmp2
    grad_sub_obj = np.concatenate([grad, -grad])
    return grad_sub_obj + grad_pen


def _Q_func(self, gamma_ext, pi_est, E_log_g1, E_g1, baseline_hazard, delta,
            indicator):
    """Computes the sub objective function denoted Q in the lights paper,
    to be minimized at each QNMCEM iteration using fmin_l_bfgs_b

    Parameters
    ----------
    gamma_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts

    pi_est : `np.ndarray`, shape=(n_samples,)
        The estimated posterior probability of the latent class membership
        obtained by the E-step

    E_log_g1 : `np.ndarray`, shape=(n_samples, J, 2)
        The approximated expectations of function logarithm of g1

    E_g1 : `np.ndarray`, shape=(n_samples, J, 2)
        The approximated expectations of function g1

    baseline_hazard : `np.ndarray`, shape=(n_samples,)
        The baseline hazard function evaluated at each censored time

    delta : `np.ndarray`, shape=(n_samples,)
        Censoring indicator

    indicator : `np.ndarray`, shape=(n_samples, J)
        The indicator matrix for comparing event times

    Returns
    -------
    output : `float`
        The value of the Q sub objective to be minimized at each QNMCEM step
    """
    n_samples = delta.shape[0]
    pen = self._sparse_group_l1_pen(gamma_ext)

    E_g1_ = E_g1.swapaxes(1, 2).swapaxes(0, 1)
    baseline_val = baseline_hazard.values.flatten()
    ind_ = indicator * 1
    sub_obj = E_log_g1 * delta.reshape(-1, 1) - np.sum(
        E_g1_ * baseline_val * ind_, axis=2).T
    sub_obj = (pi_est * sub_obj).sum()
    return -sub_obj / n_samples + pen


def _grad_Q(self, gamma_ext):
    """Computes the gradient of the sub objective Q

    Parameters
    ----------
    # TODO Van Tuan

    Returns
    -------
    output : `float`
        The value of the Q sub objective gradient
    """
    gamma = self.get_vect_from_ext(gamma_ext)
    grad_pen = self._grad_sparse_group_l1_pen(gamma)
    # TODO Van Tuan
    grad = 0
    grad_sub_obj = np.concatenate([grad, -grad])
    return grad_sub_obj + grad_pen