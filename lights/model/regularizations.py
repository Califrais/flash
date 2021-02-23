import numpy as np
from copt.penalty import L1Norm, GroupL1


class ElasticNet:
    """A class to define the Elastic Net penalty

    Parameters
    ----------
    l_pen : `float`
        Level of penalization for the ElasticNet

    eta: `float`
        The ElasticNet mixing parameter, with 0 <= eta <= 1.
        For eta = 1 this is ridge (L2) regularization
        For eta = 0 this is lasso (L1) regularization
        For 0 < eta < 1, the regularization is a linear combination of L1 and L2
    """

    def __init__(self, l_pen, eta):
        self.l_pen = l_pen
        self.eta = eta

    @property
    def l_pen(self):
        return self._l_pen

    @l_pen.setter
    def l_pen(self, val):
        if not val >= 0:
            raise ValueError("``l_pen`` must be non negative")
        self._l_pen = val

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta`` must be in (0, 1)")
        self._eta = val

    def pen(self, v):
        """Computes the elasticNet penalization of vector v

        Parameters
        ----------
        v: `np.ndarray`
            A coefficient vector

        Returns
        -------
        output : `float`
            The value of the elasticNet penalization part of vector v
        """
        l_pen, eta, = self.l_pen, self.eta
        return l_pen * ((1. - eta) * abs(v).sum() +
                        .5 * eta * np.linalg.norm(v) ** 2)

    def grad(self, v):
        """Computes the gradient of the elasticNet penalization of a vector v

        Parameters
        ----------
        v : `np.ndarray`
            A coefficient vector

        Returns
        -------
        grad : `np.array`
            The gradient of the elasticNet penalization part of vector v
        """
        l_pen, eta = self.l_pen, self.eta
        dim = v.shape[0]
        grad = np.zeros(2 * dim)
        # Gradient of lasso penalization
        grad += l_pen * (1 - eta)
        # Gradient of ridge penalization
        grad_pos = l_pen * eta * v
        grad[:dim] += grad_pos
        grad[dim:] -= grad_pos
        return grad

class L2Norm_Sq:
    """L1 norm squared, that is, the sum of squared values:

    .. math::
          \\alpha\\sum_i^d x_i^2

    Args:
    alpha: float
        constant multiplying the L2 norm squared

    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return .5 * self.alpha * np.abs(x).sum()

    def prox(self, x, step_size):
        """Proximal operator of the L1 norm.

        This routine can be used in gradient-based methods like
        minimize_proximal_gradient, minimize_three_split and
        minimize_primal_dual.
        """
        return x / (1 + self.alpha * step_size)

class ElasticNet_prox:
    """A class to define the proximal operator of the ElasticNet penalty

    Parameters
    ----------
    l_pen : `float`
        Level of penalization for the ElasticNet

    eta: `float`
        The ElasticNet mixing parameter, with 0 <= eta <= 1
        For eta = 1 this is ridge (squared L2) regularization
        For eta = 0 this is lasso (L1) regularization
        For 0 < eta < 1, the regularization is a linear combination of L1 and
        squared L2
    """

    def __init__(self, l_pen, eta):
        self.l_pen = l_pen
        self.eta = eta
        self.L1 = L1Norm(l_pen * (1 - eta))
        self.L2_Sq = L2Norm_Sq(l_pen * eta)

    @property
    def l_pen(self):
        return self._l_pen

    @l_pen.setter
    def l_pen(self, val):
        if not val >= 0:
            raise ValueError("``l_pen`` must be non negative")
        self._l_pen = val

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta`` must be in (0, 1)")
        self._eta = val

    def pen(self, v):
        L1 = self.L1.__call__(v)
        L2_Sq = self.L2_Sq.__call__(v)
        return L1 + L2_Sq

    def prox(self, v, step_size):
        """Computes the proximal operator for the ElasticNet penalization of a
         vector v

        Parameters
        ----------
        v : `np.ndarray`
            A coefficient vector

        step_size : `float`
            Value of the step size for the optimization update at the current
            solver iteration

        Returns
        -------
        grad : `np.array`
            The proximal operator of the ElasticNet computed on vector v
        """
        L1_prox = self.L1.prox(v, step_size)
        return self.L2_Sq.prox(L1_prox, step_size)

class SparseGroupL1:
    """A class to define the proximal operator of the Sparse group Lasso penalty

    Parameters
    ----------
    l_pen : `float`
        Level of penalization for the Sparse group Lasso

    eta: `float`
        The Sparse Group l1 mixing parameter, with 0 <= eta <= 1
        For eta = 1 this is group lasso regularization
        For eta = 0 this is lasso (L1) regularization
        For 0 < eta < 1, the regularization is a linear combination of L1 and
        group lasso

    groups: list of lists
        The groups to be considered for the group lasso part
    """

    def __init__(self, l_pen, eta, groups):
        self.l_pen = l_pen
        self.eta = eta
        self.groups = groups
        self.L1 = L1Norm(l_pen * (1 - eta))
        self.GL1 = GroupL1(l_pen * eta, self.groups)

    @property
    def l_pen(self):
        return self._l_pen

    @l_pen.setter
    def l_pen(self, val):
        if not val >= 0:
            raise ValueError("``l_pen`` must be non negative")
        self._l_pen = val

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta`` must be in (0, 1)")
        self._eta = val

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, val):
        for i, g in enumerate(val):
            if not np.all(np.diff(g) == 1):
                raise ValueError("Groups must be contiguous")
            if i > 0 and val[i - 1][-1] >= g[0]:
                raise ValueError("Groups must be increasing")
        self._groups = val

    def pen(self, v):
        L1 = self.L1.__call__(v)
        GL1 = self.GL1.__call__(v)
        return L1 + GL1

    def prox(self, v, step_size):
        """Computes the proximal operator for the Sparse group Lasso
         penalization of a vector v

        Parameters
        ----------
        v : `np.ndarray`
            A coefficient vector

        step_size : `float`
            Value of the step size for the optimization update at the current
            solver iteration

        Returns
        -------
        grad : `np.array`
            The proximal operator of the Sparse group Lasso computed on vector v
        """
        L1_prox = self.L1.prox(v, step_size)
        return self.GL1.prox(L1_prox, step_size)
