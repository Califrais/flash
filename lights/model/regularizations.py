# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
from copt.penalty import L1Norm, GroupL1

class Penalties:
    """A class to define the required penalties of the lights model

    Parameters
    ----------

    l_pen : `float`, default=0
        Level of penalization for the ElasticNet and the Sparse Group l1

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 0 this is ridge (L2) regularization
        For eta_elastic_net = 1 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2

    eta_sp_gp_l1: `float`, default=0.1
        The Sparse Group l1 mixing parameter, with 0 <= eta_sp_gp_l1 <= 1
    """
    def __init__(self, l_pen, eta_elastic_net, eta_sp_gp_l1):
        self.l_pen = l_pen
        self.eta_elastic_net = eta_elastic_net
        self.eta_sp_gp_l1 = eta_sp_gp_l1

    @property
    def l_pen(self):
        return self._l_pen

    @l_pen.setter
    def l_pen(self, val):
        if not val >= 0:
            raise ValueError("``l_pen`` must be non negative")
        self._l_pen = val

    @property
    def eta_elastic_net(self):
        return self._eta_elastic_net

    @eta_elastic_net.setter
    def eta_elastic_net(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta_elastic_net`` must be in (0, 1)")
        self._eta_elastic_net = val

    @property
    def eta_sp_gp_l1(self):
        return self._eta_sp_gp_l1

    @eta_sp_gp_l1.setter
    def eta_sp_gp_l1(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``eta_sp_gp_l1`` must be in (0, 1)")
        self._eta_sp_gp_l1 = val

    def elastic_net(self, v):
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
        l_pen, eta, = self.l_pen, self.eta_elastic_net
        return l_pen * ((1. - eta) * abs(v).sum() +
                        .5 * eta * np.linalg.norm(v) ** 2)

    def grad_elastic_net(self, v):
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
        l_pen, eta = self.l_pen, self.eta_elastic_net
        dim = v.shape[0]
        grad = np.zeros(2 * dim)
        # Gradient of lasso penalization
        grad += l_pen * (1 - eta)
        # Gradient of ridge penalization
        grad_pos = l_pen * eta * v
        grad[:dim] += grad_pos
        grad[dim:] -= grad_pos
        return grad

class sparse_group_l1:
    """Sparse group Lasso regularization

    Parameters
    ----------
    alpha: float
        Constant multiplying this loss

    groups: list of lists
    """

    def __init__(self, eta, l_pen, groups):
        self.eta = eta
        # groups need to be increasing
        for i, g in enumerate(groups):
            if not np.all(np.diff(g) == 1):
                raise ValueError("Groups must be contiguous")
            if i > 0 and groups[i - 1][-1] >= g[0]:
                raise ValueError("Groups must be increasing")
        self.groups = groups
        self.L1 = L1Norm(l_pen * (1 - eta))
        self.GL1 = GroupL1(l_pen * eta, self.groups)

    def __call__(self, x):
        L1 = self.L1.__call__(x)
        GL1 = self.GL1.__call__(x)
        return L1 + GL1

    def prox(self, x, step_size):
        L1_prox = self.L1.prox(x, step_size)
        out = self.GL1.prox(L1_prox, step_size)
        return out
