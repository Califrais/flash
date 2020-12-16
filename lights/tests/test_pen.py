# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.model.regularizations import Penalties, sparse_group_l1


class Test(unittest.TestCase):

    def setUp(self):
        l_pen = 2.
        eta_elastic_net = .2
        eta_sp_gp_l1 = .3
        self.Penalties = Penalties(l_pen, eta_elastic_net, eta_sp_gp_l1)
        self.v = np.array([-1, 0, 3, 4, -2, 5])
        self.n_long_features = 3

    def test_elastic_net(self):
        """Test Elastic net penalty
        """
        self.setUp()
        elastic_net = self.Penalties.elastic_net(self.v)
        elastic_net_ = 35
        np.testing.assert_almost_equal(elastic_net, elastic_net_)

    def test_grad_elastic_net(self):
        """Test gradient of Elastic net penalty
        """
        self.setUp()
        grad_elastic_net = self.Penalties.grad_elastic_net(self.v)
        grad_elastic_net_ = np.array([1.2, 1.6, 2.8, 3.2, .8, 3.6,
                                      2, 1.6, .4, 0, 2.4, -.4])
        np.testing.assert_almost_equal(grad_elastic_net, grad_elastic_net_)

    def test_sgl1(self):
        """Test Elastic net penalty
        """
        x = np.array([1, -2, -3, 4, 5])
        groups = [[0, 1], [2, 3, 4]]
        alpha = 0.7
        step_size = 0.3
        sgl1 = sparse_group_l1(alpha, groups)
        v = sgl1.__call__(x)
        v_ = 11.015
        prox = sgl1.prox(x, step_size)
        prox_ = np.array([0.82, -1.72, -2.821, 3.791, 4.761])
        np.testing.assert_almost_equal(v, v_, 3)
        np.testing.assert_almost_equal(prox, prox_, 3)

if __name__ == "main":
    unittest.main()
