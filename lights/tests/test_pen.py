# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.model.regularizations import Penalties


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

    def test_sparse_group_l1(self):
        """Test Sparse group lasso penalty
        """
        self.setUp()
        l = self.n_long_features
        sparse_group_l1 = self.Penalties.sparse_group_l1(self.v, l)
        sparse_group_l1_ = 27.831
        np.testing.assert_almost_equal(sparse_group_l1, sparse_group_l1_, 3)

    def test_grad_sparse_group_l1(self):
        """Test gradient of Sparse group lasso penalty
        """
        self.setUp()
        l = self.n_long_features
        grad_sparse_group_l1 = self.Penalties.grad_sparse_group_l1(self.v, l)
        grad_sparse_group_l1_ = np.array([.8, 1.4, 1.76, 1.88, 1.177, 1.957,
                                          2 , 1.4, 1.04, 0.92, 1.623, 0.843])
        np.testing.assert_almost_equal(grad_sparse_group_l1,
                                       grad_sparse_group_l1_, 3)



if __name__ == "main":
    unittest.main()
