# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for some Fx and Compound learners"""

import numpy as np

from mvpa2.testing import *
from mvpa2.base.learner import Learner, CompoundLearner, \
                ChainLearner, CombinedLearner
from mvpa2.base.node import Node, CompoundNode, \
                ChainNode, CombinedNode

from mvpa2.datasets.base import AttrDataset

class FxNode(Node):
    def __init__(self, f, space='targets',
                 pass_attr=None, postproc=None, **kwargs):
        super(FxNode, self).__init__(space, pass_attr, postproc, **kwargs)
        self.f = f

    def _call(self, ds):
        cp = ds.copy()
        cp.samples = self.f(ds.samples)
        return cp

class FxyLearner(Learner):
    def __init__(self, f):
        super(FxyLearner, self).__init__()
        self.f = f
        self.x = None

    def _train(self, ds):
        self.x = ds.samples

    def _call(self, ds):
        cp = ds.copy()
        cp.samples = self.f(self.x)(ds.samples)
        return cp


class CompoundTests(unittest.TestCase):
    def test_compound_node(self):
        data = np.asarray([[1, 2, 3, 4]], dtype=np.float_).T
        ds = AttrDataset(data, sa=dict(targets=[0, 0, 1, 1]))

        add = lambda x: lambda y: x + y
        mul = lambda x: lambda y: x * y

        add2 = FxNode(add(2))
        mul3 = FxNode(mul(3))

        assert_array_equal(add2(ds).samples, data + 2)

        add2mul3 = ChainNode([add2, mul3])
        assert_array_equal(add2mul3(ds), (data + 2) * 3)

        add2_mul3v = CombinedNode([add2, mul3], 'v')
        add2_mul3h = CombinedNode([add2, mul3], 'h')
        assert_array_equal(add2_mul3v(ds).samples,
                           np.vstack((data + 2, data * 3)))
        assert_array_equal(add2_mul3h(ds).samples,
                           np.hstack((data + 2, data * 3)))

    def test_compound_learner(self):
        data = np.asarray([[1, 2, 3, 4]], dtype=np.float_).T
        ds = AttrDataset(data, sa=dict(targets=[0, 0, 1, 1]))
        train = ds[ds.sa.targets == 0]
        test = ds[ds.sa.targets == 1]
        dtrain = train.samples
        dtest = test.samples

        sub = FxyLearner(lambda x: lambda y: x - y)
        assert_false(sub.is_trained)
        sub.train(train)
        assert_array_equal(sub(test).samples, dtrain - dtest)


        div = FxyLearner(lambda x: lambda y: x / y)
        div.train(train)
        assert_array_almost_equal(div(test).samples, dtrain / dtest)
        div.untrain()

        subdiv = ChainLearner((sub, div))
        assert_false(subdiv.is_trained)
        subdiv.train(train)
        assert_true(subdiv.is_trained)
        subdiv.untrain()
        assert_raises(RuntimeError, subdiv, test)
        subdiv.train(train)

        assert_array_almost_equal(subdiv(test).samples, dtrain / (dtrain - dtest))

        sub_div = CombinedLearner((sub, div), 'v')
        assert_true(sub_div.is_trained)
        sub_div.untrain()
        subdiv.train(train)
        assert_true(sub_div.is_trained)

        assert_array_almost_equal(sub_div(test).samples,
                                  np.vstack((dtrain - dtest, dtrain / dtest)))



def suite():  # pragma: no cover
    return unittest.makeSuite(CompoundTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

