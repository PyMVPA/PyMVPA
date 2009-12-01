# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.measures.splitmeasure import SplitFeaturewiseMeasure, \
        TScoredFeaturewiseMeasure
from mvpa.misc.data_generators import normalFeatureDataset
from mvpa.misc.transformers import Absolute
from mvpa.misc.attrmap import AttributeMap
from tests_warehouse import *
from tests_warehouse_clfs import *


class SplitSensitivityAnalyserTests(unittest.TestCase):

    # XXX meta should work TODO
    @sweepargs(svm=clfswh['linear', 'svm', '!meta'])
    def testAnalyzer(self, svm):
        dataset = datasets['uni2small']
        # XXX for now convert to numeric labels, but should better be taken
        # care of during clf refactoring
        dataset.labels = AttributeMap().to_numeric(dataset.labels)

        svm_weigths = svm.getSensitivityAnalyzer()

        sana = SplitFeaturewiseMeasure(
                    svm_weigths,
                    NFoldSplitter(cvtype=1),
                    enable_states=['maps'])

        maps = sana(dataset)
        nchunks = len(dataset.uniquechunks)
        nfeatures = dataset.nfeatures
        self.failUnless(len(maps) == nfeatures,
            msg='Lengths of the map %d is different from number of features %d'
                 % (len(maps), nfeatures))
        self.failUnless(sana.states.isKnown('maps'))
        allmaps = N.array(sana.states.maps)
        self.failUnless(allmaps[:,0].mean() == maps[0])
        self.failUnless(allmaps.shape == (nchunks, nfeatures))


    @sweepargs(svm=clfswh['linear', 'svm', '!meta'])
    def testTScoredAnalyzer(self, svm):
        self.dataset = normalFeatureDataset(perlabel=100,
                                            nlabels=2,
                                            nchunks=20,
                                            nonbogus_features=[0,1],
                                            nfeatures=4,
                                            snr=10)
        # XXX for now convert to numeric labels, but should better be taken
        # care of during clf refactoring
        self.dataset.labels = AttributeMap().to_numeric(self.dataset.labels)
        svm_weigths = svm.getSensitivityAnalyzer()

        sana = TScoredFeaturewiseMeasure(
                    svm_weigths,
                    NFoldSplitter(cvtype=1),
                    enable_states=['maps'])

        t = sana(self.dataset)

        # correct size?
        self.failUnlessEqual(t.shape, (4,))

        # check reasonable sensitivities
        t = Absolute(t)
        self.failUnless(N.mean(t[:2]) > N.mean(t[2:]))

        # check whether SplitSensitivityAnalyzer 'maps' state is accessible
        self.failUnless(sana.states.isKnown('maps'))
        self.failUnless(N.array(sana.states.maps).shape == (20,4))



def suite():
    return unittest.makeSuite(SplitSensitivityAnalyserTests)


if __name__ == '__main__':
    import runner

