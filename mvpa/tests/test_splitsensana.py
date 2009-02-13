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
from mvpa.measures.splitmeasure import SplitFeaturewiseMeasure

from tests_warehouse import *
from tests_warehouse_clfs import *


class SplitSensitivityAnalyserTests(unittest.TestCase):

    # XXX meta should work TODO
    @sweepargs(svm=clfswh['linear', 'svm', '!meta'])
    def testAnalyzer(self, svm):
        dataset = datasets['uni2small']

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
        allmaps = N.array(sana.maps)
        self.failUnless(allmaps[:,0].mean() == maps[0])
        self.failUnless(allmaps.shape == (nchunks, nfeatures))


def suite():
    return unittest.makeSuite(SplitSensitivityAnalyserTests)


if __name__ == '__main__':
    import runner

