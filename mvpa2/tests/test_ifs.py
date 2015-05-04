# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA incremental feature search."""

from mvpa2.testing import *
from mvpa2.testing.clfs import *
from mvpa2.testing.datasets import datasets


from mvpa2.base.dataset import vstack
from mvpa2.datasets.base import Dataset
from mvpa2.featsel.ifs import IFS
from mvpa2.measures.base import CrossValidation, ProxyMeasure
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators.splitters import Splitter
from mvpa2.featsel.helpers import FixedNElementTailSelector
from mvpa2.mappers.fx import mean_sample, BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error



class IFSTests(unittest.TestCase):

    ##REF: Name was automagically refactored
    def get_data(self):
        data = np.random.standard_normal(( 100, 2, 2, 2 ))
        labels = np.concatenate( ( np.repeat( 0, 50 ),
                                  np.repeat( 1, 50 ) ) )
        chunks = np.repeat( range(5), 10 )
        chunks = np.concatenate( (chunks, chunks) )
        return Dataset.from_wizard(samples=data, targets=labels, chunks=chunks)


    # XXX just testing based on a single classifier. Not sure if
    # should test for every known classifier since we are simply
    # testing IFS algorithm - not sensitivities
    @sweepargs(svm=clfswh['has_sensitivity', '!meta'][:1])
    @reseed_rng()
    def test_ifs(self, svm):

        # measure for feature selection criterion and performance assesment
        # use the SAME clf!
        errorfx = mean_mismatch_error
        fmeasure = CrossValidation(svm, NFoldPartitioner(), postproc=mean_sample())
        pmeasure = ProxyMeasure(svm, postproc=BinaryFxNode(errorfx, 'targets'))

        ifs = IFS(fmeasure,
                  pmeasure,
                  Splitter('purpose', attr_values=['train', 'test']),
                  fselector=
                    # go for lower tail selection as data_measure will return
                    # errors -> low is good
                    FixedNElementTailSelector(1, tail='lower', mode='select'),
                  )
        wdata = self.get_data()
        wdata.sa['purpose'] = np.repeat('train', len(wdata))
        tdata = self.get_data()
        tdata.sa['purpose'] = np.repeat('test', len(tdata))
        ds = vstack((wdata, tdata))
        orig_nfeatures = ds.nfeatures

        ifs.train(ds)
        resds = ifs(ds)

        # fail if orig datasets are changed
        self.assertTrue(ds.nfeatures == orig_nfeatures)

        # check that the features set with the least error is selected
        self.assertTrue(len(ifs.ca.errors))
        e = np.array(ifs.ca.errors)
        self.assertTrue(resds.nfeatures == e.argmin() + 1)


        # repeat with dataset where selection order is known
        wsignal = datasets['dumb2'].copy()
        wsignal.sa['purpose'] = np.repeat('train', len(wsignal))
        tsignal = datasets['dumb2'].copy()
        tsignal.sa['purpose'] = np.repeat('test', len(tsignal))
        signal = vstack((wsignal, tsignal))
        ifs.train(signal)
        resds = ifs(signal)
        self.assertTrue((resds.samples[:,0] == signal.samples[:,0]).all())


def suite():  # pragma: no cover
    return unittest.makeSuite(IFSTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

