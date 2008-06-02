#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA IO helpers"""

import os
import unittest
from tempfile import mkstemp
import numpy as N

from mvpa.misc.iohelpers import ColumnData, SampleAttributes
from mvpa.misc.fsl.base import FslEV3
from mvpa.misc.bv.base import BrainVoyagerRTC


class IOHelperTests(unittest.TestCase):

    def testColumnDataFromFile(self):
        ex1 = """eins zwei drei
        0 1 2
        3 4 5
        """
        file, fpath = mkstemp('mvpa', 'test')
        file = open(fpath, 'w')
        file.write(ex1)
        file.close()

        # intentionally rely on defaults
        d = ColumnData(fpath, header=True)

        # check header (sort because order in dict is unpredictable)
        self.failUnless(sorted(d.keys()) == ['drei','eins','zwei'])

        self.failUnless(d['eins'] == [0, 3])
        self.failUnless(d['zwei'] == [1, 4])
        self.failUnless(d['drei'] == [2, 5])

        # make a copy
        d2 = ColumnData(d)

        # check if identical
        self.failUnless(sorted(d2.keys()) == ['drei','eins','zwei'])
        self.failUnless(d2['eins'] == [0, 3])
        self.failUnless(d2['zwei'] == [1, 4])
        self.failUnless(d2['drei'] == [2, 5])

        # now merge back
        d += d2

        # same columns?
        self.failUnless(sorted(d.keys()) == ['drei','eins','zwei'])

        # but more data
        self.failUnlessEqual(d['eins'], [0, 3, 0, 3])
        self.failUnlessEqual(d['zwei'], [1, 4, 1, 4])
        self.failUnlessEqual(d['drei'], [2, 5, 2, 5])

        # test file write
        # TODO: check if correct
        header_order = ['drei', 'zwei', 'eins']
        d.tofile(fpath, header_order=header_order)

        # test sample selection
        dsel = d.selectSamples([0, 2])
        self.failUnlessEqual(dsel['eins'], [0, 0])
        self.failUnlessEqual(dsel['zwei'], [1, 1])
        self.failUnlessEqual(dsel['drei'], [2, 2])

        # test if order is read from file when available
        d3 = ColumnData(fpath)
        self.failUnlessEqual(d3._header_order, header_order)

        # add another column -- should be appended as the last column
        # while storing
        d3['four'] = [0.1] * len(d3['eins'])
        d3.tofile(fpath)

        d4 = ColumnData(fpath)
        self.failUnlessEqual(d4._header_order, header_order + ['four'])

        # cleanup
        os.remove(fpath)


    def testFslEV(self):
        ex1 = """0.0 2.0 1
        13.89 2 1
        16 2.0 0.5
        """
        file, fpath = mkstemp('mvpa', 'test')
        file = open(fpath, 'w')
        file.write(ex1)
        file.close()

        # intentionally rely on defaults
        d = FslEV3(fpath)

        # check header (sort because order in dict is unpredictable)
        self.failUnless(sorted(d.keys()) == \
            ['durations','intensities','onsets'])

        self.failUnless(d['onsets'] == [0.0, 13.89, 16.0])
        self.failUnless(d['durations'] == [2.0, 2.0, 2.0])
        self.failUnless(d['intensities'] == [1.0, 1.0, 0.5])

        self.failUnless(d.getNEVs() == 3)
        self.failUnless(d.getEV(1) == (13.89, 2.0, 1.0))
        # cleanup
        os.remove(fpath)


    def testFslEV2(self):
        attr = SampleAttributes(os.path.join('..', 'data', 'smpl_attr.txt'))

        # check header (sort because order in dict is unpredictable)
        self.failUnless(sorted(attr.keys()) == \
            ['chunks','labels'])

        self.failUnless(attr.nsamples == 3)

    def testBVRTC(self):
        """Simple testing of reading RTC files from BrainVoyager"""

        attr = BrainVoyagerRTC(os.path.join('..', 'data', 'bv/smpl_model.rtc'))
        self.failUnlessEqual(attr.ncolumns, 4, "We must have 4 colums")
        self.failUnlessEqual(attr.nrows, 147, "We must have 147 rows")

        self.failUnlessEqual(attr._header_order,
                ['l_60 B', 'r_60 B', 'l_80 B', 'r_80 B'],
                "We must got column names correctly")
        self.failUnless(len(attr.r_60_B) == attr.nrows,
                "We must have got access to column by property")
        self.failUnless(attr.toarray() != None,
                "We must have got access to column by property")

def suite():
    return unittest.makeSuite(IOHelperTests)


if __name__ == '__main__':
    import runner

