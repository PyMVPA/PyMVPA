# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

from nose.tools import ok_

from mvpa import pymvpa_dataroot
from mvpa.misc.io import *
from mvpa.misc.fsl import *
from mvpa.misc.bv import BrainVoyagerRTC


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

        # cleanup and ignore stupidity
        try:
            os.remove(fpath)
        except WindowsError:
            pass


    def testSamplesAttributes(self):
        sa = SampleAttributes(os.path.join(pymvpa_dataroot,
                                           'attributes_literal.txt'),
                              literallabels=True)

        ok_(sa.nrows == 1452, msg='There should be 1452 samples')

        # convert to event list, with some custom attr
        ev = sa.toEvents(funky='yeah')
        ok_(len(ev) == 17 * (max(sa.chunks) + 1),
            msg='Not all events got detected.')

        ok_(len([e for e in ev if e.has_key('funky')]) == len(ev),
            msg='All events need to have to custom arg "funky".')

        ok_(ev[0]['label'] == ev[-1]['label'] == 'rest',
            msg='First and last event are rest condition.')

        ok_(ev[-1]['onset'] + ev[-1]['duration'] == sa.nrows,
            msg='Something is wrong with the timiing of the events')


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
        # cleanup and ignore stupidity
        try:
            os.remove(fpath)
        except WindowsError:
            pass

        d = FslEV3(os.path.join(pymvpa_dataroot, 'fslev3.txt'))
        ev = d.toEvents()
        self.failUnless(len(ev) == 3)
        self.failUnless([e['duration'] for e in ev] == [9] * 3)
        self.failUnless([e['onset'] for e in ev] == [6, 21, 35])
        self.failUnless([e['features'] for e in ev] == [[1],[1],[1]])

        ev = d.toEvents(label='face', chunk=0, crap=True)
        ev[0]['label'] = 'house'
        self.failUnless(len(ev) == 3)
        self.failUnless([e['duration'] for e in ev] == [9] * 3)
        self.failUnless([e['onset'] for e in ev] == [6, 21, 35])
        self.failUnless([e['features'] for e in ev] == [[1],[1],[1]])
        self.failUnless([e['label'] for e in ev] == ['house', 'face', 'face'])
        self.failUnless([e['chunk'] for e in ev] == [0]*3)
        self.failUnless([e['crap'] for e in ev] == [True]*3)


    def testFslEV2(self):
        attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'smpl_attr.txt'))

        # check header (sort because order in dict is unpredictable)
        self.failUnless(sorted(attr.keys()) == \
            ['chunks','labels'])

        self.failUnless(attr.nsamples == 3)

    def testBVRTC(self):
        """Simple testing of reading RTC files from BrainVoyager"""

        attr = BrainVoyagerRTC(os.path.join(pymvpa_dataroot, 'bv', 'smpl_model.rtc'))
        self.failUnlessEqual(attr.ncolumns, 4, "We must have 4 colums")
        self.failUnlessEqual(attr.nrows, 147, "We must have 147 rows")

        self.failUnlessEqual(attr._header_order,
                ['l_60 B', 'r_60 B', 'l_80 B', 'r_80 B'],
                "We must got column names correctly")
        self.failUnless(len(attr.r_60_B) == attr.nrows,
                "We must have got access to column by property")
        self.failUnless(attr.toarray() != None,
                "We must have got access to column by property")

    def testdesign2labels(self):
        """Simple testing of helper Design2Labels"""

        attr = BrainVoyagerRTC(os.path.join(pymvpa_dataroot, 'bv', 'smpl_model.rtc'))
        labels0 = design2labels(attr, baseline_label='silence')
        labels = design2labels(attr, baseline_label='silence',
                                func=lambda x:x>0.5)
        Nsilence = lambda x:len(N.where(N.array(x) == 'silence')[0])

        nsilence0 = Nsilence(labels0)
        nsilence = Nsilence(labels)
        self.failUnless(nsilence0 < nsilence,
                        "We must have more silence if thr is higher")
        self.failUnlessEqual(len(labels), attr.nrows,
                        "We must have the same number of labels as rows")
        self.failUnlessRaises(ValueError, design2labels, attr,
                        baseline_label='silence', func=lambda x:x>-1.0)


    def testlabels2chunks(self):
        attr = BrainVoyagerRTC(os.path.join(pymvpa_dataroot, 'bv', 'smpl_model.rtc'))
        labels = design2labels(attr, baseline_label='silence')
        self.failUnlessRaises(ValueError, labels2chunks, labels, 'bugga')
        chunks = labels2chunks(labels)
        self.failUnlessEqual(len(labels), len(chunks))
        # we must got them in sorted order
        chunks_sorted = N.sort(chunks)
        self.failUnless((chunks == chunks_sorted).all())
        # for this specific one we must have just 4 chunks
        self.failUnless((N.unique(chunks) == range(4)).all())


    def testSensorLocations(self):
        sl = XAVRSensorLocations(os.path.join(pymvpa_dataroot, 'xavr1010.dat'))

        for var in ['names', 'pos_x', 'pos_y', 'pos_z']:
            self.failUnless(len(eval('sl.' + var)) == 31)


    def testFslGLMDesign(self):
        glm = FslGLMDesign(os.path.join(pymvpa_dataroot, 'glm.mat'))

        self.failUnless(glm.mat.shape == (850, 6))
        self.failUnless(len(glm.ppheights) == 6)

def suite():
    return unittest.makeSuite(IOHelperTests)


if __name__ == '__main__':
    import runner

