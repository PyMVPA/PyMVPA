# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA IO helpers"""

import re
import os
import unittest
from tempfile import mkstemp
import numpy as np

from mvpa2.testing.tools import ok_, assert_equal, with_tempfile

from mvpa2 import pymvpa_dataroot
from mvpa2.datasets.eventrelated import find_events
from mvpa2.misc.io import *
from mvpa2.misc.fsl import *
from mvpa2.misc.bv import BrainVoyagerRTC


class IOHelperTests(unittest.TestCase):

    def test_column_data_from_file(self):
        ex1 = """eins zwei drei
        0 1 2
        3 4 5
        """
        fd, fpath = mkstemp('mvpa', 'test'); os.close(fd)
        file = open(fpath, 'w')
        file.write(ex1)
        file.close()

        # intentionally rely on defaults
        d = ColumnData(fpath, header=True)

        # check header (sort because order in dict is unpredictable)
        self.assertTrue(sorted(d.keys()) == ['drei', 'eins', 'zwei'])

        self.assertTrue(d['eins'] == [0, 3])
        self.assertTrue(d['zwei'] == [1, 4])
        self.assertTrue(d['drei'] == [2, 5])

        # make a copy
        d2 = ColumnData(d)

        # check if identical
        self.assertTrue(sorted(d2.keys()) == ['drei', 'eins', 'zwei'])
        self.assertTrue(d2['eins'] == [0, 3])
        self.assertTrue(d2['zwei'] == [1, 4])
        self.assertTrue(d2['drei'] == [2, 5])

        # now merge back
        d += d2

        # same columns?
        self.assertTrue(sorted(d.keys()) == ['drei', 'eins', 'zwei'])

        # but more data
        self.assertEqual(d['eins'], [0, 3, 0, 3])
        self.assertEqual(d['zwei'], [1, 4, 1, 4])
        self.assertEqual(d['drei'], [2, 5, 2, 5])

        # test file write
        # TODO: check if correct
        header_order = ['drei', 'zwei', 'eins']
        d.tofile(fpath, header_order=header_order)

        # test sample selection
        dsel = d.select_samples([0, 2])
        self.assertEqual(dsel['eins'], [0, 0])
        self.assertEqual(dsel['zwei'], [1, 1])
        self.assertEqual(dsel['drei'], [2, 2])

        # test if order is read from file when available
        d3 = ColumnData(fpath)
        self.assertEqual(d3._header_order, header_order)

        # add another column -- should be appended as the last column
        # while storing
        d3['four'] = [0.1] * len(d3['eins'])
        d3.tofile(fpath)

        d4 = ColumnData(fpath)
        self.assertEqual(d4._header_order, header_order + ['four'])

        # cleanup and ignore stupidity
        try:
            os.remove(fpath)
        except WindowsError:
            pass


    def test_samples_attributes(self):
        sa = SampleAttributes(os.path.join(pymvpa_dataroot,
                                           'attributes_literal.txt'),
                              literallabels=True)

        ok_(sa.nrows == 1452, msg='There should be 1452 samples')

        # convert to event list, with some custom attr
        ev = find_events(**sa)
        ok_(len(ev) == 17 * (max(sa.chunks) + 1),
            msg='Not all events got detected.')

        ok_(ev[0]['targets'] == ev[-1]['targets'] == 'rest',
            msg='First and last event are rest condition.')

        ok_(ev[-1]['onset'] + ev[-1]['duration'] == sa.nrows,
            msg='Something is wrong with the timiing of the events')


    @with_tempfile('mvpa', 'sampleattr')
    def test_samples_attributes_autodtype(self, fn):
        payload = '''a b c
1 1.1 a
2 2.2 b
3 3.3 c
4 4.4 d'''

        with open(fn, 'w') as f:
            f.write(payload)

        attr = SampleAttributes(fn, header=True)

        assert_equal(set(attr.keys()), set(['a', 'b', 'c']))
        assert_equal(attr['a'], [1, 2, 3, 4])
        assert_equal(attr['b'], [1.1, 2.2, 3.3, 4.4])
        assert_equal(attr['c'], ['a', 'b', 'c', 'd'])


    def test_fsl_ev(self):
        ex1 = """0.0 2.0 1
        13.89 2 1
        16 2.0 0.5
        """
        fd, fpath = mkstemp('mvpa', 'test'); os.close(fd)
        file = open(fpath, 'w')
        file.write(ex1)
        file.close()

        # intentionally rely on defaults
        d = FslEV3(fpath)

        # check header (sort because order in dict is unpredictable)
        self.assertTrue(sorted(d.keys()) == \
            ['durations', 'intensities', 'onsets'])

        self.assertTrue(d['onsets'] == [0.0, 13.89, 16.0])
        self.assertTrue(d['durations'] == [2.0, 2.0, 2.0])
        self.assertTrue(d['intensities'] == [1.0, 1.0, 0.5])

        self.assertTrue(d.nevs == 3)
        self.assertTrue(d.get_ev(1) == (13.89, 2.0, 1.0))
        # cleanup and ignore stupidity
        try:
            os.remove(fpath)
        except WindowsError:
            pass

        d = FslEV3(os.path.join(pymvpa_dataroot, 'fslev3.txt'))
        ev = d.to_events()
        self.assertTrue(len(ev) == 3)
        self.assertTrue([e['duration'] for e in ev] == [9] * 3)
        self.assertTrue([e['onset'] for e in ev] == [6, 21, 35])
        self.assertTrue([e['features'] for e in ev] == [[1], [1], [1]])

        ev = d.to_events(label='face', chunk=0, crap=True)
        ev[0]['label'] = 'house'
        self.assertTrue(len(ev) == 3)
        self.assertTrue([e['duration'] for e in ev] == [9] * 3)
        self.assertTrue([e['onset'] for e in ev] == [6, 21, 35])
        self.assertTrue([e['features'] for e in ev] == [[1], [1], [1]])
        self.assertTrue([e['label'] for e in ev] == ['house', 'face', 'face'])
        self.assertTrue([e['chunk'] for e in ev] == [0] * 3)
        self.assertTrue([e['crap'] for e in ev] == [True] * 3)


    def test_fsl_ev2(self):
        attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'smpl_attr.txt'))

        # check header (sort because order in dict is unpredictable)
        self.assertTrue(sorted(attr.keys()) == \
            ['chunks', 'targets'])

        self.assertTrue(attr.nsamples == 3)

    def test_bv_rtc(self):
        """Simple testing of reading RTC files from BrainVoyager"""

        attr = BrainVoyagerRTC(os.path.join(pymvpa_dataroot, 'bv', 'smpl_model.rtc'))
        self.assertEqual(attr.ncolumns, 4, "We must have 4 colums")
        self.assertEqual(attr.nrows, 147, "We must have 147 rows")

        self.assertEqual(attr._header_order,
                ['l_60 B', 'r_60 B', 'l_80 B', 'r_80 B'],
                "We must got column names correctly")
        self.assertTrue(len(attr.r_60_B) == attr.nrows,
                "We must have got access to column by property")
        self.assertTrue(attr.toarray() != None,
                "We must have got access to column by property")

    def testdesign2labels(self):
        """Simple testing of helper Design2Labels"""

        attr = BrainVoyagerRTC(os.path.join(pymvpa_dataroot, 'bv', 'smpl_model.rtc'))
        labels0 = design2labels(attr, baseline_label='silence')
        labels = design2labels(attr, baseline_label='silence',
                                func=lambda x:x > 0.5)
        Nsilence = lambda x:len(np.where(np.array(x) == 'silence')[0])

        nsilence0 = Nsilence(labels0)
        nsilence = Nsilence(labels)
        self.assertTrue(nsilence0 < nsilence,
                        "We must have more silence if thr is higher")
        self.assertEqual(len(labels), attr.nrows,
                        "We must have the same number of labels as rows")
        self.assertRaises(ValueError, design2labels, attr,
                        baseline_label='silence', func=lambda x:x > -1.0)


    def testlabels2chunks(self):
        attr = BrainVoyagerRTC(os.path.join(pymvpa_dataroot, 'bv', 'smpl_model.rtc'))
        labels = design2labels(attr, baseline_label='silence')
        self.assertRaises(ValueError, labels2chunks, labels, 'bugga')
        chunks = labels2chunks(labels)
        self.assertEqual(len(labels), len(chunks))
        # we must got them in sorted order
        chunks_sorted = np.sort(chunks)
        self.assertTrue((chunks == chunks_sorted).all())
        # for this specific one we must have just 4 chunks
        self.assertTrue((np.unique(chunks) == range(4)).all())


    def test_sensor_locations(self):
        sl = XAVRSensorLocations(os.path.join(pymvpa_dataroot, 'xavr1010.dat'))

        for var in ['names', 'pos_x', 'pos_y', 'pos_z']:
            self.assertTrue(len(eval('sl.' + var)) == 31)


    def test_fsl_glm_design(self):
        glm = FslGLMDesign(os.path.join(pymvpa_dataroot, 'glm.mat'))

        self.assertTrue(glm.mat.shape == (850, 6))
        self.assertTrue(len(glm.ppheights) == 6)

    def test_read_fsl_design(self):
        fname = os.path.join(pymvpa_dataroot,
                             'sample_design.fsf')
        # use our function
        design = read_fsl_design(fname)
        # and just load manually to see either we match fine
        set_lines = [x for x in open(fname).readlines()
                     if x.startswith('set ')]
        assert_equal(len(set_lines), len(design))

        # figure out which one is missing
        """TODO: would require the same special treatment for _files fields
        re_set = re.compile("set ([^)]*\)).*")
        for line in set_lines:
            key = re_set.search(line).groups()[0]
            if not key in design:
                raise AssertionError(
                    "Key %s was not found in read FSL design" % key)
        key_list = [' '.join(l.split(None,2)[1:2]) for l in set_lines]
        for k in set(key_list):
            if len([key for key in key_list if key == k]) == 2:
                raise AssertionError(
                    "Got the non-unique beast %s" % k)
                    """

def suite():  # pragma: no cover
    return unittest.makeSuite(IOHelperTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

