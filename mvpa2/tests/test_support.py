# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA serial feature inclusion algorithm"""

from mvpa2.testing import *
from mvpa2.misc.support import *
from mvpa2.base.types import asobjarray
from mvpa2.testing import *
from mvpa2.testing.datasets import get_mv_pattern, datasets
from mvpa2.testing.clfs import *
from mvpa2.clfs.distance import one_minus_correlation

from mvpa2.support.copy import deepcopy

class SupportFxTests(unittest.TestCase):

    def test_transform_with_boxcar(self):
        data = np.arange(10)
        sp = np.arange(10)

        # check if stupid thing don't work
        self.assertRaises(ValueError,
                              transform_with_boxcar,
                              data,
                              sp,
                              0)

        # now do an identity transformation
        trans = transform_with_boxcar(data, sp, 1)
        self.assertTrue((trans == data).all())

        # now check for illegal boxes
        self.assertRaises(ValueError,
                              transform_with_boxcar,
                              data,
                              sp,
                              2)

        # now something that should work
        sp = np.arange(9)
        trans = transform_with_boxcar(data, sp, 2)
        self.assertTrue((trans == \
                           [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]).all())


        # now test for proper data shape
        data = np.ones((10, 3, 4, 2))
        sp = [ 2, 4, 3, 5 ]
        trans = transform_with_boxcar(data, sp, 4)
        self.assertTrue(trans.shape == (4, 3, 4, 2))



    def test_event(self):
        self.assertRaises(ValueError, Event)
        ev = Event(onset=2.5)

        # all there?
        self.assertTrue(ev.items() == [('onset', 2.5)])

        # conversion
        self.assertTrue(ev.as_descrete_time(dt=2).items() == [('onset', 1)])
        evc = ev.as_descrete_time(dt=2, storeoffset=True)
        self.assertTrue(evc.has_key('offset'))
        self.assertTrue(evc['offset'] == 0.5)

        # same with duration included
        evc = Event(onset=2.5, duration=3.55).as_descrete_time(dt=2)
        self.assertTrue(evc['duration'] == 3)


    def test_mof_n_combinations(self):
        self.assertEqual(
            unique_combinations(range(3), 1), [[0], [1], [2]])
        self.assertEqual(
            unique_combinations(
                        range(4), 2),
                        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
                        )
        self.assertEqual(
            unique_combinations(
                        range(4), 3),
                        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


    @reseed_rng()
    def test_xrandom_unique_combinations(self):
        for n in [4, 5, 10]:
            limit = 4
            limited = list(xrandom_unique_combinations(range(n), 3, limit))
            self.assertEqual(len(limited), limit)
            # See if we would obtain the same
            for k in [1, 2, 3, int(n / 2), n]:
                all_random = list(xrandom_unique_combinations(range(n), k))
                all_ = list(xunique_combinations(range(n), k))
                self.assertEqual(sorted(all_random), sorted(all_))

        # test that we are not sampling the same space -- two
        # consecutive samples within large number very unlikely not
        # have more than few overlapping samples
        iter_count = 100
        overlapping_count = 0
        for k in xrange(iter_count):
            c1, c2 = xrandom_unique_combinations(range(1000), 10, 2)
            if len(set(c1).intersection(c2)) == 2:
                overlapping_count += 1

        # assume this happens less than 10 percent of the time
        self.assertTrue(overlapping_count * 10 < iter_count)


    def test_break_points(self):
        items_cont = [0, 0, 0, 1, 1, 1, 3, 3, 2]
        items_noncont = [0, 0, 1, 1, 0, 3, 2]
        self.assertRaises(ValueError, get_break_points, items_noncont)
        self.assertEqual(get_break_points(items_noncont, contiguous=False),
                             [0, 2, 4, 5, 6])
        self.assertEqual(get_break_points(items_cont), [0, 3, 6, 8])
        self.assertEqual(get_break_points(items_cont, contiguous=False),
                             [0, 3, 6, 8])


    def test_map_overlap(self):
        mo = MapOverlap()

        maps = [[1, 0, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0]]

        overlap = mo(maps)

        self.assertEqual(overlap, 1. / len(maps[0]))
        self.assertTrue((mo.overlap_map == [1, 0, 0, 0]).all())
        self.assertTrue((mo.spread_map == [0, 0, 1, 1]).all())
        self.assertTrue((mo.ovstats_map == [1, 0, 2. / 3, 1. / 3]).all())

        mo = MapOverlap(overlap_threshold=0.5)
        overlap = mo(maps)
        self.assertEqual(overlap, 2. / len(maps[0]))
        self.assertTrue((mo.overlap_map == [1, 0, 1, 0]).all())
        self.assertTrue((mo.spread_map == [0, 0, 0, 1]).all())
        self.assertTrue((mo.ovstats_map == [1, 0, 2. / 3, 1. / 3]).all())


    @reseed_rng()
    @sweepargs(pair=[(np.random.normal(size=(10, 20)), np.random.normal(size=(10, 20))),
                     ([1, 2, 3, 0], [1, 3, 2, 0]),
                     ((1, 2, 3, 1), (1, 3, 2, 1))])
    def test_id_hash(self, pair):
        a, b = pair
        a1 = deepcopy(a)
        a_1 = idhash(a)
        self.assertTrue(a_1 == idhash(a), msg="Must be of the same idhash")
        self.assertTrue(a_1 != idhash(b), msg="Must be of different idhash")
        if isinstance(a, np.ndarray):
            self.assertTrue(a_1 != idhash(a.T), msg=".T must be of different idhash")
        if not isinstance(a, tuple):
            self.assertTrue(a_1 != idhash(a1), msg="Must be of different idhash")
            a[2] += 1; a_2 = idhash(a)
            self.assertTrue(a_1 != a_2, msg="Idhash must change")
        else:
            a_2 = a_1
        a = a[2:]; a_3 = idhash(a)
        self.assertTrue(a_2 != a_3, msg="Idhash must change after slicing")


    def test_asobjarray(self):
        for i in ([1, 2, 3], ['a', 2, '3'],
                  ('asd')):
            i_con = asobjarray(i)
            self.assertTrue(i_con.dtype is np.dtype('object'))
            self.assertEqual(len(i), len(i_con))

            # Note: in Python3 the ['a' , 2, '3'] list is converted to
            # an array with elements 'a', '2',' and '3' (i.e. string representation
            # for the second element), and thus np.all(i==i_con) fails.
            # Instead here each element is tested for equality seperately
            # XXX is this an issue?
            self.assertTrue(np.all((i[j] == i_con[j]) for j in xrange(len(i))))

    @reseed_rng()
    def test_correlation(self):
        # data: 20 samples, 80 features
        X = np.random.rand(20, 80)

        C = 1 - one_minus_correlation(X, X)

        # get nsample x nssample correlation matrix
        self.assertTrue(C.shape == (20, 20))
        # diagonal is 1
        self.assertTrue((np.abs(np.diag(C) - 1).mean() < 0.00001).all())

        # now two different
        Y = np.random.rand(5, 80)
        C2 = 1 - one_minus_correlation(X, Y)
        # get nsample x nssample correlation matrix
        self.assertTrue(C2.shape == (20, 5))
        # external validity check -- we are dealing with correlations
        self.assertTrue(C2[10, 2] - np.corrcoef(X[10], Y[2])[0, 1] < 0.000001)

    def test_version_to_tuple(self):
        """Test conversion of versions from strings
        """

        self.assertTrue(version_to_tuple('0.0.01') == (0, 0, 1))
        self.assertTrue(version_to_tuple('0.7.1rc3') == (0, 7, 1, 'rc', 3))


    def test_smart_version(self):
        """Test our ad-hoc SmartVersion
        """
        SV = SmartVersion

        for v1, v2 in (
            ('0.0.1', '0.0.2'),
            ('0.0.1', '0.1'),
            ('0.0.1', '0.1.0'),
            ('0.0.1', '0.0.1a'), # this might be a bit unconventional?
            ('0.0.1', '0.0.1+svn234'),
            ('0.0.1+svn234', '0.0.1+svn235'),
            ('0.0.1dev1', '0.0.1'),
            ('0.0.1dev1', '0.0.1rc3'),
            ('0.7.1rc3', '0.7.1'),
            ('0.0.1-dev1', '0.0.1'),
            ('0.0.1-svn1', '0.0.1'),
            ('0.0.1~p', '0.0.1'),
            ('0.0.1~prior.1.2', '0.0.1'),
            ):
            self.assertTrue(SV(v1) < SV(v2),
                            msg="Failed to compare %s to %s" % (v1, v2))
            self.assertTrue(SV(v2) > SV(v1),
                            msg="Failed to reverse compare %s to %s" % (v2, v1))
            # comparison to strings
            self.assertTrue(SV(v1) < v2,
                            msg="Failed to compare %s to string %s" % (v1, v2))
            self.assertTrue(v1 < SV(v2),
                            msg="Failed to compare string %s to %s" % (v1, v2))
            # to tuples
            self.assertTrue(SV(v1) < version_to_tuple(v2),
                            msg="Failed to compare %s to tuple of %s"
                            % (v1, v2))
            self.assertTrue(version_to_tuple(v1) < SV(v2),
                            msg="Failed to compare tuple of %s to %s"
                            % (v1, v2))


def test_value2idx():
    times = [1.2, 1.3, 2., 4., 0., 2., 1.1]
    assert_equal(value2idx(0, times), 4)
    assert_equal(value2idx(100, times), 3)
    assert_equal(value2idx(1.5, times), 1)
    assert_equal(value2idx(1.5, times, 'ceil'), 2)
    assert_equal(value2idx(1.2, times, 'floor'), 0)
    assert_equal(value2idx(1.14, times, 'round'), 6)
    assert_equal(value2idx(1.14, times, 'floor'), 6)
    assert_equal(value2idx(1.14, times, 'ceil'), 0)
    assert_equal(value2idx(-100, times, 'ceil'), 4)


def test_limit_filter():
    ds = datasets['uni2small']
    assert_array_equal(get_limit_filter(None, ds.sa),
                       np.ones(len(ds), dtype=np.bool))
    assert_array_equal(get_limit_filter('chunks', ds.sa),
                       ds.sa.chunks)
    assert_array_equal(get_limit_filter({'chunks': 3}, ds.sa),
                       ds.sa.chunks == 3)
    assert_array_equal(get_limit_filter({'chunks': [3, 1]}, ds.sa),
                       np.logical_or(ds.sa.chunks == 3,
                                     ds.sa.chunks == 1))

def test_mask2slice():
    slc = np.repeat(False, 5)
    assert_equal(mask2slice(slc), slice(None, 0, None))


def suite():  # pragma: no cover
    return unittest.makeSuite(SupportFxTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

