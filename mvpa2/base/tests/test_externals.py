# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test externals checking"""

import unittest

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.support import copy
from mvpa2.testing import SkipTest

class TestExternals(unittest.TestCase):

    def setUp(self):
        self.backup = []
        # paranoid check
        self.cfgstr = str(cfg)
        # clean up externals cfg for proper testing
        if cfg.has_section('externals'):
            self.backup = copy.deepcopy(cfg.items('externals'))
        cfg.remove_section('externals')


    def tearDown(self):
        if len(self.backup):
            # wipe existing one completely
            if cfg.has_section('externals'):
                cfg.remove_section('externals')
            cfg.add_section('externals')
            for o,v in self.backup:
                cfg.set('externals', o,v)
        # paranoid check
        # since order can't be guaranteed, lets check
        # each item after sorting
        self.assertEqual(sorted(self.cfgstr.split('\n')),
                             sorted(str(cfg).split('\n')))


    def test_externals(self):
        self.assertRaises(ValueError, externals.exists, 'BoGuS')


    def test_externals_no_double_invocation(self):
        # no external should be checking twice (unless specified
        # explicitely)

        class Checker(object):
            """Helper class to increment count of actual checks"""
            def __init__(self): self.checked = 0
            def check(self): self.checked += 1

        checker = Checker()

        externals._KNOWN['checker'] = 'checker.check()'
        externals.__dict__['checker'] = checker
        externals.exists('checker')
        self.assertEqual(checker.checked, 1)
        externals.exists('checker')
        self.assertEqual(checker.checked, 1)
        externals.exists('checker', force=True)
        self.assertEqual(checker.checked, 2)
        externals.exists('checker')
        self.assertEqual(checker.checked, 2)

        # restore original externals
        externals.__dict__.pop('checker')
        externals._KNOWN.pop('checker')


    def test_externals_correct2nd_invocation(self):
        # always fails
        externals._KNOWN['checker2'] = 'raise ImportError'

        self.assertTrue(not externals.exists('checker2'),
                        msg="Should be False on 1st invocation")

        self.assertTrue(not externals.exists('checker2'),
                        msg="Should be False on 2nd invocation as well")

        externals._KNOWN.pop('checker2')

    def test_absent_external_version(self):
        # should not blow, just return None
        if externals.exists('shogun'):
            raise SkipTest("shogun is present, can't test")
        self.assertEqual(externals.versions['shogun'], None)


def suite():  # pragma: no cover
    return unittest.makeSuite(TestExternals)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

