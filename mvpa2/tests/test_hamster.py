# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Hamster"""

import os
import unittest
import numpy as np

from mvpa2.testing.tools import *
from mvpa2.misc.io.hamster import *

class HamsterHelperTests(unittest.TestCase):

    def test_specification(self):

        # more than 1 positional
        self.assertRaises(ValueError, Hamster, "1", 2)
        # do not mix positional
        self.assertRaises(ValueError, Hamster, "1", bu=123)
        # need to be a string
        self.assertRaises(ValueError, Hamster, 1)
        # dump cannot be assigned
        self.assertRaises(ValueError, Hamster, dump=123)
        # need to be an existing file
        self.assertRaises(IOError, Hamster, "/dev/ZUMBARGAN123")

        hh=Hamster(budda=1, z=[123], fuga="123"); hh.h1=123;
        delattr(hh, 'budda')
        self.assertTrue(`hh` == "Hamster(fuga='123', h1=123, z=[123])")


    @with_tempfile()
    @reseed_rng()
    def test_simple_storage(self, filename):
        ex1 = """eins zwei drei
        0 1 2
        3 4 5
        """
        skip_if_no_external('cPickle')
        skip_if_no_external('gzip')

        ex2 = {'d1': np.random.normal(size=(4,4))}

        hamster = Hamster(ex1=ex1)
        hamster.d = ex2
        hamster.boo = HamsterHelperTests

        total_dict = {'ex1' : ex1,
                      'd'   : ex2,
                      'boo' : HamsterHelperTests}
        self.assertTrue(hamster.asdict() == total_dict)
        self.assertTrue(set(hamster.registered) == set(['ex1', 'd', 'boo']))

        filename_gz = filename + '.gz'
        filename_bogusgz = filename + '_bogus.gz'

        # dump
        hamster.dump(filename)
        hamster.dump(filename_gz)
        # allow to shoot yourself in the head
        hamster.dump(filename_bogusgz, compresslevel=0)
        self.assertTrue(hamster.asdict() == total_dict)

        # We should have stored plain and gzipped versions
        gzplain = gzip.open(filename)
        self.assertRaises(IOError, gzplain.readlines)
        gzipped = gzip.open(filename_gz)
        discard = gzipped.readlines()
        gzbogus = gzip.open(filename_bogusgz)
        self.assertRaises(IOError, gzbogus.readlines)

        # load plain
        hamster2 = Hamster(filename)

        # check if we re-stored all the keys
        k =  hamster.__dict__.keys();
        k2 = hamster2.__dict__.keys();
        self.assertTrue(set(k) == set(k2))

        # identity should be lost
        self.assertTrue(hamster.ex1 is hamster.ex1)
        self.assertTrue(not (hamster.ex1 is hamster2.ex1))

        # lets compare
        self.assertTrue(hamster.ex1 == hamster2.ex1)

        self.assertTrue(hamster.d.keys() == hamster2.d.keys())
        self.assertTrue((hamster.d['d1'] == hamster2.d['d1']).all())


        self.assertTrue(hamster.boo == hamster2.boo)
        # not sure if that is a feature or a bug
        self.assertTrue(hamster.boo is hamster2.boo)

        # cleanup
        gzipped.close()
        os.remove(filename_gz)
        gzbogus.close()
        os.remove(filename_bogusgz)

    @reseed_rng()
    def test_assignment(self):
        ex1 = """eins zwei drei
        0 1 2
        3 4 5
        """
        ex2 = {'d1': np.random.normal(size=(4,4))}

        h = Hamster(ex1=ex1)
        h.ex2 = ex2
        self.assertTrue(hasattr(h, 'ex2'))
        h.ex2 = None
        self.assertTrue(h.ex2 is None)
        h.ex2 = 123
        self.assertTrue(h.ex2 == 123)
        h.has_key  = 123



def suite():  # pragma: no cover
    return unittest.makeSuite(HamsterHelperTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

