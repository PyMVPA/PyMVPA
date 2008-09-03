#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dense array mapper"""


import unittest
from mvpa.base.config import ConfigManager

class ConfigTests(unittest.TestCase):

    def testConfig(self):
        cfg = ConfigManager()

        # does nothing so far, but will be used to test the default
        # configuration from doc/examples/pymvpa.cfg

        # query for some non-existing option and check if default is returned
        query = cfg.get('dasgibtsdochnicht', 'neegarnicht', default='required')
        self.failUnless(query == 'required')



def suite():
    return unittest.makeSuite(ConfigTests)


if __name__ == '__main__':
    import runner

