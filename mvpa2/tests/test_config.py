# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dense array mapper"""


import unittest
from mvpa2.base.config import ConfigManager

class ConfigTests(unittest.TestCase):

    def test_config(self):
        cfg = ConfigManager()

        # does nothing so far, but will be used to test the default
        # configuration from doc/examples/pymvpa2.cfg

        # query for some non-existing option and check if default is returned
        query = cfg.get('dasgibtsdochnicht', 'neegarnicht', default='required')
        self.assertTrue(query == 'required')



def suite():  # pragma: no cover
    return unittest.makeSuite(ConfigTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

