# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit test for PyMVPA mvpa.suite() of being loading ok"""

import unittest


class SuiteTest(unittest.TestCase):

    def testSuiteLoad(self):
        """Test if we are loading fine
        """
        try:
            exec "from mvpa.suite import *"
        except Exception, e:
            self.fail(msg="Cannot import everything from mvpa.suite."
                      "Getting %s" % e)

    def testDocstrings(self):
        from mvpa.suite import suite_stats
        # Lets do compliance checks
        # Get gross information on what we have in general
        gs = suite_stats()

        # all functions/classes/types should have some docstring
        missing = []
        for c in ('classes', 'functions', 'modules', 'objects',
                  'types'):
            missing1 = []
            for k, i in gs[c].iteritems():
                try:
                    s = i.__doc__.strip()
                except:
                    s = ""
                if s == "":
                    missing1.append(k)
            if len(missing1):
                missing.append("%s: " % c + ', '.join(missing1))
        if len(missing):
            self.fail("Some items have missing docstrings:\n "
                      + '\n '.join(missing))

def suite():
    return unittest.makeSuite(SuiteTest)


if __name__ == '__main__':
    import runner

