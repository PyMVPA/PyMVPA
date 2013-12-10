# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit test for PyMVPA mvpa2.suite() of being loading ok"""

import inspect
import re
import sys
import unittest

from mvpa2.base.dochelpers import get_docstring_split

class SuiteTest(unittest.TestCase):

    def test_suite_load(self):
        """Test if we are loading fine
        """
        try:
            exec "from mvpa2.suite import *"
        except Exception, e: # pragma: no cover - should not be hit if ok_
            self.fail(msg="Cannot import everything from mvpa2.suite: %s" % e)

    def test_docstrings(self):
        #import mvpa2.suite as mv
        from mvpa2.suite import suite_stats
        # Lets do compliance checks
        # Get gross information on what we have in general
        #mv_scope = dict((x, getattr(mv, x)) for x in dir(mv))
        gs = suite_stats()#mv_scope)

        # all functions/classes/types should have some docstring
        missing = []
        # We should not have both :Parameters: and new style Parameters
        conflicting = []
        con_re1 = re.compile(':Parameters?:')
        con_re2 = re.compile('(?::Parameters?:.*Parameters?\s*\n\s*-------'
                             '|Parameters?\s*\n\s*-------.*:Parameters?:)',
                             flags=re.DOTALL)
        for c in ('functions', 'modules', 'objects', 'types') \
          + ('classes',) if sys.version_info[0] < 3 else ():
            missing1 = []
            conflicting1 = []
            self.assertTrue(gs[c])
            for k, i in gs[c].iteritems():
                try:
                    s = i.__doc__.strip()
                except: # pragma: no cover - should not be hit if ok_
                    s = ""
                if s == "":
                    missing1.append(k)

                if hasattr(i, '__init__') and not c in ['objects']:
                    # Smoke test get_docstring_split which would be used
                    # if someone specifies incorrect keyword argument
                    _ = get_docstring_split(i.__init__)
                    #if not None in _:
                    #    print [x[0] for x in _[1]]
                    si = i.__init__.__doc__
                    k += '.__init__'
                    if si is None or si == "":
                        try:
                            i_file = inspect.getfile(i)
                            if i_file == inspect.getfile(i.__init__) \
                               and 'mvpa' in i_file:
                                # if __init__ wasn't simply taken from some parent
                                # which is not within MVPA
                                missing1.append(k)
                        except TypeError:
                            # for some things like 'debug' inspect can't figure path
                            # just skip for now
                            pass
                else:
                    si = s

                if si is not None \
                       and  con_re1.search(si) and con_re2.search(si):
                    conflicting1.append(k)
            if len(missing1): # pragma: no cover - should not be hit if ok_
                missing.append("%s: " % c + ', '.join(missing1))
            if len(conflicting1): # pragma: no cover - should not be hit if ok_
                conflicting.append("%s: " % c + ', '.join(conflicting1))

        sfailures = []
        if len(missing): # pragma: no cover - should not be hit if ok_
            sfailures += ["Some items have missing docstrings:\n "
                          + '\n '.join(missing)]
        if len(conflicting): # pragma: no cover - should not be hit if ok_
            sfailures += ["Some items have conflicting formats of docstrings:\n "
                      + '\n '.join(conflicting)]
        if len(sfailures): # pragma: no cover - should not be hit if ok_
            self.fail('\n'.join(sfailures))


def suite():  # pragma: no cover
    return unittest.makeSuite(SuiteTest)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

