# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dochelpers"""

from mvpa.base.dochelpers import single_or_plural, borrowdoc, borrowkwargs

import unittest

class DochelpersTests(unittest.TestCase):

    def test_basic(self):
        self.failUnlessEqual(single_or_plural('a', 'b', 1), 'a')
        self.failUnlessEqual(single_or_plural('a', 'b', 0), 'b')
        self.failUnlessEqual(single_or_plural('a', 'b', 123), 'b')

    def test_borrow_doc(self):

        class A(object):
            def met1(self):
                """met1doc"""
                pass
            def met2(self):
                """met2doc"""
                pass

        class B(object):
            @borrowdoc(A)
            def met1(self):
                pass
            @borrowdoc(A, 'met1')
            def met2(self):
                pass

        self.failUnlessEqual(B.met1.__doc__, A.met1.__doc__)
        self.failUnlessEqual(B.met2.__doc__, A.met1.__doc__)


    def test_borrow_kwargs(self):

        class A(object):
            def met1(self, kp1=None, kp2=1):
                """met1 doc

                Parameters
                ----------
                kp1 : None or int
                  keyword parameter 1
                kp2 : int, optional
                  something
                """
                pass

            def met2(self):
                """met2doc"""
                pass

        class B(object):

            @borrowkwargs(A)
            def met1(self, bu, **kwargs):
                """B.met1 doc

                Parameters
                ----------
                bu
                  description
                **kwargs
                  Same as in A.met1

                Some postamble
                """
                pass

            @borrowkwargs(A, 'met1')
            def met_nodoc(self, **kwargs):
                pass

            @borrowkwargs(A, 'met1')
            def met_nodockwargs(self, bogus=None, **kwargs):
                """B.met_nodockwargs

                Parameters
                ----------
                bogus
                  something
                """
                pass

            if True:
                # Just so we get different indentation level
                @borrowkwargs(A, 'met1', ['kp1'])
                def met_excludes(self, boguse=None, **kwargs):
                    """B.met_excludes

                    Parameters
                    ----------
                    boguse
                      something
                    """
                    pass

        self.failUnless('B.met1 doc' in B.met1.__doc__)
        for m in (B.met1,
                  B.met_nodoc,
                  B.met_nodockwargs,
                  B.met_excludes):
            docstring = m.__doc__
            self.failUnless('Parameters' in docstring)
            self.failUnless(not '*kwargs' in docstring,
                msg="We shouldn't carry kwargs in docstring now,"
                    "Got %r for %s" % (docstring, m))
            self.failUnless('kp2 ' in docstring)
            self.failUnless((('kp1 ' in docstring)
                                 ^ (m == B.met_excludes)))
            # indentation should have been squashed properly
            self.failUnless(not '   ' in docstring)

        # some additional checks to see if we are not loosing anything
        self.failUnless('Some postamble' in B.met1.__doc__)
        self.failUnless('B.met_nodockwargs' in B.met_nodockwargs.__doc__)
        self.failUnless('boguse' in B.met_excludes.__doc__)

    def test_searchlight_doc(self):
        # Searchlight __doc__ revealed issue of multiple enable_ca
        from mvpa.measures.searchlight import Searchlight
        sldoc = Searchlight.__init__.__doc__
        self.failUnlessEqual(sldoc.count('enable_ca'), 1)
        self.failUnlessEqual(sldoc.count('disable_ca'), 1)


# TODO: more unittests
def suite():
    return unittest.makeSuite(DochelpersTests)


if __name__ == '__main__':
    import runner

