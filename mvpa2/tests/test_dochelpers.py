# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dochelpers"""

from mvpa2.base.dochelpers import single_or_plural, borrowdoc, borrowkwargs

import unittest

if __debug__:
    from mvpa2.base import debug

from mvpa2.testing.tools import SkipTest

class DochelpersTests(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(single_or_plural('a', 'b', 1), 'a')
        self.assertEqual(single_or_plural('a', 'b', 0), 'b')
        self.assertEqual(single_or_plural('a', 'b', 123), 'b')

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

        self.assertEqual(B.met1.__doc__, A.met1.__doc__)
        self.assertEqual(B.met2.__doc__, A.met1.__doc__)


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

        self.assertTrue('B.met1 doc' in B.met1.__doc__)
        for m in (B.met1,
                  B.met_nodoc,
                  B.met_nodockwargs,
                  B.met_excludes):
            docstring = m.__doc__
            self.assertTrue('Parameters' in docstring)
            self.assertTrue(not '*kwargs' in docstring,
                msg="We shouldn't carry kwargs in docstring now,"
                    "Got %r for %s" % (docstring, m))
            self.assertTrue('kp2 ' in docstring)
            self.assertTrue((('kp1 ' in docstring)
                                 ^ (m == B.met_excludes)))
            # indentation should have been squashed properly
            self.assertTrue(not '   ' in docstring)

        # some additional checks to see if we are not loosing anything
        self.assertTrue('Some postamble' in B.met1.__doc__)
        self.assertTrue('B.met_nodockwargs' in B.met_nodockwargs.__doc__)
        self.assertTrue('boguse' in B.met_excludes.__doc__)

    def test_searchlight_doc(self):
        # Searchlight __doc__ revealed issue of multiple enable_ca
        from mvpa2.measures.searchlight import Searchlight
        sldoc = Searchlight.__init__.__doc__
        self.assertEqual(sldoc.count('enable_ca'), 1)
        self.assertEqual(sldoc.count('disable_ca'), 1)


    def test_recursive_reprs(self):
        # https://github.com/PyMVPA/PyMVPA/issues/122

        from mvpa2.base.param import Parameter
        from mvpa2.base.state import ClassWithCollections

        class C1(ClassWithCollections):
            f = Parameter(None)

        class C2(ClassWithCollections):
            p = Parameter(None)
            def trouble(self, results):
                return results

        # provide non-default value of sl
        c1 = C1()
        c2 = C2(p=c1)
        # bind sl's results_fx to hsl's instance method
        c2.params.p.params.f = c2.trouble
        c1id = c2id = mod = ''
        # kaboom -- this should not crash now
        if __debug__:
            if 'ID_IN_REPR' in debug.active:
                from mvpa2.base.dochelpers import _strid
                c1id = _strid(c1)
                c2id = _strid(c2)

            if 'MODULE_IN_REPR' in debug.active:
                mod = 'mvpa2.tests.test_dochelpers.'
                raise SkipTest("TODO: needs similar handling in _saferepr")

        self.assertEqual(
            repr(c2), '%(mod)sC2(p=%(mod)sC1(f=<bound %(mod)sC2%(c2id)s.trouble>)%(c1id)s)%(c2id)s' % locals())

# TODO: more unittests
def suite():  # pragma: no cover
    return unittest.makeSuite(DochelpersTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

