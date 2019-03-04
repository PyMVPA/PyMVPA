# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Unit tests for basic constraints functionality.'''


from mvpa2.testing import *
import unittest

from mvpa2.base.constraints import *
import sys

class SimpleConstraintsTests(unittest.TestCase):

    def test_int(self):
        c = EnsureInt()
        # this should always work
        assert_equal(c(7), 7)
        assert_equal(c(7.0), 7)
        assert_equal(c('7'), 7)
        assert_equal(c([7, 3]), [7, 3])
        # this should always fail
        assert_raises(ValueError, lambda: c('fail'))
        assert_raises(ValueError, lambda: c([3, 'fail']))
        # this will also fail
        assert_raises(ValueError, lambda: c('17.0'))

    def test_float(self):
        c = EnsureFloat()
        # this should always work
        assert_equal(c(7.0), 7.0)
        assert_equal(c(7), 7.0)
        assert_equal(c('7'), 7.0)
        assert_equal(c([7.0, '3.0']), [7.0, 3.0])
        # this should always fail
        assert_raises(ValueError, lambda: c('fail'))
        assert_raises(ValueError, lambda: c([3.0, 'fail']))

    def test_bool(self):
        c = EnsureBool()
        # this should always work
        assert_equal(c(True), True)
        assert_equal(c(False), False)
        # all that resuls in True
        assert_equal(c('True'), True)
        assert_equal(c('true'), True)
        assert_equal(c('1'), True)
        assert_equal(c('yes'), True)
        assert_equal(c('on'), True)
        assert_equal(c('enable'), True)
        # all that resuls in False
        assert_equal(c('false'), False)
        assert_equal(c('False'), False)
        assert_equal(c('0'), False)
        assert_equal(c('no'), False)
        assert_equal(c('off'), False)
        assert_equal(c('disable'), False)
        # this should always fail
        assert_raises(ValueError, c, 0)
        assert_raises(ValueError, c, 1)

    def test_str(self):
        c = EnsureStr()
        # this should always work
        assert_equal(c('hello'), 'hello')
        assert_equal(c('7.0'), '7.0')
        # this should always fail
        assert_raises(ValueError, lambda: c(['ab']))
        assert_raises(ValueError, lambda: c(['a', 'b']))
        assert_raises(ValueError, lambda: c(('a', 'b')))
        # no automatic conversion attempted
        assert_raises(ValueError, lambda: c(7.0))

    def test_none(self):
        c = EnsureNone()
        # this should always work
        assert_equal(c(None), None)
        # this should always fail
        assert_raises(ValueError, lambda: c('None'))
        assert_raises(ValueError, lambda: c([]))

    def test_choice(self):
        c = EnsureChoice('choice1', 'choice2', None)
        # this should always work
        assert_equal(c('choice1'), 'choice1')
        assert_equal(c(None), None)
        # this should always fail
        assert_raises(ValueError, lambda: c('fail'))
        assert_raises(ValueError, lambda: c('None'))

    def test_range(self):
        c = EnsureRange(min=3, max=7)
        # this should always work
        assert_equal(c(3.0), 3.0)

        # Python 3 raises an TypeError if incompatible types are compared,
        # whereas Python 2 raises a ValueError
        type_error = TypeError if sys.version_info[0] >= 3 else ValueError

        # this should always fail
        assert_raises(ValueError, lambda: c(2.9999999))
        assert_raises(ValueError, lambda: c(77))
        assert_raises(type_error, lambda: c('fail'))
        assert_raises(type_error, lambda: c((3, 4)))
        # since no type checks are performed
        assert_raises(type_error, lambda: c('7'))

        # Range doesn't have to be numeric
        c = EnsureRange(min="e", max="qqq")
        assert_equal(c('e'), 'e')
        assert_equal(c('fa'), 'fa')
        assert_equal(c('qq'), 'qq')
        assert_raises(ValueError, c, 'a')
        assert_raises(ValueError, c, 'qqqa')


    def test_listof(self):
        c = EnsureListOf(str)
        assert_equal(c(['a', 'b']), ['a', 'b'])
        assert_equal(c(['a1', 'b2']), ['a1', 'b2'])

    def test_tupleof(self):
        c = EnsureTupleOf(str)
        assert_equal(c(('a', 'b')), ('a', 'b'))
        assert_equal(c(('a1', 'b2')), ('a1', 'b2'))
        
        
class ComplexConstraintsTests(unittest.TestCase):

    def test_constraints(self):
        # this should always work
        c = Constraints(EnsureFloat())
        assert_equal(c(7.0), 7.0)
        c = Constraints(EnsureFloat(), EnsureRange(min=4.0))
        assert_equal(c(7.0), 7.0)
        # __and__ form
        c = EnsureFloat() & EnsureRange(min=4.0)
        assert_equal(c(7.0), 7.0)
        assert_raises(ValueError, c, 3.9)
        c = Constraints(EnsureFloat(), EnsureRange(min=4), EnsureRange(max=9))
        assert_equal(c(7.0), 7.0)
        assert_raises(ValueError, c, 3.9)
        assert_raises(ValueError, c, 9.01)
        # __and__ form
        c = EnsureFloat() & EnsureRange(min=4) & EnsureRange(max=9)
        assert_equal(c(7.0), 7.0)
        assert_raises(ValueError, c, 3.99)
        assert_raises(ValueError, c, 9.01)
        # and reordering should not have any effect
        c = Constraints(EnsureRange(max=4), EnsureRange(min=9), EnsureFloat())
        assert_raises(ValueError, c, 3.99)
        assert_raises(ValueError, c, 9.01)

    def test_altconstraints(self):
        # this should always work
        c = AltConstraints(EnsureFloat())
        assert_equal(c(7.0), 7.0)
        c = AltConstraints(EnsureFloat(), EnsureNone())
        assert_equal(c(7.0), 7.0)
        assert_equal(c(None), None)
        # __or__ form
        c = EnsureFloat() | EnsureNone()
        assert_equal(c(7.0), 7.0)
        assert_equal(c(None), None)

        # this should always fail
        c = Constraints(EnsureRange(min=0, max=4), EnsureRange(min=9, max=11))
        assert_raises(ValueError, c, 7.0)
        c = EnsureRange(min=0, max=4) | EnsureRange(min=9, max=11)
        assert_equal(c(3.0), 3.0)
        assert_equal(c(9.0), 9.0)
        assert_raises(ValueError, c, 7.0)
        assert_raises(ValueError, c, -1.0)

    def test_both(self):
        # this should always work
        c = AltConstraints(Constraints(EnsureFloat(), \
                                      EnsureRange(min=7.0, max=44.0)), \
                                      EnsureNone())
        assert_equal(c(7.0), 7.0)
        assert_equal(c(None), None)
        # this should always fail
        assert_raises(ValueError, lambda: c(77.0))




if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()


