# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA verbose and debug output"""

import unittest, re
from StringIO import StringIO

from mvpa2.base.verbosity import OnceLogger

from mvpa2.base import verbose, error

if __debug__:
    from mvpa2.base import debug
    debug.register('1', 'id 1')           # needed for testing
    debug.register('2', 'id 2')


## XXX There must be smth analogous in python... don't know it yet
# And it is StringIO
#class StringStream(object):
#    def __init__(self):
#        self.__str = ""
#
#    def __repr__(self):
#        return self.__str
#
#    def write(self, s):
#        self.__str += s
#
#    def clean(self):
#        self.__str = ""
#
class VerboseOutputTest(unittest.TestCase):

    def setUp(self):
        self.msg = "Test level 2"
        # output stream
        self.sout = StringIO()

        self.once = OnceLogger(handlers=[self.sout])

        # set verbose to 4th level
        self.__oldverbosehandlers = verbose.handlers
        self.__oldverbose_level = verbose.level
        verbose.handlers = []           # so debug doesn't spoil it
        verbose.level = 4
        if __debug__:
            self.__olddebughandlers = debug.handlers
            self.__olddebugactive = debug.active
            self.__olddebugmetrics = debug.metrics
            debug.active = ['1', '2', 'SLC']
            debug.handlers = [self.sout]
            debug.offsetbydepth = False
        verbose.handlers = [self.sout]

    def tearDown(self):
        if __debug__:
            debug.active = self.__olddebugactive
            debug.handlers = self.__olddebughandlers
            debug.metrics = self.__olddebugmetrics
            debug.offsetbydepth = True
        verbose.handlers = self.__oldverbosehandlers
        verbose.level = self.__oldverbose_level
        self.sout.close()


    def test_verbose_above(self):
        """Test if it doesn't output at higher levels"""
        verbose(5, self.msg)
        self.assertEqual(self.sout.getvalue(), "")


    def test_verbose_below(self):
        """Test if outputs at lower levels and indents
        by default with spaces
        """
        verbose(2, self.msg)
        self.assertEqual(self.sout.getvalue(),
                             "  %s\n" % self.msg)

    def test_verbose_indent(self):
        """Test indent symbol
        """
        verbose.indent = "."
        verbose(2, self.msg)
        self.assertEqual(self.sout.getvalue(), "..%s\n" % self.msg)
        verbose.indent = " "            # restore

    def test_verbose_negative(self):
        """Test if chokes on negative level"""
        self.assertRaises( ValueError,
                               verbose._set_level, -10 )

    def test_no_lf(self):
        """Test if it works fine with no newline (LF) symbol"""
        verbose(2, self.msg, lf=False)
        verbose(2, " continue ", lf=False)
        verbose(2, "end")
        verbose(0, "new %s" % self.msg)
        self.assertEqual(self.sout.getvalue(),
                             "  %s continue end\nnew %s\n" % \
                             (self.msg, self.msg))

    def test_cr(self):
        """Test if works fine with carriage return (cr) symbol"""
        verbose(2, self.msg, cr=True)
        verbose(2, "rewrite", cr=True)
        verbose(1, "rewrite 2", cr=True)
        verbose(1, " add", cr=False, lf=False)
        verbose(1, " finish")
        target = '\r  %s\r              \rrewrite' % self.msg + \
                 '\r       \rrewrite 2 add finish\n'
        self.assertEqual(self.sout.getvalue(), target)

    def test_once_logger(self):
        """Test once logger"""
        self.once("X", self.msg)
        self.once("X", self.msg)
        self.assertEqual(self.sout.getvalue(), self.msg+"\n")

        self.once("Y", "XXX", 2)
        self.once("Y", "XXX", 2)
        self.once("Y", "XXX", 2)
        self.assertEqual(self.sout.getvalue(), self.msg+"\nXXX\nXXX\n")


    def test_error(self):
        """Test error message"""
        error(self.msg, critical=False) # should not exit
        self.assertTrue(self.sout.getvalue().startswith("ERROR"))


    if __debug__:
        def test_debug(self):
            verbose.handlers = []           # so debug doesn't spoil it
            debug.active = ['1', '2', 'SLC']
            debug.metrics = debug._known_metrics.keys()
            # do not offset for this test
            debug('SLC', self.msg, lf=False)
            self.assertRaises(ValueError, debug, 3, 'bugga')
            #Should complain about unknown debug id
            svalue = self.sout.getvalue()
            regexp = "\[SLC\] DBG(?:{.*})?: %s" % self.msg
            rematch = re.match(regexp, svalue)
            self.assertTrue(rematch, msg="Cannot match %s with regexp %s" %
                            (svalue, regexp))
            # find metrics
            self.assertTrue('RSS/VMS:' in svalue,
                            msg="Cannot find vmem metric in " + svalue)
            self.assertTrue('>test_verbosity:' in svalue,
                            msg="Cannot find tbc metric in " + svalue)
            self.assertTrue(' sec' in svalue,
                            msg="Cannot find tbc metric in " + svalue)


        def test_debug_rgexp(self):
            verbose.handlers = []           # so debug doesn't spoil it
            debug.active = ['.*']
            # we should have enabled all of them
            self.assertEqual(set(debug.active),
                                 set(debug.registered.keys()))
            debug.active = ['S.*', 'CLF']
            self.assertEqual(set(debug.active),
                                 set(filter(lambda x:x.startswith('S'),
                                            debug.registered.keys())+['CLF']))
            debug.active = ['SG', 'CLF']
            self.assertEqual(set(debug.active), set(['SG', 'CLF']),
                                 msg="debug should do full line matching")

            debug.offsetbydepth = True


        # TODO: More tests needed for debug output testing

def suite():  # pragma: no cover
    return unittest.makeSuite(VerboseOutputTest)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

