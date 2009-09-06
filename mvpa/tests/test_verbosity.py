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

from mvpa.base.verbosity import OnceLogger

from mvpa.base import verbose, error

if __debug__:
    from mvpa.base import debug
    debug.register('1', 'id 1')           # needed for testing
    debug.register('2', 'id 2')

    from sets import Set

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
        verbose.handlers = []           # so debug doesn't spoil it
        verbose.level = 4
        if __debug__:
            self.__olddebughandlers = debug.handlers
            self.__olddebugactive = debug.active
            debug.active = ['1', '2', 'SLC']
            debug.handlers = [self.sout]
            debug.offsetbydepth = False

        verbose.handlers = [self.sout]

    def tearDown(self):
        if __debug__:
            debug.active = self.__olddebugactive
            debug.handlers = self.__olddebughandlers
            debug.offsetbydepth = True
        verbose.handlers = self.__oldverbosehandlers
        self.sout.close()


    def testVerboseAbove(self):
        """Test if it doesn't output at higher levels"""
        verbose(5, self.msg)
        self.failUnlessEqual(self.sout.getvalue(), "")


    def testVerboseBelow(self):
        """Test if outputs at lower levels and indents
        by default with spaces
        """
        verbose(2, self.msg)
        self.failUnlessEqual(self.sout.getvalue(),
                             "  %s\n" % self.msg)

    def testVerboseIndent(self):
        """Test indent symbol
        """
        verbose.indent = "."
        verbose(2, self.msg)
        self.failUnlessEqual(self.sout.getvalue(), "..%s\n" % self.msg)
        verbose.indent = " "            # restore

    def testVerboseNegative(self):
        """Test if chokes on negative level"""
        self.failUnlessRaises( ValueError,
                               verbose._setLevel, -10 )

    def testNoLF(self):
        """Test if it works fine with no newline (LF) symbol"""
        verbose(2, self.msg, lf=False)
        verbose(2, " continue ", lf=False)
        verbose(2, "end")
        verbose(0, "new %s" % self.msg)
        self.failUnlessEqual(self.sout.getvalue(),
                             "  %s continue end\nnew %s\n" % \
                             (self.msg, self.msg))

    def testCR(self):
        """Test if works fine with carriage return (cr) symbol"""
        verbose(2, self.msg, cr=True)
        verbose(2, "rewrite", cr=True)
        verbose(1, "rewrite 2", cr=True)
        verbose(1, " add", cr=False, lf=False)
        verbose(1, " finish")
        target = '\r  %s\r              \rrewrite' % self.msg + \
                 '\r       \rrewrite 2 add finish\n'
        self.failUnlessEqual(self.sout.getvalue(), target)

    def testOnceLogger(self):
        """Test once logger"""
        self.once("X", self.msg)
        self.once("X", self.msg)
        self.failUnlessEqual(self.sout.getvalue(), self.msg+"\n")

        self.once("Y", "XXX", 2)
        self.once("Y", "XXX", 2)
        self.once("Y", "XXX", 2)
        self.failUnlessEqual(self.sout.getvalue(), self.msg+"\nXXX\nXXX\n")


    def testError(self):
        """Test error message"""
        error(self.msg, critical=False) # should not exit
        self.failUnless(self.sout.getvalue().startswith("ERROR"))


    if __debug__:
        def testDebug(self):
            verbose.handlers = []           # so debug doesn't spoil it
            debug.active = ['1', '2', 'SLC']
            # do not offset for this test
            debug('SLC', self.msg, lf=False)
            self.failUnlessRaises(ValueError, debug, 3, 'bugga')
            #Should complain about unknown debug id
            svalue = self.sout.getvalue()
            regexp = "\[SLC\] DBG(?:{.*})?: %s" % self.msg
            rematch = re.match(regexp, svalue)
            self.failUnless(rematch, msg="Cannot match %s with regexp %s" %
                            (svalue, regexp))


        def testDebugRgexp(self):
            verbose.handlers = []           # so debug doesn't spoil it
            debug.active = ['.*']
            # we should have enabled all of them
            self.failUnlessEqual(Set(debug.active),
                                 Set(debug.registered.keys()))
            debug.active = ['S.*', 'CLF']
            self.failUnlessEqual(Set(debug.active),
                                 Set(filter(lambda x:x.startswith('S'),
                                            debug.registered.keys())+['CLF']))
            debug.active = ['SG', 'CLF']
            self.failUnlessEqual(Set(debug.active), Set(['SG', 'CLF']),
                                 msg="debug should do full line matching")

            debug.offsetbydepth = True


        # TODO: More tests needed for debug output testing

def suite():
    return unittest.makeSuite(VerboseOutputTest)


if __name__ == '__main__':
    import runner

