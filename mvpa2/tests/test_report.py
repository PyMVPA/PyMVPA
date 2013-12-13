# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA simple report facility"""

import unittest, os, shutil

from mvpa2.base import verbose, externals

from mvpa2.base.report_dummy import Report as DummyReport
_test_classes = [ DummyReport ]

from mvpa2.testing import sweepargs, with_tempfile

if externals.exists('reportlab', raise_=False):
    from mvpa2.base.report import Report
    _test_classes += [ Report ]

if __debug__:
    from mvpa2.base import debug

class ReportTest(unittest.TestCase):
    """Just basic testing of reports -- pretty much that nothing fails
    """

    def setUp(self):
        # preserve handlers/level for verbose
        self.__oldverbosehandlers = verbose.handlers
        self.__oldverbose_level = verbose.level


    def tearDown(self):
        verbose.handlers = self.__oldverbosehandlers
        verbose.level = self.__oldverbose_level

    ##REF: Name was automagically refactored
    def aux_basic(self, dirname, rc):
        """Helper function -- to assure that all filehandlers
           get closed so we could remove trash directory.

           Otherwise -- .nfs* files on NFS-mounted drives cause problems
           """
        report = rc('UnitTest report',
                    title="Sample report for testing",
                    path=dirname)
        isdummy = isinstance(report, DummyReport)

        verbose.handlers = [report]
        verbose.level = 3
        verbose(1, "Starting")
        verbose(2, "Level 2")

        if not isdummy:
            self.assertTrue(len(report._story) == 2,
                            msg="We should have got some lines from verbose")

        if __debug__:
            odhandlers = debug.handlers
            debug.handlers = [report]
            oactive = debug.active
            debug.active = ['TEST'] + debug.active
            debug('TEST', "Testing report as handler for debug")
            if not isdummy:
                self.assertTrue(len(report._story) == 4,
                            msg="We should have got some lines from debug")
            debug.active = oactive
            debug.handlers = odhandlers

        os.makedirs(dirname)

        if externals.exists('pylab plottable'):
            if not isdummy:
                clen = len(report._story)
            import pylab as pl
            pl.ioff()
            pl.close('all')
            pl.figure()
            pl.plot([1, 2], [3, 2])

            pl.figure()
            pl.plot([2, 10], [3, 2])
            pl.title("Figure 2 must be it")
            report.figures()

            if not isdummy:
                self.assertTrue(
                    len(report._story) == clen+2,
                    msg="We should have got some lines from figures")

        report.text("Dugi bugi")
        # make sure we don't puke on xml like text with crap
        report.text("<kaj>$lkj&*()^$%#%</kaj>")
        report.text("locals:\n%s globals:\n%s" % (`locals()`, `globals()`))
        # bloody XML - just to check that there is no puke
        report.xml("<b>Dugi bugi</b>")
        report.save()

        if externals.exists('pylab'):
            import pylab as pl
            pl.close('all')
            pl.ion()

        pass


    @with_tempfile()
    @sweepargs(rc=_test_classes)
    def test_basic(self, dirname, rc):
        """Test all available reports, real or dummy for just working
        """
        self.aux_basic(dirname, rc)
        # cleanup
        shutil.rmtree(dirname, ignore_errors=True)


def suite():  # pragma: no cover
    return unittest.makeSuite(ReportTest)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
