# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

import numpy as np
import time

from mvpa2.testing import *
from mvpa2.base.progress import ProgressBar


class ProgressTests(unittest.TestCase):
    def test_progress(self):

        pre = '+0:00:02 ===='
        post = '===________ -0:00:01  <msg>'

        for show in [True, False]:
            p = ProgressBar(progress_bar_width=20,
                            show_percentage=show)
            t = time.time()
            p.start(t)

            while time.time() < t + 2:
                pass

            s = p(.6, '<msg>')

            infix = '[60%]' if show else '====='
            assert_equal(s, pre + infix + post)


def suite():  # pragma: no cover
    return unittest.makeSuite(SensitivityAnalysersTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
