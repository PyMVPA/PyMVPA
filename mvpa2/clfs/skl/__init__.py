# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classifiers provided by scikit-learn (skl) library"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.clfs.skl')

# to pacify the nose
# (see e.g. http://nipy.bic.berkeley.edu/builders/pymvpa-py2.7-osx-10.8/builds/4/steps/shell_3/logs/stdio)
# import submodules only if skl they need is available
if externals.exists('skl'):
    from mvpa2.clfs.skl.base import SKLLearnerAdapter

if __debug__:
    debug('INIT', 'mvpa2.clfs.skl end')
