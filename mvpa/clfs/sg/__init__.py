# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classifiers provided by shogun (sg) library"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.clfs.sg')

from mvpa.clfs.sg.svm import SVM

if __debug__:
    debug('INIT', 'mvpa.clfs.sg end')
