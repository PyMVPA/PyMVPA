#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Compute the percentage of correct classifications of"""

from mvpa.misc.errorfx import MeanMatchErrorFx
from mvpa.splitprocessor import *

class MeanMatchProcessor(SplitProcessor):
    """ Computes the percentage of correct classifications of a
    cross-validation fold.
    """
    def __call__(self, splitter, split, classifier):
        predictions = classifier.predict( split[1].samples )
        return MeanMatchErrorFx()( predictions, split[1].labels )
