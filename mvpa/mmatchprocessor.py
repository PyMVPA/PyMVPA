### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Compute the percentage of correct classifications of
#            cross-validation folds
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from mmatcherrorfx import *
from splitprocessor import *

class MeanMatchProcessor(SplitProcessor):
    """ Computes the percentage of correct classifications of a
    cross-validation fold.
    """
    def __call__(self, splitter, split, classifier):
        predictions = classifier.predict( split[1].samples )
        return MeanMatchErrorFx()( predictions, split[1].labels )
