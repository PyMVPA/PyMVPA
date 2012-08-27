# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Things concerned with types and type-checking in PyMVPA"""

import sys
import numpy as np


def is_datasetlike(obj):
    """Check if an object looks like a Dataset."""
    if hasattr(obj, 'samples') and \
       hasattr(obj, 'sa') and \
       hasattr(obj, 'fa') and \
       hasattr(obj, 'a'):
        return True

    return False


def accepts_dataset_as_samples(fx):
    """Decorator to extract samples from Datasets.

    Little helper to allow methods to be written for plain data (if they
    don't need information from a Dataset), but at the same time also
    accept whole Datasets as input.
    """
    def extract_samples(obj, data):
        if is_datasetlike(data):
            return fx(obj, data.samples)
        else:
            return fx(obj, data)
    return extract_samples


def asobjarray(x):
    """Generates numpy.ndarray with dtype object from an iterable

    Is needed to assure object dtype, so first empty array of
    dtype=object needs to be constructed and then only items to be
    assigned.

    Parameters
    ----------
    x : list or tuple or ndarray
    """
    res = np.empty(len(x), dtype=object)
    res[:] = x
    return res

# compatibility layer for Python3
if sys.version_info[0] < 3:

    from operator import isSequenceType as is_sequence_type

    def as_char(x):
        """Identity mapping in python2"""
        return x

else:

    def is_sequence_type(inst):
        """Return True if an instance is of an iterable type

        Verified by wrapping with iter() call
        """
        try:
            _ = iter(inst)
            return True
        except:
            return False

    import codecs

    def as_char(x):
        """Return character representation for a unicode symbol"""
        return codecs.latin_1_encode(x)[0]
