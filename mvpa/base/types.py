# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Things concerned with types and type-checking in PyMVPA"""

def is_datasetlike(obj):
    """Check if an object looks like a Dataset."""
    if hasattr(obj, 'samples') and \
       hasattr(obj, 'sa') and \
       hasattr(obj, 'fa') and \
       hasattr(obj, 'a'):
        return True

    return False
