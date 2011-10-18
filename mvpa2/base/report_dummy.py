# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dummy report class, to just be there in case if reportlab is not available.

"""

__docformat__ = 'restructuredtext'


from mvpa2.base import warning

if __debug__:
    from mvpa2.base import debug

def _dummy(*args, **kwargs):
    pass


class Report(object):
    """Dummy report class which does nothing but complains if used

    """

    def __init__(self, *args, **kwargs):
        """Initialize dummy report
        """
        warning("You are using DummyReport - no action will be taken. "
                "Please install reportlab to enjoy reporting facility "
                "within PyMVPA")

    def __getattribute__(self, index):
        """
        """
        # returns a dummy function
        return _dummy
