# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""I/O support for neuroimaging dataset in BIDS_ format

.. _BIDS: http://bids.neuroimaging.io
"""

__docformat__ = 'restructuredtext'

import numpy as np


def load_events(fname, as_recarr=False):
    """Load event specifications from _events.tsv files

    Parameters
    ----------
    as_recarr : bool
      If True, return events as a NumPy recarray with field
      types corresponding to the columns in the TSV file.
      Otherwise return PyMVPA's standard list of dictionaries
      with one dictionary per event.
    """
    # always load via recfromcsv to get uniform type conversion
    evrec = np.recfromcsv(fname, delimiter='\t')
    if as_recarr:
        return evrec
    columns = evrec.dtype.names
    return [dict(zip(columns, ev)) for ev in evrec]
