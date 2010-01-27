# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset that gets its samples from an EEP binary file"""

__docformat__ = 'restructuredtext'

from mvpa.datasets import Dataset
from mvpa.misc.io.eepbin import EEPBin

def eep_dataset(samples, labels=None, chunks=None):
    """Create a dataset using an EEP binary file as source.

    EEP files are used by *eeprobe* a software for analysing even-related
    potentials (ERP), which was developed at the Max-Planck Institute for
    Cognitive Neuroscience in Leipzig, Germany.

      http://www.ant-neuro.com/products/eeprobe

    Parameters
    ----------
    samples : str or EEPBin instance
      This is either a filename of an EEP file, or an EEPBin instance, providing
      the samples data in EEP format.
    labels, chunks : sequence or scalar or None
      Values are pass through to `Dataset.from_wizard()`. See its documentation
      for more information.

    Returns
    -------
    Dataset
      Besides is usual attributes (e.g. labels, chunks, and a mapper). The
      returned dataset also includes feature attributes associating each same
      with a channel (by id), and a specific timepoint -- based on information
      read from the EEP data.
    """
    if isinstance(samples, str):
        # open the eep file
        eb = EEPBin(samples)
    elif isinstance(samples, EEPBin):
        # nothing special
        eb = samples
    else:
        raise ValueError("eep_dataset takes the filename of an "
              "EEP file or a EEPBin object as 'samples' argument.")

    # init dataset
    ds = Dataset.from_channeltimeseries(
            eb.data, labels=labels, chunks=chunks, t0=eb.t0, dt=eb.dt,
            channelids=eb.channels)
    return ds
