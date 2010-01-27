# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""The module `mvpa.datasets` offers data storage and handling functionality.

Virtually any processing done with PyMVPA involves datasets -- the primary form
of data representation in PyMVPA. Datasets serve as containers for input data,
as well as the return datatype of more complex PyMVPA algorithms. The following
sections introduce some basic concepts of datasets and offers an overview of
typical procedures perform with them.

"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.datasets')

# nothing in here that works without the base class
from mvpa.datasets.base import Dataset, dataset_wizard
from mvpa.base.dataset import hstack, vstack

if __debug__:
    debug('INIT', 'mvpa.datasets end')
