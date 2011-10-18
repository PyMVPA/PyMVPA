# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection methods.

Brief Description of Available Methods
======================================

* `SensitivityBasedFeatureSelection` - generic class to provide feature
  selection given some sensitivity measure
* `RFE` - recursive feature elimination (RFE)
* `IFS` - incremental feature selection

"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.featsel')

if __debug__:
    debug('INIT', 'mvpa2.featsel end')
