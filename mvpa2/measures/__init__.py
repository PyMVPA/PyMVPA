# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA measures.

Module Description
==================

Provide some measures given a dataset. Most of the time, derivatives of
`FeaturewiseMeasure` are used, such as

* `OneWayAnova`
* `CorrCoef`
* `IterativeRelief`
* `NoisePerturbationSensitivity`

Also many classifiers natively provide sensitivity estimators via the call to
`get_sensitivity_analyzer` method
"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.measures')

if __debug__:
    debug('INIT', 'mvpa2.measures end')
