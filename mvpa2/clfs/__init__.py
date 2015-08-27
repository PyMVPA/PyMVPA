# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA classifiers

Module Organization
===================
mvpa2.clfs module contains various classifiers

:group Base: base
:group Meta Classifiers: meta
:group Specific Implementations: knn svm _svmbase plr ridge smlr libsmlrc gpr blr
:group External Interfaces: lars libsvmc sg
:group Utilities: transerror model_selector stats kernel distance
:group Warehouse of Classifiers: warehouse
"""

__docformat__ = 'restructuredtext'

from mvpa2.support.due import due, Doi

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.clfs')

due.cite(
    Doi('10.1007/b94608'),
    path="mvpa2.clfs",
    description="Thorough textbook on statistical learning (available online)",
    tags=["edu"])

if __debug__:
    debug('INIT', 'mvpa2.clfs end')
