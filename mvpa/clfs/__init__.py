#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA classifiers

Module Organization
===================
mvpa.clfs module contains various classifiers

.. packagetree::
   :style: UML

:group Basic: classifier
:group Specific Implementations: knn svm plr ridge smlr
:group Internal Implementations: libsvm
:group Utilities: transerror
"""

__docformat__ = 'restructuredtext'


if __debug__:
    from mvpa.misc import debug
    debug('INIT', 'mvpa.clfs')

if __debug__:
    debug('INIT', 'mvpa.clfs end')
