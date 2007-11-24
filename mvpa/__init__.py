#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""MultiVariate Pattern Analysis


Package Organization
====================
The epydoc package contains the following subpackages and modules:

.. packagetree::
   :style: UML

:group Basic Data Structures: datasets
:group Classifiers: clf
:group Algorithms: algorithms
:group Miscellaneous: misc

:author: `Michael Hanke <michael.hanke@gmail.com>`__
:requires: Python 2.4+
:version: XXX
:see: `The PyMVPA webpage <http://XXX>`__
:see: `GIT Repository Browser <http://git.debian.org/?p=pkg-exppsy/pymvpa.git;a=summary>`__

:license: The MIT License
:copyright: |copy| 2006-2007 Michael Hanke <michael.hanke@gmail.com>

:newfield contributor: Contributor, Contributors (Alphabetical Order)
:contributor: `Yaroslav Halchenko  <mailto:debian@onerussian.com>`__

.. |copy| unicode:: 0xA9 .. copyright sign
"""

__docformat__ = 'restructuredtext'


if not __debug__:
# TODO: psyco should be moved upstairs anyways
    try:
        import psyco
        psyco.profile()
    except:
        from mvpa.misc import verbose
        verbose(5, "Psyco online compilation is not enabled in knn")

#from mvpa.dataset import *
