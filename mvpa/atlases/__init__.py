#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA anatomical atlases

Module Organization
===================
mvpa.atlases module contains support for various atlases

.. packagetree::
   :style: UML

:group Base: BaseAtlas XMLBasedAtlas Label Level LabelsLevel
:group Talairach: RumbaAtlas LabelsAtlas ReferencesAtlas
:group Atlases from FSL: FSLAtlas FSLProbabilisticAtlas
:group Exceptions: XMLAtlasException
"""

__docformat__ = 'restructuredtext'


if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.atlases')

from base import LabelsAtlas, ReferencesAtlas, FSLProbabilisticAtlas

if __debug__:
    debug('INIT', 'mvpa.atlases end')
