# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA anatomical atlases

Module Organization
===================
mvpa2.atlases module contains support for various atlases

:group Base Implementations: base
:group Atlases from FSL: fsl
:group Helpers: warehouse transformation
"""

__docformat__ = 'restructuredtext'


if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.atlases')

from mvpa2.base import externals

# to pacify the nose
# (see e.g. http://nipy.bic.berkeley.edu/builders/pymvpa-py2.7-osx-10.8/builds/4/steps/shell_3/logs/stdio)
# import submodules only if lxml they need is available
if externals.exists('lxml'):
    from mvpa2.atlases.base import LabelsAtlas, ReferencesAtlas, XMLAtlasException
    from mvpa2.atlases.fsl import FSLProbabilisticAtlas
    from mvpa2.atlases.warehouse import Atlas, KNOWN_ATLASES, KNOWN_ATLAS_FAMILIES

if __debug__:
    debug('INIT', 'mvpa2.atlases end')
