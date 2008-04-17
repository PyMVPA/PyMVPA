#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to verify presence of external libraries and modules
"""

__docformat__ = 'restructuredtext'

from mvpa.misc import warning

if __debug__:
    from mvpa.misc import debug

# contains list of available (optional) external classifier extensions
_KNOWN = {'libsvm':'import svm as __; x=__._convert_to_svm_node_array',
          'shogun':'import shogun as __'}
present = []

for external,testcode in _KNOWN.iteritems():
    if __debug__:
        debug('EXT', "Checking for the presence of %s" % external)
    # conditional import of libsvm
    try:
        exec testcode
        present += [external]
    except:
        warning("Known external %s is not present, thus not available" % external)

if __debug__:
    debug('EXT', 'Following optional externals are present: %s' % `present`)

