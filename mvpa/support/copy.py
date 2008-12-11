#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for python's copy module.

"""

__docformat__ = 'restructuredtext'

import sys

# We have to use deepcopy from python 2.5, since otherwise it fails to
# copy sensitivity analyzers with assigned combiners which are just
# functions not functors
if sys.version_info[0] > 2 or sys.version_info[1] > 4:
    # little trick to be able to import 'copy' package (which has same name)
    oldname = __name__
    # crazy name with close to zero possibility to cause whatever
    __name__ = 'iaugf9zrkjsbdv8'
    from copy import copy, deepcopy
    # restore old settings
    __name__ = oldname
else:
    from mvpa.support._copy import copy, deepcopy
