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

def __check_shogun(bottom_version, custom_versions=[2456]):
    """Check if version of shogun is high enough (or custom known) to
    be enabled in the testsuite"""
    import shogun.Classifier as __sc
    ver = __sc.Version_get_version_revision()
    if (ver in custom_versions) or (ver >= bottom_version) : # custom built
        return True
    else:
        raise ImportError, 'Version %s is smaller than needed %s' % \
              (ver, bottom_version)


# contains list of available (optional) external classifier extensions
_KNOWN = {'libsvm':'import mvpa.clfs.libsvm._svm as __; x=__.convert2SVMNode',
          'nifti':'from nifti import NiftiImage as __',
          'shogun':'import shogun as __',
          'shogun.lightsvm': 'import shogun.Classifier as __; x=__.SVMLight',
          'shogun.svrlight': 'from shogun.Regression import SVRLight as __',
          'rpy': "import rpy",
          'lars': "import rpy; rpy.r.library('lars')",
          'pylab': "import pylab as __",
          'sg_fixedcachesize': "__check_shogun(3043)",
          }

_VERIFIED = {}

_caught_exceptions = [ImportError, AttributeError]
"""Exceptions which are silently caught while running tests for externals"""
try:
    import rpy
    _caught_exceptions += [rpy.RException]
except:
    pass

def exists(dep, force=False):
    """
    Test whether a known dependency is installed on the system.

    This method allows us to test for individual dependencies without
    testing all known dependencies. It also ensures that we only test
    for a dependency once.

    :Parameters:
      dep : string
        The dependency key to test.
      force : boolean
        Whether to force the test even if it has already been
        performed.

    """
    if _VERIFIED.has_key(dep) and not force:
        # we have already tested for it, so return our previous result
        return _VERIFIED[dep]
    elif not _KNOWN.has_key(dep):
        warning("%s is not a known dependency key." % (dep))
        return False
    else:
        # try and load the specific dependency
        # default to false
        _VERIFIED[dep] = False

        if __debug__:
            debug('EXT', "Checking for the presence of %s" % dep)

        try:
            exec _KNOWN[dep]
            _VERIFIED[dep] = True
        except tuple(_caught_exceptions):
            pass

        if __debug__:
            debug('EXT', "Presence of %s is%s verified" %
                  (dep, {True:'', False:' NOT'}[_VERIFIED[dep]]))

        return _VERIFIED[dep]


def testAllDependencies(force=False):
    """
    Test for all known dependencies.

    :Parameters:
      force : boolean
        Whether to force the test even if it has already been
        performed.

    """
    # loop over all known dependencies
    for dep in _KNOWN:
        if not exists(dep, force):
            warning("Known dependency %s is not present, thus not available." \
                    % dep)

    if __debug__:
        debug('EXT', 'The following optional externals are present: %s' \
                     % [ k for k in _VERIFIED.keys() if _VERIFIED[k]])
