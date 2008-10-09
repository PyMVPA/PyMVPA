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

from mvpa.base import warning
from mvpa import cfg

if __debug__:
    from mvpa.base import debug

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


def __check_weave():
    """Apparently presence of scipy is not sufficient since some
    versions experience problems. E.g. in Sep,Oct 2008 lenny's weave
    failed to work. May be some other converter could work (? See
    http://lists.debian.org/debian-devel/2008/08/msg00730.html for a
    similar report.

    Following simple snippet checks compilation of the basic code using
    weave
    """
    from scipy import weave
    from scipy.weave import converters, build_tools
    import numpy as N
    # to shut weave up
    import sys
    # we can't rely on weave at all at the restoring argv. On etch box
    # restore_sys_argv() is apparently is insufficient
    oargv = sys.argv[:]
    ostdout = sys.stdout
    if not( __debug__ and 'EXT_' in debug.active):
        from StringIO import StringIO
        sys.stdout = StringIO()
        # *nix specific solution to shut weave up.
        # Some users must complain and someone
        # needs to fix this to become more generic.
        cargs = [">/dev/null", "2>&1"]
    else:
        cargs = []
    fmsg = None
    try:
        data = N.array([1,2,3])
        counter = weave.inline("data[0]=fabs(-1);", ['data'],
                               type_converters=converters.blitz,
                               verbose=0,
                               extra_compile_args=cargs,
                               compiler = 'gcc')
    except Exception, e:
        fmsg = "Failed to build simple weave sample." \
               " Exception was %s" % str(e)

    sys.stdout = ostdout
    # needed to fix sweave which might "forget" to restore sysv
    # build_tools.restore_sys_argv()
    sys.argv = oargv
    if fmsg is not None:
        raise ImportError, fmsg
    else:
        return "Everything is cool"


# contains list of available (optional) external classifier extensions
_KNOWN = {'libsvm':'import mvpa.clfs.libsvm._svm as __; x=__.convert2SVMNode',
          'nifti':'from nifti import NiftiImage as __',
          'ctypes':'import ctypes as __',
          'shogun':'import shogun as __',
          'shogun.mpd': 'import shogun.Classifier as __; x=__.MPDSVM',
          'shogun.lightsvm': 'import shogun.Classifier as __; x=__.SVMLight',
          'shogun.svrlight': 'from shogun.Regression import SVRLight as __',
          'scipy': "import scipy as __",
          'weave': "__check_weave()",
          'pywt': "import pywt as __",
          'rpy': "import rpy as __",
          'lars': "import rpy; rpy.r.library('lars')",
          'pylab': "import pylab as __",
          'openopt': "import scikits.openopt as __",
          'mdp': "import mdp as __",
          'sg_fixedcachesize': "__check_shogun(3043)",
          'hcluster': "import hcluster as __",
          'griddata': "import griddata as __",
          }

_VERIFIED = {}

_caught_exceptions = [ImportError, AttributeError, RuntimeError]
"""Exceptions which are silently caught while running tests for externals"""
try:
    import rpy
    _caught_exceptions += [rpy.RException]
except:
    pass

def exists(dep, force=False, raiseException=False):
    """
    Test whether a known dependency is installed on the system.

    This method allows us to test for individual dependencies without
    testing all known dependencies. It also ensures that we only test
    for a dependency once.

    :Parameters:
      dep : string or list of string
        The dependency key(s) to test.
      force : boolean
        Whether to force the test even if it has already been
        performed.
      raiseException : boolean
        Whether to raise RintimeError if dependency is missing.

    """
    # if we are provided with a list of deps - go through all of them
    if isinstance(dep, list) or isinstance(dep, tuple):
        results = [ exists(dep_, force, raiseException) for dep_ in dep ]
        return bool(reduce(lambda x,y: x and y, results, True))

    if _VERIFIED.has_key(dep) and not force:
        # we have already tested for it, so return our previous result
        result = _VERIFIED[dep]
    elif not _KNOWN.has_key(dep):
        warning("%s is not a known dependency key." % (dep))
        result = False
    else:
        # try and load the specific dependency
        # default to false
        _VERIFIED[dep] = False

        if __debug__:
            debug('EXT', "Checking for the presence of %s" % dep)

        estr = ''
        try:
            exec _KNOWN[dep]
            _VERIFIED[dep] = True
        except tuple(_caught_exceptions), e:
            estr = ". Caught exception was: " + str(e)

        if __debug__:
            debug('EXT', "Presence of %s is%s verified%s" %
                  (dep, {True:'', False:' NOT'}[_VERIFIED[dep]], estr))

        result = _VERIFIED[dep]

    if not result and raiseException and cfg.getboolean('apidoc', 'raise exception', True):
        raise RuntimeError, "Required external '%s' was not found" % dep

    return result


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
            warning("Known dependency %s is not present or is broken, " \
                    "thus not available." % dep)

    if __debug__:
        debug('EXT', 'The following optional externals are present: %s' \
                     % [ k for k in _VERIFIED.keys() if _VERIFIED[k]])
