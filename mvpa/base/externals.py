# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to verify presence of external libraries and modules
"""

__docformat__ = 'restructuredtext'
import os

from mvpa.base import warning
from mvpa import cfg
from mvpa.misc.support import SmartVersion

if __debug__:
    from mvpa.base import debug

versions = {}
"""Versions of available externals, as tuples
"""


def __check_scipy():
    """Check if scipy is present an if it is -- store its version
    """
    import warnings
    # To don't allow any crappy warning to sneak in
    warnings.simplefilter('ignore', DeprecationWarning)
    try:
        import scipy as sp
    except:
        warnings.simplefilter('default', DeprecationWarning)
        raise
    warnings.simplefilter('default', DeprecationWarning)
    # Infiltrate warnings if necessary
    numpy_ver = versions['numpy']
    scipy_ver = versions['scipy'] = SmartVersion(sp.__version__)
    # There is way too much deprecation warnings spit out onto the
    # user. Lets assume that they should be fixed by scipy 0.7.0 time
    if scipy_ver >= "0.6.0" and scipy_ver < "0.7.0" \
        and numpy_ver > "1.1.0":
        import warnings
        if not __debug__ or (__debug__ and not 'PY' in debug.active):
            debug('EXT', "Setting up filters for numpy DeprecationWarnings")
            filter_lines = [
                ('NumpyTest will be removed in the next release.*',
                 DeprecationWarning),
                ('PyArray_FromDims: use PyArray_SimpleNew.',
                 DeprecationWarning),
                ('PyArray_FromDimsAndDataAndDescr: use PyArray_NewFromDescr.',
                 DeprecationWarning),
                # Trick re.match, since in warnings absent re.DOTALL in re.compile
                ('[\na-z \t0-9]*The original semantics of histogram is scheduled to be.*'
                 '[\na-z \t0-9]*', Warning) ]
            for f, w in filter_lines:
                warnings.filterwarnings('ignore', f, w)


def __check_numpy():
    """Check if numpy is present (it must be) an if it is -- store its version
    """
    import numpy as N
    versions['numpy'] = SmartVersion(N.__version__)


def __check_pywt(features=None):
    """Check for available functionality within pywt

    :Parameters:
      features : list of basestring
        List of known features to check such as 'wp reconstruct',
        'wp reconstruct fixed'
    """
    import pywt
    import numpy as N
    data = N.array([ 0.57316901,  0.65292526,  0.75266733,  0.67020084,  0.46505364,
                     0.76478331,  0.33034164,  0.49165547,  0.32979941,  0.09696717,
                     0.72552711,  0.4138999 ,  0.54460628,  0.786351  ,  0.50096306,
                     0.72436454, 0.2193098 , -0.0135051 ,  0.34283984,  0.65596245,
                     0.49598417,  0.39935064,  0.26370727,  0.05572373,  0.40194438,
                     0.47004551,  0.60327258,  0.25628266,  0.32964893,  0.24009889,])
    mode = 'per'
    wp = pywt.WaveletPacket(data, 'sym2', mode)
    wp2 = pywt.WaveletPacket(data=None, wavelet='sym2', mode=mode)
    try:
        for node in wp.get_level(2): wp2[node.path] = node.data
    except:
        raise ImportError, \
               "Failed to reconstruct WP by specifying data in the layer"

    if 'wp reconstruct fixed' in features:
        rec = wp2.reconstruct()
        if N.linalg.norm(rec[:len(data)] - data) > 1e-3:
            raise ImportError, \
                  "Failed to reconstruct WP correctly"
    return True


def __check_libsvm_verbosity_control():
    """Check for available verbose control functionality
    """
    import mvpa.clfs.libsvmc._svmc as _svmc
    try:
        _svmc.svm_set_verbosity(0)
    except:
        raise ImportError, "Provided version of libsvm has no way to control " \
              "its level of verbosity"

def __check_shogun(bottom_version, custom_versions=[]):
    """Check if version of shogun is high enough (or custom known) to
    be enabled in the testsuite

    :Parameters:
      bottom_version : int
        Bottom version which must be satisfied
      custom_versions : list of int
        Arbitrary list of versions which could got patched for
        a specific issue
    """
    import shogun.Classifier as __sc
    ver = __sc.Version_get_version_revision()
    if (ver in custom_versions) or (ver >= bottom_version):
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


def __check_atlas_family(family):
    # XXX I guess pylint will dislike it a lot
    from mvpa.atlases.warehouse import KNOWN_ATLAS_FAMILIES
    names, pathpattern = KNOWN_ATLAS_FAMILIES[family]
    filename = pathpattern % {'name':names[0]}
    if not os.path.exists(filename):
        raise ImportError, "Cannot find file %s for atlas family %s" \
              % (filename, family)
    pass


def __check_stablerdist():
    import scipy.stats
    import numpy as N
    ## Unfortunately 0.7.0 hasn't fixed the issue so no chance but to do
    ## a proper numerical test here
    try:
        scipy.stats.rdist(1.32, 0, 1).cdf(-1.0 + N.finfo(float).eps)
        # Actually previous test is insufficient for 0.6, so enabling
        # elderly test on top
        # ATM all known implementations which implement custom cdf for
        #     rdist are misbehaving, so there should be no _cdf
        if '_cdf' in scipy.stats.distributions.rdist_gen.__dict__.keys():
            raise ImportError, \
                  "scipy.stats carries misbehaving rdist distribution"
    except ZeroDivisionError:
        raise RuntimeError, "RDist in scipy is still unstable on the boundaries"


def __check_in_ipython():
    # figure out if ran within IPython
    if '__IPYTHON__' in globals()['__builtins__']:
        return
    raise RuntimeError, "Not running in IPython session"


def __check_matplotlib():
    """Check for presence of matplotlib and set backend if requested."""
    import matplotlib
    backend = cfg.get('matplotlib', 'backend')
    if backend:
        matplotlib.use(backend)

def __check_pylab():
    """Check if matplotlib is there and then pylab"""
    exists('matplotlib', raiseException=True)
    import pylab as P

def __check_pylab_plottable():
    """Simple check either we can plot anything using pylab.

    Primary use in unittests
    """
    try:
        exists('pylab', raiseException=True)
        import pylab as P
        fig = P.figure()
        P.plot([1,2], [1,2])
        P.close(fig)
    except:
        raise RuntimeError, "Cannot plot in pylab"
    return True


def __check_griddata():
    """griddata might be independent module or part of mlab
    """

    try:
        from griddata import griddata as __
        return True
    except ImportError:
        if __debug__:
            debug('EXT_', 'No python-griddata available')

    from matplotlib.mlab import griddata as __
    return True

# contains list of available (optional) external classifier extensions
_KNOWN = {'libsvm':'import mvpa.clfs.libsvmc._svm as __; x=__.convert2SVMNode',
          'libsvm verbosity control':'__check_libsvm_verbosity_control();',
          'nifti':'from nifti import NiftiImage as __',
          'nifti >= 0.20090205.1':
                'from nifti.clib import detachDataFromImage as __',
          'ctypes':'import ctypes as __',
          'shogun':'import shogun as __',
          'shogun.mpd': 'import shogun.Classifier as __; x=__.MPDSVM',
          'shogun.lightsvm': 'import shogun.Classifier as __; x=__.SVMLight',
          'shogun.svrlight': 'from shogun.Regression import SVRLight as __',
          'numpy': "__check_numpy()",
          'scipy': "__check_scipy()",
          'good scipy.stats.rdist': "__check_stablerdist()",
          'weave': "__check_weave()",
          'pywt': "import pywt as __",
          'pywt wp reconstruct': "__check_pywt(['wp reconstruct'])",
          'pywt wp reconstruct fixed': "__check_pywt(['wp reconstruct fixed'])",
          'rpy': "import rpy as __",
          'lars': "import rpy; rpy.r.library('lars')",
          'elasticnet': "import rpy; rpy.r.library('elasticnet')",
          'glmnet': "import rpy; rpy.r.library('glmnet')",
          'matplotlib': "__check_matplotlib()",
          'pylab': "__check_pylab()",
          'pylab plottable': "__check_pylab_plottable()",
          'openopt': "import scikits.openopt as __",
          'mdp': "import mdp as __",
          'mdp >= 2.4': "from mdp.nodes import LLENode as __",
          'sg_fixedcachesize': "__check_shogun(3043, [2456])",
           # 3318 corresponds to release 0.6.4
          'sg >= 0.6.4': "__check_shogun(3318)",
          'hcluster': "import hcluster as __",
          'griddata': "__check_griddata()",
          'cPickle': "import cPickle as __",
          'gzip': "import gzip as __",
          'lxml': "from lxml import objectify as __",
          'atlas_pymvpa': "__check_atlas_family('pymvpa')",
          'atlas_fsl': "__check_atlas_family('fsl')",
          'running ipython env': "__check_in_ipython()",
          }


def exists(dep, force=False, raiseException=False, issueWarning=None):
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
        Whether to raise RuntimeError if dependency is missing.
      issueWarning : string or None or True
        If string, warning with given message would be thrown.
        If True, standard message would be used for the warning
        text.
    """
    # if we are provided with a list of deps - go through all of them
    if isinstance(dep, list) or isinstance(dep, tuple):
        results = [ exists(dep_, force, raiseException) for dep_ in dep ]
        return bool(reduce(lambda x,y: x and y, results, True))

    # where to look in cfg
    cfgid = 'have ' + dep

    # prevent unnecessarry testing
    if cfg.has_option('externals', cfgid) \
       and not cfg.getboolean('externals', 'retest', default='no') \
       and not force:
        if __debug__:
            debug('EXT', "Skip retesting for '%s'." % dep)

        # check whether an exception should be raised, even though the external
        # was already tested previously
        if not cfg.getboolean('externals', cfgid) \
               and raiseException \
               and cfg.getboolean('externals', 'raise exception', True):
            raise RuntimeError, "Required external '%s' was not found" % dep
        return cfg.getboolean('externals', cfgid)


    # determine availability of external (non-cached)

    # default to 'not found'
    result = False

    if not _KNOWN.has_key(dep):
        raise ValueError, "%s is not a known dependency key." % (dep)
    else:
        # try and load the specific dependency
        if __debug__:
            debug('EXT', "Checking for the presence of %s" % dep)

        # Exceptions which are silently caught while running tests for externals
        _caught_exceptions = [ImportError, AttributeError, RuntimeError]

        # check whether RPy is involved and catch its excpetions as well.
        # however, try to determine whether this is really necessary, as
        # importing RPy also involved starting a full-blown R session, which can
        # take seconds and therefore is quite nasty...
        if dep.count('rpy') or _KNOWN[dep].count('rpy'):
            try:
                from rpy import RException
                _caught_exceptions += [RException]
            except:
                pass


        estr = ''
        try:
            exec _KNOWN[dep]
            result = True
        except tuple(_caught_exceptions), e:
            estr = ". Caught exception was: " + str(e)

        if __debug__:
            debug('EXT', "Presence of %s is%s verified%s" %
                  (dep, {True:'', False:' NOT'}[result], estr))

    if not result:
        if raiseException \
               and cfg.getboolean('externals', 'raise exception', True):
            raise RuntimeError, "Required external '%s' was not found" % dep
        if issueWarning is not None \
               and cfg.getboolean('externals', 'issue warning', True):
            if issueWarning is True:
                warning("Required external '%s' was not found" % dep)
            else:
                warning(issueWarning)


    # store result in config manager
    if not cfg.has_section('externals'):
        cfg.add_section('externals')
    if result:
        cfg.set('externals', 'have ' + dep, 'yes')
    else:
        cfg.set('externals', 'have ' + dep, 'no')

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
            warning("%s is not available." % dep)

    if __debug__:
        debug('EXT', 'The following optional externals are present: %s' \
                     % [ k[5:] for k in cfg.options('externals')
                            if k.startswith('have') \
                            and cfg.getboolean('externals', k) == True ])

