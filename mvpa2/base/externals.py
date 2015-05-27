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
import numpy as np                      # NumPy is required anyways

from mvpa2.base import warning
from mvpa2 import cfg
from mvpa2.misc.support import SmartVersion

if __debug__:
    from mvpa2.base import debug

class _VersionsChecker(dict):
    """Helper class to check the versions of the available externals
    """

    def __init__(self, *args, **kwargs):
        self._KNOWN = {}
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            if key in self._KNOWN:
                # run registered procedure to obtain versions
                self._KNOWN[key]()
            else:
                # just check for presence -- that function might set
                # the version information
                exists(key, force=True, raise_=True)
        return super(_VersionsChecker, self).__getitem__(key)

versions = _VersionsChecker()
"""Versions of available externals, as tuples
"""

def __assign_numpy_version():
    """Check if numpy is present (it must be) an if it is -- store its version
    """
    import numpy as np
    versions['numpy'] = SmartVersion(np.__version__)

def __check_numpy_correct_unique():
    """ndarray.unique fails to operate on heterogeneous object ndarrays
    See http://projects.scipy.org/numpy/ticket/2188
    """
    import numpy as np
    try:
        _ = np.unique(np.array([1, None, "str"]))
    except TypeError, e:
        raise RuntimeError("numpy.unique thrown %s" % e)

def __assign_scipy_version():
    # To don't allow any crappy warning to sneak in
    import warnings
    warnings.simplefilter('ignore', DeprecationWarning)
    try:
        import scipy as sp
    except:
        warnings.simplefilter('default', DeprecationWarning)
        raise
    warnings.simplefilter('default', DeprecationWarning)
    versions['scipy'] = SmartVersion(sp.__version__)

def __check_scipy():
    """Check if scipy is present an if it is -- store its version
    """
    exists('numpy', raise_=True)
    __assign_numpy_version()
    __assign_scipy_version()
    import scipy as sp

def _suppress_scipy_warnings():
    # Infiltrate warnings if necessary
    numpy_ver = versions['numpy']
    scipy_ver = versions['scipy']
    # There is way too much deprecation warnings spit out onto the
    # user. Lets assume that they should be fixed by scipy 0.7.0 time
    if scipy_ver >= "0.6.0" and scipy_ver < "0.7.0" \
        and numpy_ver > "1.1.0":
        import warnings
        if not __debug__ or (__debug__ and not 'PY' in debug.active):
            if __debug__:
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


def __assign_mdp_version():
    """Check if mdp is present (it must be) an if it is -- store its version
    """
    import mdp
    ver = mdp.__version__
    if SmartVersion(ver) == "2.5" and not hasattr(mdp.nodes, 'IdentityNode'):
        # Thanks to Yarik's shipment of svn snapshots into Debian we
        # can't be sure if that was already released version, since
        # mdp guys didn't use -dev suffix
        ver += '-dev'
    versions['mdp'] = SmartVersion(ver)

def __assign_nibabel_version():
    try:
        import nibabel
    except Exception, e:
        # FloatingError is defined in the same module which precludes
        # its specific except
        e_str = str(e)
        if "We had not expected long double type <type 'numpy.float128'>" in e_str:
            warning("Must be running under valgrind?  Available nibabel experiences "
                    "difficulty with float128 upon import and fails to work, thus is "
                    "report as N/A")
            raise ImportError("Fail to import nibabel due to %s" % e_str)
        raise
    versions['nibabel'] = SmartVersion(nibabel.__version__)

def __check_pywt(features=None):
    """Check for available functionality within pywt

    Parameters
    ----------
    features : list of str
      List of known features to check such as 'wp reconstruct',
      'wp reconstruct fixed'
    """
    import pywt
    import numpy as np
    data = np.array([ 0.57316901,  0.65292526,  0.75266733,  0.67020084,  0.46505364,
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
        if np.linalg.norm(rec[:len(data)] - data) > 1e-3:
            raise ImportError, \
                  "Failed to reconstruct WP correctly"
    return True


def __check_libsvm_verbosity_control():
    """Check for available verbose control functionality
    """
    import mvpa2.clfs.libsvmc._svmc as _svmc
    try:
        _svmc.svm_set_verbosity(0)
    except:
        raise ImportError, "Provided version of libsvm has no way to control " \
              "its level of verbosity"

def __assign_shogun_version():
    """Assign shogun versions
    """
    if 'shogun' in versions:
        return
    import shogun.Classifier as __sc
    versions['shogun:rev'] = __sc.Version_get_version_revision()
    ver = __sc.Version_get_version_release().lstrip('v')
    versions['shogun:full'] = ver
    if '_' in ver:
        ver = ver[:ver.index('_')]
    versions['shogun'] = ver


def __check_shogun(bottom_version, custom_versions=[]):
    """Check if version of shogun is high enough (or custom known) to
    be enabled in the testsuite

    Parameters
    ----------
    bottom_version : int
      Bottom version which must be satisfied
    custom_versions : list of int
      Arbitrary list of versions which could got patched for
      a specific issue
    """
    import shogun.Classifier as __sc
    ver = __sc.Version_get_version_revision()
    __assign_shogun_version()
    if (ver in custom_versions) or (ver >= bottom_version):
        return True
    else:
        raise ImportError, 'Version %s is smaller than needed %s' % \
              (ver, bottom_version)

def __check_nipy_neurospin():
    from nipy.neurospin.utils import emp_nul

def __assign_skl_version():
    try:
        import sklearn as skl
    except ImportError:
        # Let's try older space
        import scikits.learn as skl
        if skl.__doc__ is None or skl.__doc__.strip() == "":
            raise ImportError("Verify your installation of scikits.learn. "
                              "Its docstring is empty -- could be that only -lib "
                              "was installed without the native Python modules")
    versions['skl'] = SmartVersion(skl.__version__)

def __check_weave():
    """Apparently presence of scipy is not sufficient since some
    versions experience problems. E.g. in Sep,Oct 2008 lenny's weave
    failed to work. May be some other converter could work (? See
    http://lists.debian.org/debian-devel/2008/08/msg00730.html for a
    similar report.

    Following simple snippet checks compilation of the basic code using
    weave
    """
    try:
        from scipy import weave
    except OSError, e:
        raise ImportError(
            "Weave cannot be used due to failure to import because of %s"
            % e)
    from scipy.weave import converters, build_tools
    import numpy as np
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
        data = np.array([1,2,3])
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
    from mvpa2.atlases.warehouse import KNOWN_ATLAS_FAMILIES
    names, pathpattern = KNOWN_ATLAS_FAMILIES[family]
    filename = pathpattern % {'name':names[0]}
    if not os.path.exists(filename):
        raise ImportError, "Cannot find file %s for atlas family %s" \
              % (filename, family)
    pass


def __check_stablerdist():
    import scipy.stats
    import numpy as np
    ## Unfortunately 0.7.0 hasn't fixed the issue so no chance but to do
    ## a proper numerical test here
    try:
        scipy.stats.rdist(1.32, 0, 1).cdf(-1.0 + np.finfo(float).eps)
        # Actually previous test is insufficient for 0.6, so enabling
        # elderly test on top
        # ATM all known implementations which implement custom cdf for
        #     rdist are misbehaving, so there should be no _cdf
        distributions = scipy.stats.distributions
        if 'rdist_gen' in dir(distributions) \
            and ('_cdf' in distributions.rdist_gen.__dict__.keys()):
            raise ImportError, \
                  "scipy.stats carries misbehaving rdist distribution"
    except ZeroDivisionError:
        raise RuntimeError, "RDist in scipy is still unstable on the boundaries"


def __check_rv_discrete_ppf():
    """Unfortunately 0.6.0-12 of scipy pukes on simple ppf
    """
    import scipy.stats
    try:
        bdist = scipy.stats.binom(100, 0.5)
        bdist.ppf(0.9)
    except TypeError:
        raise RuntimeError, "pmf is broken in discrete dists of scipy.stats"

def __check_rv_continuous_reduce_func():
    """Unfortunately scipy 0.10.1 pukes when fitting with two params fixed
    """
    import scipy.stats as ss
    try:
        ss.t.fit(np.arange(6), floc=0.0, fscale=1.)
    except IndexError, e:
        raise RuntimeError("rv_continuous.fit can't candle 2 fixed params")

def __check_in_ipython():
    # figure out if ran within IPython
    if '__IPYTHON__' in globals()['__builtins__']:
        return
    raise RuntimeError, "Not running in IPython session"

def __assign_ipython_version():
    ipy_version = None
    try:
        # Development post 0.11 version finally carries
        # conventional one
        import IPython
        ipy_version = IPython.__version__
    except:
        try:
            from IPython import Release
            ipy_version = Release.version
        except:
            pass
        pass
    versions['ipython'] = SmartVersion(ipy_version)

def __check_openopt():
    m = None
    try:
        import openopt as m
    except ImportError:
        import scikits.openopt as m
    versions['openopt'] = m.__version__
    return True


def _set_matplotlib_backend():
    """Check if we have custom backend to set and it is different
    from current one
    """
    backend = cfg.get('matplotlib', 'backend')
    if backend:
        import matplotlib as mpl
        mpl_backend = mpl.get_backend().lower()
        if mpl_backend != backend.lower():
            if __debug__:
                debug('EXT_', "Trying to set matplotlib backend to %s" % backend)
            mpl.use(backend)
            import warnings
            # And disable useless warning from matplotlib in the future
            warnings.filterwarnings(
                'ignore', 'This call to matplotlib.use() has no effect.*',
                UserWarning)
        elif __debug__:
            debug('EXT_',
                  "Not trying to set matplotlib backend to %s since it was "
                  "already set" % backend)


def __assign_matplotlib_version():
    """Check for matplotlib version and set backend if requested."""
    import matplotlib
    versions['matplotlib'] = SmartVersion(matplotlib.__version__)
    _set_matplotlib_backend()

def __check_pylab():
    """Check if matplotlib is there and then pylab"""
    exists('matplotlib', raise_='always')
    import pylab as pl

def __check_pylab_plottable():
    """Simple check either we can plot anything using pylab.

    Primary use in unittests
    """
    try:
        exists('pylab', raise_='always')
        import pylab as pl
        fig = pl.figure()
        pl.plot([1,2], [1,2])
        pl.close(fig)
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


def __check_reportlab():
    import reportlab as rl
    versions['reportlab'] = SmartVersion(rl.Version)

def __check(name, a='__version__'):
    exec "import %s" % name
    try:
        exec "v = %s.%s" % (name, a)
        # it might be lxml.etree, so take only first module
        versions[name.split('.')[0]] = SmartVersion(v)
    except Exception, e:
        # we can't assign version but it is there
        if __debug__:
            debug('EXT', 'Failed to acquire a version of %(name)s: %(e)s'
                  % locals())
        pass
    return True

def __check_h5py():
    __check('h5py', 'version.version')
    import h5py
    versions['hdf5'] = SmartVersion(h5py.version.hdf5_version)

def __check_rpy():
    """Check either rpy is available and also set it for the sane execution
    """
    #import rpy_options
    #rpy_options.set_options(VERBOSE=False, SETUP_READ_CONSOLE=False) # SETUP_WRITE_CONSOLE=False)
    #rpy_options.set_options(VERBOSE=False, SETUP_WRITE_CONSOLE=False) # SETUP_WRITE_CONSOLE=False)
    #    if not cfg.get('rpy', 'read_console', default=False):
    #        print "no read"
    #        rpy_options.set_options(SETUP_READ_CONSOLE=False)
    #    if not cfg.get('rpy', 'write_console', default=False):
    #        print "no write"
    #        rpy_options.set_options(SETUP_WRITE_CONSOLE=False)
    import rpy
    if not cfg.getboolean('rpy', 'interactive', default=True) \
           and (rpy.get_rpy_input() is rpy.rpy_io.rpy_input):
        if __debug__:
            debug('EXT_', "RPy: providing dummy callback for input to return '1'")
        def input1(*args): return "1"      # which is "1: abort (with core dump, if enabled)"
        rpy.set_rpy_input(input1)

def _R_library(libname):
    import rpy2.robjects as ro

    try:
        if not tuple(ro.r(
            "suppressMessages(suppressWarnings(require(%r, quiet=TRUE)))"
            % libname))[0]:
            raise ImportError("It seems that R cannot load library %r"
                              % libname)
    except Exception, e:
        raise ImportError("Failed to load R library %r due to %s"
                          % (libname, e))

def __check_rpy2():
    """Check either rpy2 is available and also set it for the sane execution
    """
    import rpy2
    versions['rpy2'] = SmartVersion(rpy2.__version__)

    import rpy2.robjects
    r = rpy2.robjects.r
    r.options(warn=cfg.get_as_dtype('rpy', 'warn', dtype=int, default=-1))

    # To shut R up while it is importing libraries to do not ruin out
    # doctests
    r.library = _R_library

def __check_liblapack_so():
    """Check either we could load liblapack.so library via ctypes
    """
    from ctypes import cdll
    try:
        lapacklib = cdll.LoadLibrary('liblapack.so')
    except OSError, e:
        # reraise with exception type we catch/handle while testing externals
        raise RuntimeError("Failed to import liblapack.so: %s" % e)

# contains list of available (optional) external classifier extensions
_KNOWN = {'libsvm':'import mvpa2.clfs.libsvmc._svm as __; x=__.seq_to_svm_node',
          'libsvm verbosity control':'__check_libsvm_verbosity_control();',
          'nibabel':'__assign_nibabel_version()',
          'ctypes':'__check("ctypes")',
          'liblapack.so': "__check_liblapack_so()",
          'shogun':'__assign_shogun_version()',
          'shogun.krr': '__assign_shogun_version(); import shogun.Regression as __; x=__.KRR',
          'shogun.mpd': '__assign_shogun_version(); import shogun.Classifier as __; x=__.MPDSVM',
          'shogun.lightsvm': '__assign_shogun_version(); import shogun.Classifier as __; x=__.SVMLight',
          'shogun.svmocas': '__assign_shogun_version(); import shogun.Classifier as __; x=__.SVMOcas',
          'shogun.svrlight': '__assign_shogun_version(); from shogun.Regression import SVRLight as __',
          'numpy': "__assign_numpy_version()",
          'numpy_correct_unique': "__check_numpy_correct_unique()",
          'numpydoc': "import numpydoc",
          'scipy': "__check_scipy()",
          'good scipy.stats.rdist': "__check_stablerdist()",
          'good scipy.stats.rv_discrete.ppf': "__check_rv_discrete_ppf()",
          'good scipy.stats.rv_continuous._reduce_func(floc,fscale)': "__check_rv_continuous_reduce_func()",
          'weave': "__check_weave()",
          'pywt': "import pywt as __",
          'pywt wp reconstruct': "__check_pywt(['wp reconstruct'])",
          'pywt wp reconstruct fixed': "__check_pywt(['wp reconstruct fixed'])",
          #'rpy': "__check_rpy()",
          'rpy2': "__check_rpy2()",
          'lars': "exists('rpy2', raise_='always');" \
                  "import rpy2.robjects; rpy2.robjects.r.library('lars')",
          'mass': "exists('rpy2', raise_='always');" \
                  "import rpy2.robjects; rpy2.robjects.r.library('MASS')",
          'elasticnet': "exists('rpy2', raise_='always'); "\
                  "import rpy2.robjects; rpy2.robjects.r.library('elasticnet')",
          'glmnet': "exists('rpy2', raise_='always'); " \
                  "import rpy2.robjects; rpy2.robjects.r.library('glmnet')",
          'cran-energy': "exists('rpy2', raise_='always'); " \
                  "import rpy2.robjects; rpy2.robjects.r.library('energy')",
          'matplotlib': "__assign_matplotlib_version()",
          'pylab': "__check_pylab()",
          'pylab plottable': "__check_pylab_plottable()",
          'openopt': "__check_openopt()",
          'skl': "__assign_skl_version()",
          'mdp': "__assign_mdp_version()",
          'mdp ge 2.4': "from mdp.nodes import LLENode as __",
          'sg_fixedcachesize': "__check_shogun(3043, [2456])",
           # 3318 corresponds to release 0.6.4
          'sg ge 0.6.4': "__check_shogun(3318)",
           # 3377 corresponds to release 0.6.5
          'sg ge 0.6.5': "__check_shogun(3377)",
          'hcluster': "import hcluster as __",
          'griddata': "__check_griddata()",
          'cPickle': "import cPickle as __",
          'gzip': "import gzip as __",
          'lxml': "__check('lxml.etree', '__version__');"
                  "from lxml import objectify as __",
          'atlas_pymvpa': "__check_atlas_family('pymvpa')",
          'atlas_fsl': "__check_atlas_family('fsl')",
          'ipython': "__assign_ipython_version()",
          'running ipython env': "__check_in_ipython()",
          'reportlab': "__check('reportlab', 'Version')",
          'nose': "import nose as __",
          'pprocess': "__check('pprocess')",
          'pywt': "__check('pywt')",
          'h5py': "__check_h5py()",
          'hdf5': "__check_h5py()",
          'nipy': "__check('nipy')",
          'nipy.neurospin': "__check_nipy_neurospin()",
          'statsmodels': 'import statsmodels.api as __',
          'mock': "__check('mock')",
          'joblib': "import joblib as __",
          }


def exists(dep, force=False, raise_=False, issueWarning=None,
           exception=RuntimeError):
    """
    Test whether a known dependency is installed on the system.

    This method allows us to test for individual dependencies without
    testing all known dependencies. It also ensures that we only test
    for a dependency once.

    Parameters
    ----------
    dep : string or list of string
      The dependency key(s) to test.
    force : boolean
      Whether to force the test even if it has already been
      performed.
    raise_ : boolean, str
      Whether to raise an exception if dependency is missing.
      If True, it is still conditioned on the global setting
      MVPA_EXTERNALS_RAISE_EXCEPTION, while would raise exception
      if missing despite the configuration if 'always'.
    issueWarning : string or None or True
      If string, warning with given message would be thrown.
      If True, standard message would be used for the warning
      text.
    exception : exception, optional
      What exception to raise.  Defaults to RuntimeError
    """
    # if we are provided with a list of deps - go through all of them
    if isinstance(dep, list) or isinstance(dep, tuple):
        results = [ exists(dep_, force, raise_) for dep_ in dep ]
        return bool(reduce(lambda x,y: x and y, results, True))

    # where to look in cfg
    cfgid = 'have ' + dep

    # pre-handle raise_ according to the global settings and local argument
    if isinstance(raise_, str):
        if raise_.lower() == 'always':
            raise_ = True
        else:
            raise ValueError("Unknown value of raise_=%s. "
                             "Must be bool or 'always'" % raise_)
    else: # must be bool conditioned on the global settings
        raise_ = raise_ \
                and cfg.getboolean('externals', 'raise exception', True)

    # prevent unnecessarry testing
    if cfg.has_option('externals', cfgid) \
       and not cfg.getboolean('externals', 'retest', default='no') \
       and not force:
        if __debug__:
            debug('EXT', "Skip retesting for '%s'." % dep)

        # check whether an exception should be raised, even though the external
        # was already tested previously
        if not cfg.getboolean('externals', cfgid) and raise_:
            raise exception, "Required external '%s' was not found" % dep
        return cfg.getboolean('externals', cfgid)


    # determine availability of external (non-cached)

    # default to 'not found'
    result = False

    if dep not in _KNOWN:
        raise ValueError, "%r is not a known dependency key." % (dep,)
    else:
        # try and load the specific dependency
        if __debug__:
            debug('EXT', "Checking for the presence of %s" % dep)

        # Exceptions which are silently caught while running tests for externals
        _caught_exceptions = [ImportError, AttributeError, RuntimeError]

        try:
            # Suppress NumPy warnings while testing for externals
            olderr = np.seterr(all="ignore")

            estr = ''
            try:
                exec _KNOWN[dep]
                result = True
            except tuple(_caught_exceptions), e:
                estr = ". Caught exception was: " + str(e)
            except Exception, e:
                # Add known ones by their names so we don't need to
                # actually import anything manually to get those classes
                if e.__class__.__name__ in ['RPy_Exception', 'RRuntimeError',
                                            'RPy_RException']:
                    _caught_exceptions += [e.__class__]
                    estr = ". Caught exception was: " + str(e)
                else:
                    raise
        finally:
            # And restore warnings
            np.seterr(**olderr)

        if __debug__:
            debug('EXT', "Presence of %s is%s verified%s" %
                  (dep, {True:'', False:' NOT'}[result], estr))

    if not result:
        if raise_:
            raise exception, "Required external '%s' was not found" % dep
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

# Bind functions for some versions checkings
versions._KNOWN.update({
    'shogun' : __assign_shogun_version,
    'shogun:rev' : __assign_shogun_version,
    'shogun:full' : __assign_shogun_version,
    })


##REF: Name was automagically refactored
def check_all_dependencies(force=False, verbosity=1):
    """
    Test for all known dependencies.

    Parameters
    ----------
    force : boolean
      Whether to force the test even if it has already been
      performed.

    """
    # loop over all known dependencies
    for dep in _KNOWN:
        if not exists(dep, force):
            if verbosity:
                warning("%s is not available." % dep)

    if __debug__:
        debug('EXT', 'The following optional externals are present: %s' \
                     % [ k[5:] for k in cfg.options('externals')
                            if k.startswith('have') \
                            and cfg.getboolean('externals', k) == True ])
