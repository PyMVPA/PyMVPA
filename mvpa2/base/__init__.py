# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Plumbing layer for PyMVPA

Module Organization
===================

mvpa2.base module contains various modules which are used through out
PyMVPA code, and are generic building blocks

:group Basic: externals, config, verbosity, dochelpers
"""

__docformat__ = 'restructuredtext'


import sys
import os
from mvpa2.base.config import ConfigManager
from mvpa2.base.verbosity import LevelLogger, OnceLogger


#
# Setup verbose and debug outputs
#
class _SingletonType(type):
    """Simple singleton implementation adjusted from
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/412551
    """
    def __init__(mcs, *args):
        type.__init__(mcs, *args)
        mcs._instances = {}

    def __call__(mcs, sid, instance, *args):
        if not sid in mcs._instances:
            mcs._instances[sid] = instance
        return mcs._instances[sid]


class __Singleton:
    """To ensure single instance of a class instantiation (object)

    """

    __metaclass__ = _SingletonType

    def __init__(self, *args):
        pass

    # Provided __call__ just to make silly pylint happy
    def __call__(self):
        raise NotImplementedError

#
# As the very first step: Setup configuration registry instance and
# read all configuration settings from files and env variables
#
_cfgfile = os.environ.get('MVPACONFIG', None)
if _cfgfile:
    # We have to provide a list
    _cfgfile = [_cfgfile]
cfg = __Singleton('cfg', ConfigManager(_cfgfile))

verbose = __Singleton("verbose", LevelLogger(
    handlers=cfg.get('verbose', 'output', default='stdout').split(',')))

# Not supported/explained/used by now since verbose(0, is to print errors
#error = __Singleton("error", LevelLogger(
#    handlers=environ.get('MVPA_ERROR_OUTPUT', 'stderr').split(',')))

# Levels for verbose
# 0 -- nothing besides errors
# 1 -- high level stuff -- top level operation or file operations
# 2 -- cmdline handling
# 3 --
# 4 -- computation/algorithm relevant thingies


# Helper for errors printing
def error(msg, critical=True):
    """Helper function to output errors in a consistent way.

    Parameters
    ----------
    msg : string
      Actual error message (will be prefixed with ERROR:)
    critical : bool
      If critical error -- exit with
    """
    verbose(0, "ERROR: " + msg)
    if critical:
        raise sys.exit(1)

# Lets check if environment can tell us smth
if cfg.has_option('general', 'verbose'):
    verbose.level = cfg.getint('general', 'verbose')


class WarningLog(OnceLogger):
    """Logging class of messsages to be printed just once per each message

    """

    def __init__(self, btlevels=10, btdefault=False,
                 maxcount=1, *args, **kwargs):
        """Define Warning logger.

        It is defined by
          btlevels : int
            how many levels of backtrack to print to give a hint on WTF
          btdefault : bool
            if to print backtrace for all warnings at all
          maxcount : int
            how many times to print each warning
        """
        OnceLogger.__init__(self, *args, **kwargs)
        self.__btlevels = btlevels
        self.__btdefault = btdefault
        self.__maxcount = maxcount
        self.__explanation_seen = False

    def __call__(self, msg, bt=None):
        import traceback
        if bt is None:
            bt = self.__btdefault
        tb = traceback.extract_stack(limit=2)
        msgid = repr(tb[-2])         # take parent as the source of ID
        fullmsg = "WARNING: %s" % msg
        if not self.__explanation_seen:
            self.__explanation_seen = True
            fullmsg += "\n * Please note: warnings are " + \
                "printed only once, but underlying problem might " + \
                "occur many times *"
        if bt and self.__btlevels > 0:
            fullmsg += "Top-most backtrace:\n"
            fullmsg += reduce(
                lambda x, y:
                x + "\t%s:%d in %s where '%s'\n" % y,
                traceback.extract_stack(limit=self.__btlevels),
                "")

        OnceLogger.__call__(self, msgid, fullmsg, self.__maxcount)

    def _set_max_count(self, value):
        """Set maxcount for the warning"""
        self.__maxcount = value

    maxcount = property(fget=lambda x: x.__maxcount, fset=_set_max_count)

# XXX what is 'bt'? Maybe more verbose name?
if cfg.has_option('warnings', 'bt'):
    warnings_btlevels = cfg.getint('warnings', 'bt')
    warnings_bt = True
else:
    warnings_btlevels = 10
    warnings_bt = False

if cfg.has_option('warnings', 'count'):
    warnings_maxcount = cfg.getint('warnings', 'count')
else:
    warnings_maxcount = 1

warning = WarningLog(
    handlers={
        False: cfg.get('warnings', 'output', default='stdout').split(','),
        True: []}[cfg.getboolean('warnings', 'suppress', default=False)],
    btlevels=warnings_btlevels,
    btdefault=warnings_bt,
    maxcount=warnings_maxcount
)


if __debug__:
    from mvpa2.base.verbosity import DebugLogger
    # NOTE: all calls to debug must be preconditioned with
    # if __debug__:

    debug = __Singleton("debug", DebugLogger(
        handlers=cfg.get('debug', 'output', default='stdout').split(',')))

    # set some debugging matricses to report
    # debug.register_metric('vmem')

    # List agreed sets for debug
    debug.register('PY', "No suppression of various warnings (numpy, scipy) etc.")
    debug.register('VERBOSE', "Verbose control debugging")
    debug.register('DBG', "Debug output itself")
    debug.register('STDOUT', "To decorate stdout with debug metrics")
    debug.register('DOCH', "Doc helpers")
    debug.register('INIT', "Just sequence of inits")
    debug.register('RANDOM', "Random number generation")
    debug.register('EXT', "External dependencies")
    debug.register('EXT_', "External dependencies (verbose)")
    debug.register('TEST', "Debug unittests")
    debug.register('MODULE_IN_REPR', "Include module path in __repr__")
    debug.register('ID_IN_REPR', "Include id in __repr__")
    debug.register('CMDLINE', "Handling of command line parameters")

    debug.register('NO', "Nodes")
    debug.register('DG', "Data generators")
    debug.register('LAZY', "Miscelaneous 'lazy' evaluations")
    debug.register('LOOP', "Support's loop construct")
    debug.register('PLR', "PLR call")
    debug.register('NBH', "Neighborhood estimations")
    debug.register('SLC', "Searchlight call")
    debug.register('SLC_', "Searchlight call (verbose)")
    debug.register('SVS', "Surface-based voxel selection (a.k.a. 'surfing')")
    debug.register('SA', "Sensitivity analyzers")
    debug.register('SOM', "Self-organizing-maps (SOM)")
    debug.register('IRELIEF', "Various I-RELIEFs")
    debug.register('SA_', "Sensitivity analyzers (verbose)")
    debug.register('PSA', "Perturbation analyzer call")
    debug.register('RFE', "Recursive Feature Elimination")
    debug.register('RFEC', "Recursive Feature Elimination call")
    debug.register('RFEC_', "Recursive Feature Elimination call (verbose)")
    debug.register('IFSC', "Incremental Feature Search call")
    debug.register('DS', "*Dataset")
    debug.register('DS_NIFTI', "NiftiDataset(s)")
    debug.register('DS_', "*Dataset (verbose)")
    debug.register('DS_ID', "ID Datasets")
    debug.register('DS_STATS', "Datasets statistics")
    debug.register('SPL', "*Splitter")
    debug.register('APERM', "AttributePermutator")

    debug.register('TRAN', "Transformers")
    debug.register('TRAN_', "Transformers (verbose)")

    # CHECKs
    debug.register('CHECK_DS_SELECT',
                   "Check in dataset.select() for sorted and unique indexes")
    debug.register('CHECK_DS_SORTED', "Check in datasets for sorted")
    debug.register('CHECK_IDS_SORTED',
                   "Check for ids being sorted in mappers")
    debug.register('CHECK_TRAINED',
                   "Checking in checking if clf was trained on given dataset")
    debug.register('CHECK_RETRAIN', "Checking in retraining/retesting")
    debug.register('CHECK_STABILITY', "Checking for numerical stability")
    debug.register('ENFORCE_CA_ENABLED', "Forcing all ca to be enabled")

    debug.register('MAP', "*Mapper")
    debug.register('MAP_', "*Mapper (verbose)")
    debug.register('FX', "FxMapper")
    debug.register('ZSCM', "ZScoreMapper")

    debug.register('COL', "Generic Collectable")
    debug.register('COL_RED', "__reduce__ of collectables")
    debug.register('UATTR', "Attributes with unique")
    debug.register('ST', "State")
    debug.register('STV', "State Variable")
    debug.register('COLR', "Collector for ca and classifier parameters")
    debug.register('ES', "Element selectors")

    debug.register('LRN', "Base learners")
    # TODO remove once everything is a learner
    debug.register('CLF', "Base Classifiers")
    debug.register('CLF_', "Base Classifiers (verbose)")
    #debug.register('CLF_TB',
    #    "Report traceback in train/predict. Helps to resolve WTF calls it")
    debug.register('CLFBST', "BoostClassifier")
    #debug.register('CLFBST_TB', "BoostClassifier traceback")
    debug.register('CLFPRX', "ProxyClassifier")
    debug.register('CLFBIN', "BinaryClassifier")
    debug.register('CLFTREE', "TreeClassifier")
    debug.register('CLFMC', "MulticlassClassifier")
    debug.register('CLFSPL', "SplitClassifier")
    debug.register('CLFSPL_', "SplitClassifier (verbose)")
    debug.register('CLFFS', "FeatureSelectionClassifier")
    debug.register('CLFFS_', "FeatureSelectionClassifier (verbose)")

    debug.register('STAT', "Statistics estimates")
    debug.register('STAT_', "Statistics estimates (verbose)")
    debug.register('STAT__', "Statistics estimates (very verbose)")
    debug.register('STATMC', "Progress in Monte-Carlo estimation")

    debug.register('FS', "FeatureSelections")
    debug.register('FS_', "FeatureSelections (verbose)")
    debug.register('FSPL', "FeatureSelectionPipeline")

    debug.register('KNN', "kNN")

    debug.register('SVM', "SVM")
    debug.register('SVM_', "SVM (verbose)")
    debug.register('LIBSVM', "Internal libsvm output")

    debug.register('SMLR', "SMLR")
    debug.register('SMLR_', "SMLR verbose")

    debug.register('LARS', "LARS")
    debug.register('LARS_', "LARS (verbose)")

    debug.register('ENET', "ENET")
    debug.register('ENET_', "ENET (verbose)")

    debug.register('GLMNET', "GLMNET")
    debug.register('GLMNET_', "GLMNET (verbose)")

    debug.register('GNB', "GNB - Gaussian Naive Bayes")

    debug.register('GDA', "GDA - Gaussian Discriminant Analyses")

    debug.register('GPR', "GPR")
    debug.register('GPR_WEIGHTS', "Track progress of GPRWeights computation")
    debug.register('KRN', "Kernels module (mvpa2.kernels)")
    debug.register('KRN_SG', "Shogun kernels module (mvpa2.kernels.sg)")
    debug.register('SAL', "Samples lookup (for cached kernels)")
    debug.register('MOD_SEL', "Model Selector (also makes openopt's iprint=0)")
    debug.register('OPENOPT', "OpenOpt toolbox verbose (iprint=1)")

    debug.register('SG', "PyMVPA SG wrapping")
    debug.register('SG_', "PyMVPA SG wrapping verbose")
    debug.register('SG__', "PyMVPA SG wrapping debug")
    debug.register('SG_GC', "For all entities enable highest level"
                            " (garbage collector)")
    debug.register('SG_LINENO', "Enable printing of the file:lineno"
                                " where SG_ERROR occurred.")
    debug.register('SG_SVM', "Internal shogun debug output for SVM itself")
    debug.register('SG_FEATURES', "Internal shogun debug output for features")
    debug.register('SG_LABELS', "Internal shogun debug output for labels")
    debug.register('SG_KERNELS', "Internal shogun debug output for kernels")
    debug.register('SG_PROGRESS',
                   "Internal shogun progress bar during computation")

    debug.register('IOH', "IO Helpers")
    debug.register('NIML', "NeuroImaging Markup Language")
    debug.register('HDF5', "HDF5 IO")
    debug.register('CM', "Confusion matrix computation")
    debug.register('ROC', "ROC analysis")
    debug.register('REPM', "Repeated measure (e.g. super-class of CrossValidation)")
    debug.register('CERR', "Various ClassifierErrors")

    debug.register('HPAL',   "Hyperalignment")
    debug.register('HPAL_',  "Hyperalignment (verbose)")
    debug.register('SHPAL',  "Searchlight Hyperalignment")
    debug.register('GCTHR', "Group cluster threshold")
    debug.register('ATL', "Atlases")
    debug.register('ATL_', "Atlases (verbose)")
    debug.register('ATL__', "Atlases (very verbose)")

    debug.register('PLLB', "plot_lightbox")

    debug.register('REP', "Reports")
    debug.register('REP_', "Reports (verbose)")

    debug.register('SUITE', "Import of mvpa2.suite")

    debug.register('ATTRREFER', "Debugging of top-level attribute referencing, "
                   "needed for current refactoring carried out in tent/flexds")

    debug.register('BM', "Benchmark")

    # Lets check if environment can tell us smth
    if cfg.has_option('general', 'debug'):
        debug.set_active_from_string(cfg.get('general', 'debug'))

    # Lets check if environment can tell us smth
    if cfg.has_option('debug', 'metrics'):
        debug.register_metric(cfg.get('debug', 'metrics').split(","))

    if 'STDOUT' in debug.active:
        # Lets decorate sys.stdout to possibly figure out what brings
        # the noise

        class _pymvpa_stdout_debug(object):
            """

            Kudos to CODEHEAD
            http://codingrecipes.com/decorating-pythons-sysstdout
            for this design pattern
            """

            def __init__(self, sys):
                self.stdout = sys.stdout
                sys.stdout = self
                self._inhere = False
                self._newline = True

            def write(self, txt):
                try:
                    if not self._inhere and self._newline:
                        self._inhere = True
                        debug('STDOUT', "", lf=False, cr=False)
                    self.stdout.write(txt)
                    self._newline = txt.endswith('\n')
                finally:
                    self._inhere = False

            def flush(self):
                self.stdout.flush()

            def isatty(self):
                return False

        _out = _pymvpa_stdout_debug(sys)

else:  # if not __debug__

    # this debugger function does absolutely nothing.
    # It avoids the need of using 'if __debug__' for debug(...) calls.

    from mvpa2.base.verbosity import BlackHoleLogger

    debug = __Singleton("debug", BlackHoleLogger())

if __debug__:
    debug('INIT', 'mvpa2.base end')
