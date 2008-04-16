#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA misc modules"""

__docformat__ = 'restructuredtext'

from sys import stdout, stderr

from os import environ

from mvpa.misc.verbosity import LevelLogger, OnceLogger, Logger

#
# Setup verbose and debug outputs
#
class _SingletonType(type):
    """Simple singleton implementation adjusted from
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/412551
    """
    def __init__(self, *args):
        type.__init__(self, *args)
        self._instances = {}

    def __call__(self, sid, instance, *args):
        if not sid in self._instances:
            self._instances[sid] = instance
        return self._instances[sid]

class __Singleton:
    __metaclass__ = _SingletonType
    def __init__(self, *args):
        pass
    # Provided __call__ just to make silly pylint happy
    def __call__(self):
        raise NotImplementedError

verbose = __Singleton("verbose", LevelLogger(handlers=[stdout]))
errors = __Singleton("errors", LevelLogger(handlers=[stderr]))

# Levels for verbose
# 0 -- nothing besides errors
# 1 -- high level stuff -- top level operation or file operations
# 2 -- cmdline handling
# 3 --
# 4 -- computation/algorithm relevant thingies

# Lets check if environment can tell us smth
if environ.has_key('MVPA_VERBOSE'):
    verbose.level = int(environ['MVPA_VERBOSE'])


# Define Warning class so it is printed just once per each message
class WarningLog(OnceLogger):

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


    def __call__(self, msg, bt=None):
        import traceback
        if bt is None:
            bt = self.__btdefault
        tb = traceback.extract_stack(limit=2)
        msgid = `tb[-2]`                # take parent as the source of ID
        fullmsg = "WARNING: %s.\n\t(Please note: this warning is " % msg + \
                  "printed only once, but underlying problem might " + \
                  "occur many times.\n"
        if bt and self.__btlevels > 0:
            fullmsg += "Top-most backtrace:\n"
            fullmsg += reduce(lambda x, y: x + "\t%s:%d in %s where '%s'\n" % \
                              y,
                              traceback.extract_stack(limit=self.__btlevels),
                              "")

        OnceLogger.__call__(self, msgid, fullmsg, self.__maxcount)


if environ.has_key('MVPA_WARNINGS_BT'):
    warnings_btlevels = int(environ['MVPA_WARNINGS_BT'])
    warnings_bt = True
else:
    warnings_btlevels = 10
    warnings_bt = False

if environ.has_key('MVPA_WARNINGS_COUNT'):
    warnings_maxcount = int(environ['MVPA_WARNINGS_COUNT'])
else:
    warnings_maxcount = 1

warning = WarningLog(handlers={False: [stdout],
                               True: []}[environ.has_key('MVPA_NO_WARNINGS')],
                     btlevels=warnings_btlevels,
                     btdefault=warnings_bt,
                     maxcount=warnings_maxcount
                     )


if __debug__:
    from mvpa.misc.verbosity import DebugLogger
    # NOTE: all calls to debug must be preconditioned with
    # if __debug__:
    debug = __Singleton("debug", DebugLogger(handlers=[stderr]))

    # set some debugging matricses to report
    # debug.registerMetric('vmem')

    # List agreed sets for debug
    debug.register('DBG',  "Debug output itself")
    debug.register('EXT',  "External dependencies")
    debug.register('TEST', "Debug unittests")
    debug.register('DG',   "Data generators")
    debug.register('LAZY', "Miscelaneous 'lazy' evaluations")
    debug.register('LOOP', "Support's loop construct")
    debug.register('PLR',  "PLR call")
    debug.register('SLC',  "Searchlight call")
    debug.register('SA',   "Sensitivity analyzers call")
    debug.register('PSA',  "Perturbation analyzer call")
    debug.register('RFEC', "Recursive Feature Elimination call")
    debug.register('RFEC_', "Recursive Feature Elimination call (verbose)")
    debug.register('IFSC', "Incremental Feature Search call")
    debug.register('DS',   "*Dataset")
    debug.register('DS_',  "*Dataset (verbose)")
    debug.register('ST',   "State")
    debug.register('STV',  "State Variable")
    debug.register('STCOL', "State Collector")

    debug.register('CLF',    "Base Classifiers")
    debug.register('CLF_',   "Base Classifiers (verbose)")
    debug.register('CLF_TB',
        "Report traceback in train/predict. Helps to resolve WTF calls it")
    debug.register('CLFBST', "BoostClassifier")
    debug.register('CLFBIN', "BinaryClassifier")
    debug.register('CLFMC',  "MulticlassClassifier")
    debug.register('CLFSPL', "SplitClassifier")
    debug.register('CLFFS',  "FeatureSelectionClassifier")
    debug.register('CLFFS_', "FeatureSelectionClassifier (verbose)")

    debug.register('FS',     "FeatureSelections")
    debug.register('FS_',    "FeatureSelections (verbose)")
    debug.register('FSPL',   "FeatureSelectionPipeline")

    debug.register('SVM',    "SVM")
    debug.register('SVMLIB', "Internal libsvm output")

    debug.register('SMLR',    "SMLR")
    debug.register('SMLR_',   "SMLR verbose")

    debug.register('SG',  "PyMVPA SG wrapping")
    debug.register('SG_', "PyMVPA SG wrapping verbose")
    debug.register('SG__', "PyMVPA SG wrapping debug")
    debug.register('SG_SVM', "Internal shogun debug output for SVM itself")
    debug.register('SG_FEATURES', "Internal shogun debug output for features")
    debug.register('SG_LABELS', "Internal shogun debug output for labels")
    debug.register('SG_KERNELS', "Internal shogun debug output for kernels")
    debug.register('SG_PROGRESS', "Internal shogun progress bar during computation")

    debug.register('IOH',  "IO Helpers")
    debug.register('CM',   "Confusion matrix computation")
    debug.register('CROSSC',"Cross-validation call")
    debug.register('CERR', "Various ClassifierErrors")

    # Lets check if environment can tell us smth
    if environ.has_key('MVPA_DEBUG'):
        debug.setActiveFromString(environ['MVPA_DEBUG'])

    # Lets check if environment can tell us smth
    if environ.has_key('MVPA_DEBUG_METRICS'):
        debug.registerMetric(environ['MVPA_DEBUG_METRICS'].split(","))
