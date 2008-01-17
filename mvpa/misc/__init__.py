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

from verbosity import LevelLogger, OnceLogger

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
    __metaclass__=_SingletonType
    def __init__(self, *args): pass
    # Provided __call__ just to make silly pylint happy
    def __call__(self): raise NotImplementedError

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

    def __init__(self, btlevels=4, *args, **kwargs):
        """Define Warning logger.

        It is defined by
          `btlevels`, int: how many levels of backtrack to print to
                           give a hint on WTF
        """
        OnceLogger.__init__(self, *args, **kwargs)
        self.__btlevels = btlevels

    def __call__(self, msg):
        import traceback
        tb = traceback.extract_stack(limit=2)
        msgid = `tb[-2]`                # take parent as the source of ID
        fullmsg = "WARNING: %s.\n\t(Please note: this warning is " % msg + \
                  "printed only once, but underlying problem might " + \
                  "occur many times.\n"
        if self.__btlevels > 0:
            fullmsg += "Top-most backtrace:\n"
            fullmsg += reduce(lambda x, y: x + "\t%s:%d in %s where '%s'\n" % \
                              y,
                              traceback.extract_stack(limit=self.__btlevels),
                              "")

        OnceLogger.__call__(self, msgid, fullmsg)

warning = WarningLog(handlers=[stdout])


if __debug__:
    from verbosity import DebugLogger
    # NOTE: all calls to debug must be preconditioned with
    # if __debug__:
    debug = __Singleton("debug", DebugLogger(handlers=[stderr]))

    # set some debugging matricses to report
    # debug.registerMetric('vmem')

    # List agreed sets for debug
    debug.register('DBG',  "Debug output itself")
    debug.register('LAZY', "Miscelaneous 'lazy' evaluations")
    debug.register('PLF',  "PLF call")
    debug.register('SLC',  "Searchlight call")
    debug.register('SA',   "Sensitivity analyzers call")
    debug.register('PSA',  "Perturbation analyzer call")
    debug.register('RFEC', "Recursive Feature Elimination call")
    debug.register('IFSC', "Incremental Feature Search call")
    debug.register('DS',   "*Dataset")
    debug.register('ST',   "State")
    debug.register('STV',  "State Variable")
    debug.register('STCOL', "State Collector")

    debug.register('CLF',    "Base Classifiers")
    debug.register('CLF_TB',
        "Report traceback in train/predict. Helps to resolve WTF calls it")
    debug.register('CLFBST', "BoostClassifier")
    debug.register('CLFBIN', "BinaryClassifier")
    debug.register('CLFMC',  "MulticlassClassifier")
    debug.register('CLFSPL', "SplitClassifier")
    debug.register('CLFFS',  "FeatureSelectionClassifier")

    debug.register('FSPL',  "FeatureSelectionPipeline")

    debug.register('SVM',    "SVM")
    debug.register('SVMLIB', "Internal libsvm verbose output")

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
