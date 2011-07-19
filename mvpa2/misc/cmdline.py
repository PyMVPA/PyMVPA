# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common functions and options definitions for command line

Conventions:
Every option (instance of optparse.Option) has prefix "opt". Lists of options
has prefix opts (e.g. `opts.common`).

Option name should be camelbacked version of .dest for the option.
"""

__docformat__ = 'restructuredtext'

import mvpa2

# TODO? all options (opt*) might migrate to respective module? discuss
from optparse import OptionParser, Option, OptionGroup, OptionConflictError

# needed for verboseCallback
from mvpa2.base import verbose, externals

class Options(object):
    """Just a convenience placeholder for all available options
    """
    pass

class OptionGroups(object):
    """Group creation is delayed until instance is requested.

    This allows to overcome the problem of poluting handled cmdline options
    """

    def __init__(self, parser):
        """
        Parameters
        ----------
        parser : OptionParser
          To which parser to add groups
        """
        self._d = {}
        self._parser = parser

    def add(self, name, l, doc):
        self._d[name] = (doc, l)

    ##REF: Name was automagically refactored
    def _get_group(self, name):
        try:
            doc, l = self._d[name]
        except KeyError:
            raise ValueError, "No group with name %s" % name
        opts = OptionGroup(self._parser, doc)
        try:
            opts.add_options(l)
        except OptionConflictError:
            print "Problem addition options to the group '%s'. Most probably" \
                  " the option was independently added already." % name
            raise
        return opts

    def __getattribute__(self, index):
        if index[0] == '_':
            return object.__getattribute__(self, index)
        if self._d.has_key(index):
            return self._get_group(index)
        return object.__getattribute__(self, index)


# TODO: try to make groups definition somewhat lazy, since now
# whenever a group is created, those parameters are already known by
# parser, although might not be listed in the list of used and not by
# --help. But their specification on cmdline doesn't lead to
# error/help msg.
#
# Conflict hanlder to resolve situation that we have the same option added
# to some group and also available 'freely'
#
# set default version string, otherwise '--version' option is not enabled
# can be overwritten later on by assigning to `parser.version`
parser = OptionParser(version="%prog",
                      add_help_option=False,
                      conflict_handler="error")


opt = Options()
opts = OptionGroups(parser)

#
# Verbosity options
#
##REF: Name was automagically refactored
def _verbose_callback(option, optstr, value, parser):
    """Callback for -v|--verbose cmdline option
    """
    if __debug__:
        debug("CMDLINE", "Setting verbose.level to %s" % str(value))
    verbose.level = value
    optstr = optstr                     # pylint shut up
    setattr(parser.values, option.dest, value)

opt.help = \
    Option("-h", "--help", "--sos",
           action="help",
           help="Show this help message and exit")

opt.verbose = \
    Option("-v", "--verbose", "--verbosity",
           action="callback", callback=_verbose_callback, nargs=1,
           type="int", dest="verbose", default=0,
           help="Verbosity level of output")
"""Pre-cooked `optparse`'s option to specify verbose level"""

commonopts_list = [opt.verbose, opt.help]

if __debug__:
    from mvpa2.base import debug

    ##REF: Name was automagically refactored
    def _debug_callback(option, optstr, value, parser):
        """Callback for -d|--debug cmdline option
        """
        if value == "list":
            print "Registered debug IDs:"
            keys = debug.registered.keys()
            keys.sort()
            for k in keys:
                print "%-7s: %s" % (k, debug.registered[k])
            print "Use ALL: to enable all of the debug IDs listed above."
            print "Use python regular expressions to select group. CLF.* will" \
              " enable all debug entries starting with CLF (e.g. CLFBIN, CLFMC)"
            raise SystemExit, 0

        optstr = optstr                     # pylint shut up
        debug.set_active_from_string(value)

        setattr(parser.values, option.dest, value)


    optDebug = Option("-d", "--debug",
                      action="callback", callback=_debug_callback,
                      nargs=1,
                      type="string", dest="debug", default="",
                      help="Debug entries to report. " +
                      "Run with '-d list' to get a list of " +
                      "registered entries")

    commonopts_list.append(optDebug)

opts.add("common", commonopts_list, "Common generic options")

#
# Classifiers options
#
opt.clf = \
    Option("--clf",
           type="choice", dest="clf",
           choices=['knn', 'svm', 'ridge', 'gpr', 'smlr'], default='svm',
           help="Type of classifier to be used. Default: svm")

opt.radius = \
    Option("-r", "--radius",
           action="store", type="float", dest="radius",
           default=5.0,
           help="Radius to be used (eg for the searchlight). Default: 5.0")


opt.knearestdegree = \
    Option("-k", "--k-nearest",
           action="store", type="int", dest="knearestdegree", default=3,
           help="Degree of k-nearest classifier. Default: 3")

opts.add('KNN', [opt.radius, opt.knearestdegree], "Specification of kNN")


opt.svm_C = \
    Option("-C", "--svm-C",
           action="store", type="float", dest="svm_C", default=1.0,
           help="C parameter for soft-margin C-SVM classification. " \
                "Default: 1.0")

opt.svm_nu = \
    Option("--nu", "--svm-nu",
           action="store", type="float", dest="svm_nu", default=0.1,
           help="nu parameter for soft-margin nu-SVM classification. " \
                "Default: 0.1")

opt.svm_gamma = \
    Option("--gamma", "--svm-gamma",
           action="store", type="float", dest="svm_gamma", default=1.0,
           help="gamma parameter for Gaussian kernel of RBF SVM. " \
                "Default: 1.0")

opts.add('SVM', [opt.svm_nu, opt.svm_C, opt.svm_gamma], "SVM specification")

opt.do_sweep = \
             Option("--sweep",
                    action="store_true", dest="do_sweep",
                    default=False,
                    help="Sweep through various classifiers")

# Crossvalidation options

opt.crossfolddegree = \
    Option("-c", "--crossfold",
           action="store", type="int", dest="crossfolddegree", default=1,
           help="Degree of N-fold crossfold. Default: 1")

opts.add('general', [opt.crossfolddegree], "Generalization estimates")


# preprocess options

opt.zscore = \
    Option("--zscore",
           action="store_true", dest="zscore", default=0,
           help="Enable zscoring of dataset samples. Default: Off")

opt.tr = \
    Option("--tr",
           action="store", dest="tr", default=2.0, type='float',
           help="fMRI volume repetition time. Default: 2.0")

opt.detrend = \
    Option("--detrend",
           action="store_true", dest="detrend", default=0,
           help="Do linear detrending. Default: Off")

opts.add('preproc', [opt.zscore, opt.tr, opt.detrend], "Preprocessing options")


# Wavelets options
if externals.exists('pywt'):
    import pywt
    ##REF: Name was automagically refactored
    def _wavelet_family_callback(option, optstr, value, parser):
        """Callback for -w|--wavelet-family cmdline option
        """
        wl_list = pywt.wavelist()
        wl_list_str = ", ".join(
                ['-1: None'] + ['%d:%s' % w for w in enumerate(wl_list)])
        if value == "list":
            print "Available wavelet families: " + wl_list_str
            raise SystemExit, 0

        wl_family = value
        try:
            # may be int? ;-)
            wl_family_index = int(value)
            if wl_family_index >= 0:
                try:
                    wl_family = wl_list[wl_family_index]
                except IndexError:
                    print "Index is out of range. " + \
                          "Following indexes with names are known: " + \
                          wl_list_str
                    raise SystemExit, -1
            else:
                wl_family = 'None'
        except ValueError:
            pass
        # Check the value
        wl_family = wl_family.lower()
        if wl_family == 'none':
            wl_family = None
        elif not wl_family in wl_list:
            print "Uknown family '%s'. Known are %s" % (wl_family, ', '.join(wl_list))
            raise SystemExit, -1
        # Store it in the parser
        setattr(parser.values, option.dest, wl_family)


    opt.wavelet_family = \
            Option("-w", "--wavelet-family", callback=_wavelet_family_callback,
                   action="callback", type="string", dest="wavelet_family",
                   default='-1',
                   help="Wavelet family: string or index among the available. " +
                   "Run with '-w list' to see available families")

    opt.wavelet_decomposition = \
            Option("-W", "--wavelet-decomposition",
                   action="store", type="choice", dest="wavelet_decomposition",
                   default='dwt', choices=['dwt', 'dwp'],
                   help="Wavelet decomposition: discrete wavelet transform "+
                   "(dwt) or packet (dwp)")

    opts.add('wavelet', [opt.wavelet_family, opt.wavelet_decomposition],
             "Wavelets mappers")


# Box options

opt.boxlength = \
    Option("--boxlength",
           action="store", dest="boxlength", default=1, type='int',
           help="Length of the box in volumes (integer). Default: 1")

opt.boxoffset = \
    Option("--boxoffset",
           action="store", dest="boxoffset", default=0, type='int',
           help="Offset of the box from the event onset in volumes. Default: 0")

opts.add('box', [opt.boxlength, opt.boxoffset], "Box options")


# sample attributes

opt.chunk = \
    Option("--chunk",
           action="store", dest="chunk", default='0',
           help="Id of the data chunk. Default: 0")

opt.chunkLimits = \
    Option("--chunklimits",
           action="store", dest="chunklimits", default=None,
           help="Limit processing to a certain chunk of data given by start " \
                "and end volume number (including lower, excluding upper " \
                "limit). Numbering starts with zero.")

opts.add('chunk', [opt.chunk, opt.chunkLimits],
         "Chunk options AKA Sample attributes XXX")

