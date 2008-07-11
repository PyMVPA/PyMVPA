#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common functions and options definitions for command line

__docformat__ = 'restructuredtext'

Conventions:
Every option (instance of optparse.Option) has prefix "opt". Lists of options
has prefix opts (e.g. `optsCommon`).

Option name should be camelbacked version of .dest for the option.
"""

# TODO? all options (opt*) might migrate to respective module? discuss
from optparse import OptionParser, Option, OptionGroup

# needed for verboseCallback
from mvpa.base import verbose, externals


# TODO: try to make groups definition somewhat lazy, since now
# whenever a group is created, those parameters are already known by
# parser, although might not be listed in the list of used and not by
# --help. But their specification on cmdline doesn't lead to
# error/help msg.
#
# Conflict hanlder to resolve situation that we have the same option added
# to some group and also available 'freely'
#
parser = OptionParser(add_help_option=False,
                      conflict_handler="resolve")

#
# Verbosity options
#
def verboseCallback(option, optstr, value, parser):
    """Callback for -v|--verbose cmdline option
    """
    verbose.level = value
    optstr = optstr                     # pylint shut up
    setattr(parser.values, option.dest, value)

optHelp = \
    Option("-h", "--help", "--sos",
           action="help",
           help="Show this help message and exit")

optVerbose = \
    Option("-v", "--verbose", "--verbosity",
           action="callback", callback=verboseCallback, nargs=1,
           type="int", dest="verbose", default=0,
           help="Verbosity level of output")
"""Pre-cooked `optparse`'s option to specify verbose level"""

optsCommon = OptionGroup(parser, title="Generic"
#   , description="Options often used in a PyMVPA application"
                         )

optsCommon.add_options([optVerbose, optHelp])


if __debug__:
    from mvpa.base import debug

    def debugCallback(option, optstr, value, parser):
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
        debug.setActiveFromString(value)


        setattr(parser.values, option.dest, value)


    optDebug = Option("-d", "--debug",
                      action="callback", callback=debugCallback,
                      nargs=1,
                      type="string", dest="debug", default="",
                      help="Debug entries to report. " +
                      "Run with '-d list' to get a list of " +
                      "registered entries")

    optsCommon.add_option(optDebug)


#
# Classifiers options
#
optClf = \
    Option("--clf",
           type="choice", dest="clf",
           choices=['knn', 'svm', 'ridge'], default='svm',
           help="Type of classifier to be used. Default: svm")

optRadius = \
    Option("-r", "--radius",
           action="store", type="float", dest="radius",
           default=5.0,
           help="Radius to be used (eg for the searchlight). Default: 5.0")


optKNearestDegree = \
    Option("-k", "--k-nearest",
           action="store", type="int", dest="knearestdegree", default=3,
           help="Degree of k-nearest classifier. Default: 3")

optsKNN = OptionGroup(parser, "Specification of kNN")
optsKNN.add_option(optKNearestDegree)

optSVMC = \
    Option("-C", "--svm-C",
           action="store", type="float", dest="svm_C", default=1.0,
           help="C parameter for soft-margin C-SVM classification. " \
                "Default: 1.0")

optSVMNu = \
    Option("--nu", "--svm-nu",
           action="store", type="float", dest="svm_nu", default=0.1,
           help="nu parameter for soft-margin nu-SVM classification. " \
                "Default: 0.1")

optSVMGamma = \
    Option("--gamma", "--svm-gamma",
           action="store", type="float", dest="svm_gamma", default=1.0,
           help="gamma parameter for Gaussian kernel of RBF SVM. " \
                "Default: 1.0")


optsSVM = OptionGroup(parser, "SVM specification")
optsSVM.add_options([optSVMNu, optSVMC, optSVMGamma])


# Crossvalidation options

optCrossfoldDegree = \
    Option("-c", "--crossfold",
           action="store", type="int", dest="crossfolddegree", default=1,
           help="Degree of N-fold crossfold. Default: 1")

optsGener = OptionGroup(parser, "Generalization estimates")
optsGener.add_options([optCrossfoldDegree])

# preprocess options

optZScore = \
    Option("--zscore",
           action="store_true", dest="zscore", default=0,
           help="Enable zscoring of dataset samples. Default: Off")

optTr = \
    Option("--tr",
           action="store", dest="tr", default=2.0, type='float',
           help="fMRI volume repetition time. Default: 2.0")

optDetrend = \
    Option("--detrend",
           action="store_true", dest="detrend", default=0,
           help="Do linear detrending. Default: Off")

optsPreproc = OptionGroup(parser, "Preprocessing options")
optsPreproc.add_options([optZScore, optTr, optDetrend])

# Wavelets options
if externals.exists('pywt'):
    import pywt
    def waveletFamilyCallback(option, optstr, value, parser):
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


    optWaveletFamily = \
            Option("-w", "--wavelet-family", callback=waveletFamilyCallback,
                   action="callback", type="string", dest="wavelet_family",
                   default='-1',
                   help="Wavelet family: string or index among the available. " +
                   "Run with '-w list' to see available families")


    # No sense for a single option
    # optsWavelets = OptionGroup(parser, "Options for wavelets mappers")
    # optsWavelets.add_options([optWaveletFamily])


# Box options

optBoxLength = \
    Option("--boxlength",
           action="store", dest="boxlength", default=1, type='int',
           help="Length of the box in volumes (integer). Default: 1")

optBoxOffset = \
    Option("--boxoffset",
           action="store", dest="boxoffset", default=0, type='int',
           help="Offset of the box from the event onset in volumes. Default: 0")

optsBox = OptionGroup(parser, "Box options")
optsBox.add_options([optBoxLength, optBoxOffset])


# sample attributes

optChunk = \
    Option("--chunk",
           action="store", dest="chunk", default='0',
           help="Id of the data chunk. Default: 0")

optChunkLimits = \
    Option("--chunklimits",
           action="store", dest="chunklimits", default=None,
           help="Limit processing to a certain chunk of data given by start " \
                "and end volume number (including lower, excluding upper " \
                "limit). Numbering starts with zero.")

optsChunk = OptionGroup(parser, "Chunk options AKA Sample attributes XXX")
optsChunk.add_options([optChunk, optChunkLimits])



