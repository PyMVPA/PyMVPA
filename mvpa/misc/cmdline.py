#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common functions and options definitions for command line

Conventions:
Every option (instance of optparse.Option) has prefix "opt". Lists of options
has prefix opts (e.g. optsCommon).

Option name should be camelbacked version of .dest for the option.
"""

# TODO? all options (opt*) might migrate to respective module? discuss
from optparse import Option

from mvpa.misc import verbose

#
# Verbosity options
#
def verboseCallback(option, optstr, value, parser):
    """Callback for -v|--verbose cmdline option
    """
    verbose.level = value
    optstr = optstr                     # pylint shut up
    setattr(parser.values, option.dest, value)

optVerbose = \
    Option("-v", "--verbose", "--verbosity",
           action="callback", callback=verboseCallback, nargs=1,
           type="int", dest="verbose", default=0,
           help="Verbosity level of output")

optsCommon = [optVerbose]

if __debug__:
    from mvpa.misc import debug

    def debugCallback(option, optstr, value, parser):
        """Callback for -d|--debug cmdline option
        """
        if value == "list":
            print "Registered debug IDs:"
            print debug.registered
            raise SystemExit, 0

        optstr = optstr                     # pylint shut up
        entries = value.split(",")
        if entries != "":
            debug.active = entries
        setattr(parser.values, option.dest, value)


    optDebug = Option("-d", "--debug",
                      action="callback", callback=debugCallback,
                      nargs=1,
                      type="string", dest="debug", default="",
                      help="Debug entries to report. " +
                      "Run with '-d list' to get a list of " +
                      "registered entries")

    optsCommon.append(optDebug)


#
# Classifiers options
#
optClf = \
    Option("--clf",
           action="store", type="string", dest="clf",
           default='knn',
           help="Type of classifier to be used. Possible values are: 'knn', " \
                "'lin_nu_svmc', 'rbf_nu_svmc'. Default: knn")

optRadius = \
    Option("-r", "--radius",
           action="store", type="float", dest="radius",
           default=5.0,
           help="Radius to be used (eg for the searchlight). Default: 5.0")

optKNearestDegree = \
    Option("-k", "--k-nearest",
           action="store", type="int", dest="knearestdegree", default=3,
           help="Degree of k-nearest classifier. Default: 3")

optSVMNu = \
    Option("--nu",
           action="store", type="float", dest="nu", default=0.1,
           help="nu parameter for soft-margin nu-SVM classification. " \
                "Default: 0.1")

optsSVM = [optSVMNu]

optCrossfoldDegree = \
    Option("-c", "--crossfold",
           action="store", type="int", dest="crossfolddegree", default=1,
           help="Degree of N-fold crossfold. Default: 1")

optZScore = \
    Option("--zscore",
           action="store_true", dest="zscore", default=0,
           help="Enable zscoring of dataset samples. Default: Off")
