#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Common functions for command line"""

# XXX all ones below might migrate to respective module? discuss
from optparse import Option

from mvpa.misc import verbose

def verboseCallback(option, optstr, value, parser):
    """Callback for -v|--verbose cmdline option
    """
    verbose.level = value
    optstr = optstr                     # pylint shut up
    setattr(parser.values, option.dest, value)

optionVerbose = Option("-v", "--verbose", "--verbosity",
                       action="callback", callback=verboseCallback, nargs=1,
                       type="int", dest="verbose", default=0,
                       help="Verbosity level of output")

commonOptions = [optionVerbose]

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


    optionDebug = Option("-d", "--debug",
                         action="callback", callback=debugCallback,
                         nargs=1,
                         type="string", dest="debug", default="",
                         help="Debug entries to report. " +
                         "Run with '-d list' to get a list of " +
                         "registered entries")

    commonOptions.append(optionDebug)
