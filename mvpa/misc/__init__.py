#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA misc modules"""

from sys import stdout, stderr

from os import environ

from verbosity import LevelLogger

#
# Setup verbose and debug outputs
#

# TODO: check if they are actually singletons...
verbose = LevelLogger(handlers=[stdout])
errors = LevelLogger(handlers=[stderr])

# Levels for verbose
# 0 -- nothing besides errors
# 1 -- high level stuff -- top level operation or file operations
# 2 -- cmdline handling
# 3 --
# 4 -- computation/algorithm relevant thingies

# Lets check if environment can tell us smth
if environ.has_key('MVPA_VERBOSE'):
    verbose.level = int(environ['MVPA_VERBOSE'])

if __debug__:
    from verbosity import DebugLogger
    # NOTE: all calls to debug must be preconditioned with
    # if __debug__:
    debug = DebugLogger(handlers=[stderr])

    # set some debugging matricses to report
    # debug.registerMetric('vmem')

    # List agreed sets for debug
    debug.register('LAZY', "Miscelaneous 'lazy' evaluations")
    debug.register('PLF',  "PLF call")
    debug.register('SLC',  "Searchlight call")
    debug.register('RFEC', "Recursive Feature Elimination call")
    debug.register('DS',   "*Dataset debugging")

    # Lets check if environment can tell us smth
    if environ.has_key('MVPA_DEBUG'):
        debug.setActiveFromString(environ['MVPA_DEBUG'])
