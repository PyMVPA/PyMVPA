# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper module to enable profiling of the testcase

 If environment variable PROFILELEVEL is set it uses hotshot profiler
 for unittest.main() call. Value of PROFILELEVEL defines number of top
 busy functions to report.

 Environment variable PROFILELINES=1 makes hotshot store information
 per each line, so it could be easily inspected later on.

 Output:
   Profiler stores its Stats into a file named after original script
   (sys.argv[0]) with suffix".prof" appended

 Usage:
   Replace unittest.main() with import runner

 Visualization:
   kcachegrind provides nice interactive GUI to inspect profiler
   results. If PROFILELINES was set to 1, it provides information per
   each line.

   To convert .prof file into a file suitable for kcachegrind, use
   utility hotshot2calltree which comes in package
   kcachegrind-converters.

 Example:

 # profile and output 3 most expensive function calls
 PROFILELEVEL=3 PROFILELINES=1 PYTHONPATH=../ python test_searchlight.py
 # convert to kcachegrind format
 hotshot2calltree -o test_searchlight.py.kcache  test_searchlight.py.prof
 # inspect
 kcachegrind test_searchlight.py.kcache

"""

__test__ = False

import unittest
import sys

from os import environ

from mvpa2 import _random_seed

# Extend TestProgram to print out the seed which was used
class TestProgramPyMVPA(unittest.TestProgram):
    ##REF: Name was automagically refactored
    def run_tests(self):
        if self.verbosity:
            print "MVPA_SEED=%s:" % _random_seed,
            sys.stdout.flush()
        super(TestProgramPyMVPA, self).run_tests()

def run():
    profilelevel = None

    if environ.has_key('PROFILELEVEL'):
        profilelevel = int(environ['PROFILELEVEL'])


    if profilelevel is None:
        TestProgramPyMVPA()
    else:
        profilelines = environ.has_key('PROFILELINES')

        import hotshot, hotshot.stats
        pname = "%s.prof" % sys.argv[0]
        prof = hotshot.Profile(pname, lineevents=profilelines)
        try:
            # actually return values are never setup
            # since unittest.main sys.exit's
            benchtime, stones = prof.runcall( unittest.main )
        except SystemExit:
            pass
        print "Saving profile data into %s" % pname
        prof.close()
        if profilelevel > 0:
            # we wanted to see the summary right here
            # instead of just storing it into a file
            print "Loading profile data from %s" % pname
            stats = hotshot.stats.load(pname)
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(profilelevel)
