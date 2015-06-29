# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Submodule to provide sweepargs decorator for unittests"""

__docformat__ = 'restructuredtext'

import sys
import traceback as tbm

from mvpa2 import cfg
from mvpa2.base.dochelpers import safe_str
from mvpa2.testing.tools import SkipTest

if __debug__:
    from mvpa2.base import debug

__all__ = [ 'sweepargs' ]


def sweepargs(**kwargs):
    """Decorator function to sweep over a given set of classifiers

    Parameters
    ----------
    clfs : list of `Classifier`
      List of classifiers to run method on

    Often some unittest method can be ran on multiple classifiers.
    So this decorator aims to do that
    """

    from mvpa2.clfs.base import Classifier
    from mvpa2.base.state import ClassWithCollections

    def unittest_method(method):
        def do_sweep(*args_, **kwargs_):
            """Perform sweeping over provided keyword arguments
            """
            def untrain_clf(argvalue):
                """Little helper"""
                if isinstance(argvalue, Classifier):
                    # clear classifier after its use -- just to be sure ;-)
                    argvalue.params.retrainable = False
                    argvalue.untrain()

            failed_tests = {}
            skipped_tests = []
            report_progress = cfg.get('tests', 'verbosity', default=1) > 1
            for argname in kwargs.keys():
                for argvalue in kwargs[argname]:
                    if isinstance(argvalue, Classifier):
                        # clear classifier before its use
                        argvalue.untrain()
                    if isinstance(argvalue, ClassWithCollections):
                        argvalue.ca.reset()
                    # update kwargs_
                    kwargs_[argname] = argvalue
                    # do actual call
                    try:
                        if __debug__:
                            debug('TEST', 'Running %s on args=%r and kwargs=%r'
                                  % (method.__name__, args_, kwargs_))
                        method(*args_, **kwargs_)
                        status = '+'
                    except SkipTest, e:
                        skipped_tests += [e]
                        status = 'S'
                    except AssertionError, e:
                        status = 'F'
                        estr = str(e)
                        etype, value, tb = sys.exc_info()
                        # literal representation of exception tb, so
                        # we could group them later on
                        eidstr = '  '.join(
                            [l for l in tbm.format_exception(etype, value, tb)
                             if not ('do_sweep' in l
                                     or 'unittest.py' in l
                                     or 'AssertionError' in l
                                     or 'Traceback (most' in l)])

                        # Store exception information for later on groupping
                        if not eidstr in failed_tests:
                            failed_tests[eidstr] = []

                        sargvalue = safe_str(argvalue)
                        if not (__debug__ and 'TEST' in debug.active):
                            # by default lets make it of sane length
                            if len(sargvalue) > 100:
                                sargvalue = sargvalue[:95] + ' ...'
                        failed_tests[eidstr].append(
                            # skip top-most tb in sweep_args
                            (argname, sargvalue, tb.tb_next, estr))

                        if __debug__:
                            msg = "%s on %s=%s" % (estr, argname, safe_str(argvalue))
                            debug('TEST', 'Failed unittest: %s\n%s'
                                  % (eidstr, msg))
                    if report_progress:
                        sys.stdout.write(status)
                        sys.stdout.flush()

                    untrain_clf(argvalue)
                    # TODO: handle different levels of unittests properly
                    if cfg.getboolean('tests', 'quick', False):
                        # on TESTQUICK just run test for 1st entry in the list,
                        # the rest are omitted
                        # TODO: proper partitioning of unittests
                        break
            if report_progress:
                sys.stdout.write(' ')
                sys.stdout.flush()
            if len(failed_tests):
                # Lets now create a single AssertionError exception
                # which would nicely incorporate all failed exceptions
                multiple = len(failed_tests) != 1 # is it unique?
                # if so, we don't need to reinclude traceback since it
                # would be spitted out anyways below
                estr = ""
                cestr = "lead to failures of unittest %s" % method.__name__
                if multiple:
                    estr += "\n Different scenarios %s "\
                            "(specific tracebacks are below):" % cestr
                else:
                    estr += "\n Single scenario %s:" % cestr
                for ek, els in failed_tests.iteritems():
                    estr += '\n'
                    if multiple:
                        estr += ek
                    estr += "  on\n    %s" % ("    ".join(
                            ["%s=%s%s\n" %
                             (ea, eav,
                              # Why didn't I just do regular for loop? ;)
                              ":\n     ".join([xx for xx in [' ', es]
                                               if xx != '']))
                             for ea, eav, etb, es in els]))
                    # take first one... they all should be identical
                    etb = els[0][2]
                raise AssertionError(estr), None, etb
            if len(skipped_tests):
                # so if nothing has failed, lets at least report that some were
                # skipped -- for now just  a simple SkipTest message
                raise SkipTest("%d tests were skipped in testing %s"
                               % (len(skipped_tests), method.func_name))
        do_sweep.func_name = method.func_name
        do_sweep.__doc__ = method.__doc__
        return do_sweep

    if len(kwargs) > 1:
        raise NotImplementedError, \
              "No sweeping over multiple arguments in sweepargs. Meanwhile " \
              "use two @sweepargs decorators for the test."

    return unittest_method
