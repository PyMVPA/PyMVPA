.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. _chap_misc:
.. index:: misc

*************
Miscellaneous
*************

.. automodule:: mvpa2.misc

.. only:: html

   Related API documentation
   =========================

   .. currentmodule:: mvpa2
   .. autosummary::
      :toctree: generated

      atlases
      misc.args
      misc.attrmap
      misc.cmdline
      misc.data_generators
      misc.errorfx
      misc.exceptions
      misc.fx
      misc.neighborhood
      misc.sampleslookup
      misc.stats
      misc.support
      misc.surfing
      misc.transformers
      misc.vproperty



.. index:: settings, configuration, cfg

Managing (Custom) Configurations
================================

PyMVPA provides a facility to handle arbitrary configuration settings. This
facility can be used to control some aspects of the behavior of PyMVPA
itself, as well as to store and query custom configuration items, e.g. to
control one's own analysis scripts.

An instance of this configuration manager is loaded whenever the `mvpa2` module
is imported. It can be used from any script like this:

  >>> from mvpa2 import cfg

By default the config manager reads settings from two config files (if any of
them exists). The first is a file named `.pymvpa2.cfg` and located in the
user's home directory.  The second is `pymvpa2.cfg` in the current directory.
Please note, that settings found in the second file override the ones in the
first.

The syntax of both files is the one also known from the Windows INI files.
Basically, `Python's ConfigParser`_ is used to read those file and the config
supports whatever this parser can read. A minimal example config file might
look like this::

    [general]
    verbose = 1

It consists of a section `general` containing a single setting `verbose`,
which is set to `1`. PyMVPA recognizes a number of such sections and
configuration variables. A full list is shown at the end of this section and
is also available in the source package (`doc/examples/pymvpa2.cfg`).

.. _Python's ConfigParser: http://docs.python.org/lib/module-ConfigParser.html

In addition to configuration files, the config manager also looks for special
environment variables to read settings from. Names of such variables have to
start with `MVPA_` following by the an optional section name and the variable
name itself (with `_` as delimiter). If no section name is provided, the
variables will be associated with section `general`. Some examples::

    MVPA_VERBOSE=1

will become::

    [general]
    verbose = 1

However, :envvar:`MVPA_VERBOSE_OUTPUT` `= stdout` becomes::

    [verbose]
    output = stdout

Any lenght of variable name is allowed, e.g. ``MVPA_SEC1_LONG_VARIABLE_NAME=1``
becomes::

    [sec1]
    long variable name = 1

Settings read from environment variables have the highest priority and
override settings found in the config files. Therefore environment variables
can be used to quickly adjust some setting without having to edit the config
files.

The config manager can easily be queried from inside scripts. In addition to
the interface of `Python's ConfigParser`_ it has a few convenience functions
mostly to allow for a default value in case no setting was found. For example:

  >>> cfg.getboolean('warnings', 'suppress', default=False)
  True

queries the config manager whether warnings should be suppressed (i.e. if
there is a variable `suppress` in section `warnings`). In case, there is no
such setting, i.e. neither config files nor environment variables defined it,
the `default` values is returned. Please see the documentation of
`ConfigManager`_ for its full functionality.

.. _ConfigManager: api/mvpa2.base.config.ConfigManager-class.html


.. index:: config file

The source tarballs includes an example configuration file
(`doc/examples/pymvpa2.cfg`) with the comprehensive list of settings recognized
by PyMVPA itself:

.. literalinclude:: ../examples/pymvpa2.cfg
   :language: ini


.. index:: progress tracking, verbosity, debug, warning

Progress Tracking
=================

.. some parts should migrate into developer reference I guess

There are 3 types of messages PyMVPA can produce:

verbose_
   regular informative messages about generic actions being performed

debug_
   messages about the progress of computation, manipulation on data
   structures

warning_
    messages which are reported by mvpa if something goes a little
    unexpected but not critical


.. _verbose: api/mvpa2.misc-module.html#verbose
.. _debug: api/mvpa2.misc-module.html#debug
.. _warning: api/mvpa2.misc-module.html#warning


.. index:: redirecting output

Redirecting Output
------------------

By default, all types of messages are printed by PyMVPA to the standard
output. It is possible to redirect them to standard error, or a file, or a
list of multiple such targets, by using environment variable
``MVPA_?_OUTPUT``, where X is either ``VERBOSE``, ``DEBUG``, or ``WARNING``
correspondingly. E.g.::

  export MVPA_VERBOSE_OUTPUT=stdout,/tmp/1 MVPA_WARNING_OUTPUT=/tmp/3 MVPA_DEBUG_OUTPUT=stderr,/tmp/2

would direct verbose messages to standard output as well as to ``/tmp/1``
file, warnings will be stored only in ``/tmp/3``, and debug output would
appear on standard error output, as well as in the file ``/tmp/2``.

PyMVPA output redirection though has no effect on external libraries debug
output if corresponding debug_ target is enabled

shogun
   debug output (if any of internal ``SG_`` debug_ targets is enabled)
   appears on standard output

SMLR
   debug output (if ``SMLR_`` debug_ target is enabled) appears on standard
   output

LIBSVM
   debug output (if ``LIBSVM`` debug_ target is enabled) appears on
   standard error

One of the possible redirections is Python's ``StringIO`` class. Instance of
such class can be added to the ``handlers`` and queried later on for the
information to be dumped to a file later on. It is useful if output path is
specified at run time, thus it is impossible to redirect verbose or debug from
the start of the program:

  >>> import sys
  >>> from mvpa2.base import verbose
  >>> from StringIO import StringIO
  >>> stringout = StringIO()
  >>> verbose.handlers = [sys.stdout, stringout]
  >>> verbose.level = 3
  >>>
  >>> verbose(1, 'msg1')
   msg1
  >>> out_prefix='/tmp/'
  >>>
  >>> verbose(2, 'msg2')
    msg2
  >>> # open('%sverbose.log' % out_prefix, 'w').write(stringout.getvalue())
  >>> print stringout.getvalue(),
   msg1
    msg2
  >>>

.. index:: verbosity

Verbose Messages
----------------

Primarily for a user of PyMVPA to provide information about the
progress of their scripts. Such messages are printed out if their
level specified as the first parameter to verbose_ function call is
less than specified. There are two easy ways to specify verbosity
level:

* command line: you can use opt.verbose_ for precrafted command
  line option for to give facility to change it from your script (see
  examples)
* environment variable :envvar:`MVPA_VERBOSE`
* code: verbose.level property

The following verbosity levels are supported:

:0: nothing besides errors
:1: high level stuff -- top level operation or file operations
:2: cmdline handling
:3: n.a.
:4: computation/algorithm relevant thing


.. index:: warning

Warning Messages
----------------

Reported by PyMVPA if something goes a little unexpected but not critical. By
default they are printed just once per occasion, i.e. once per piece of code
where it is called. Following environment variables control the
behavior of warnings:

* :envvar:`MVPA_WARNINGS_COUNT` `=<int>` controls for how many invocations of
  specific warning it gets printed (default behavior is 1 for
  once). Specification of negative count results in all invocations
  being printed, and value of 0 obviously suppresses the warnings
* :envvar:`MVPA_WARNINGS_SUPPRESS` analogous to :envvar:`MVPA_WARNINGS_COUNT` `=0` it
  resultant behavior
* :envvar:`MVPA_WARNINGS_BT` `=<int>` controls up to how many lines of traceback
  is printed for the warnings

In python code, invocation of warning with argument ``bt = True``
enforces printout of traceback whenever warning tracebacks are
disabled by default.


.. index:: debug

Debug Messages
--------------

Debug messages are used to track progress of any computation inside
PyMVPA while the code run by python without optimization (i.e. without
``-O`` switch to python). They are specified not by the level but by
some id usually specific for a particular PyMVPA routine. For example
``RFEC`` id causes debugging information about `Recursive Feature
Elimination call`_ to be printed (See `base module sources`_ for the
list of all ids, or print ``debug.registered`` property).

Analogous to verbosity level there are two easy ways to specify set of
ids to be enabled (reported):

* command line: you can use optDebug_ for precrafted command line
  option to provide it from your script (see examples). If in command
  line if optDebug_ is used, ``-d list`` is given, PyMVPA will print
  out list of known ids.
* environment: variable :envvar:`MVPA_DEBUG` can contain comma-separated
  list of ids or python regular expressions to match multiple ids. Thus
  specifying :envvar:`MVPA_DEBUG` `=CLF.*` would enable all ids which start with
  ``CLF``, and :envvar:`MVPA_DEBUG` `=.*` would enable all known ids.
* code: debug.active property (e.g. ``debug.active = [ 'RFEC', 'CLF' ]``)

Besides printing debug messages, it is also possible to print some
metric. You can define new metrics or select predefined ones:

vmem
  (Linux specific): amount of virtual memory consumed by the task

pid
  (Linux specific): PID of the process

reltime
  How many seconds passed since previous debug printout

asctime
  Time stamp

tb
  Traceback (``module1:line_number1[,line_number2...]>module2:line_number..``)
  where this debug statement was requested

tbc
  Concise traceback printout -- prefix common with the previous
  invocation is replaced with ``...``

To enable list of metrics you can use :envvar:`MVPA_DEBUG_METRICS` environment
variable to list desired metric names comma-separated. If ``ALL`` is provided,
it enables all the metrics.

As it was mentioned earlier, debug messages are printed only in
non-optimized python invocation. That was done to eliminate any
slowdown introduced by such 'debugging' output, which might appear at
some computational bottleneck places in the code.

Some of the debug ids are defined to facilitate additional checking of the
validity of the analysis. Their debug ids a prefixed by
``CHECK_``. E.g. ``CHECK_RETRAIN`` id would cause additional checking of the
data in retraining phase. Such additional testing might spot out some bugs in
the internal logic, thus enabled when full test suite is ran.

.. TODO: Unify loggers behind verbose and debug. imho debug should have
   also way to specify the level for the message so we could provide
   more debugging information if desired.

.. _opt.verbose: api/mvpa2.misc.cmdline-module.html#opt.verbose
.. _optDebug: api/mvpa2.misc.cmdline-module.html#optDebug
.. _base module sources: api/mvpa2.base-pysrc.html
.. _Recursive Feature Elimination call: api/mvpa2.featsel.rfe.RFE-class.html#__call__


PyMVPA Status Summary
---------------------

While reporting found bugs, it is advised to provide information about the
operating system/environment and availability of PyMVPA externals.  Please use
:func:`~mvpa2.base.info.wtf` to collect such useful information to be included
with the bug reports.

Alternatively, same printout can be obtained upon not handled exception
automagically, if environment variable :envvar:`MVPA_DEBUG_WTF` is set.


Additional Little Helpers
=========================

.. index:: random number generation, RNG

Random Number Generation
------------------------

To facilitate reproducible troubleshooting, a seed value of random generator
of NumPy can be provided in debug mode (python is called without ``-O``) via
environment variable :envvar:`MVPA_SEED` `=<int>`. Otherwise it gets seeded with random
integer which can be displayed with debug id ``RANDOM`` e.g.::

  > MVPA_SEED=123 MVPA_DEBUG=RANDOM python test_clf.py
  [RANDOM] DBG: Seeding RNG with 123
  ...
  > MVPA_DEBUG=RANDOM python test_clf.py
  [RANDOM] DBG: Seeding RNG with 1447286079
  ...


Unittests at a Grasp
--------------------

.. index:: unittests

If it is needed to just quickly grasp through all unittests without making
them to test multiple classifiers (implemented with sweeparg), define
environmental variable :envvar:`MVPA_TESTS_QUICK` e.g.::

  > MVPA_WARNINGS_SUPPRESS=no MVPA_TESTS_QUICK=yes python test_clf.py
  ...............
  ----------------------------------------------------------------------
  Ran 15 tests in 0.845s

Some tests are not 100% deterministic as they operate on random data (e.g.
the performance of a randomly initialized classifier). Therefore, in some cases,
specific unit tests might fail when running the full test battery. To exclude
these test cases (and only those where non-deterministic behavior immanent) one
can use the :envvar:`MVPA_TESTS_LABILE` configuration and set it to 'off'.


.. index:: FSL, detrending, motion correction

FSL Bindings
============

PyMVPA contains a few little helpers to make interfacing with FSL_ easier.
The purpose of these helpers is to increase the efficiency when doing an
analysis by (re)using useful information that is already available from some
FSL output. FSL usually stores most interesting information in the NIfTI
format. Therefore it can be easily imported into PyMVPA using PyNIfTI. However,
some information is stored in text files, e.g. estimated motion correction
parameters and *FEAT's three-column custom EV* files. PyMVPA provides import
and export helpers for both of them (among other stuff like a *MELODIC*
results import helper).

.. _motion-aware_detrending:

Here is an example how the *McFlirt* parameter output can be used to perform
motion-aware data detrending:


  >>> from os import path
  >>> import numpy as np
  >>>
  >>> # some dummy dataset
  >>> from mvpa2.datasets import Dataset
  >>> ds = Dataset(samples=np.random.normal(size=(19, 3)))
  >>>
  >>> # load motion correction output
  >>> from mvpa2.misc.fsl.base import McFlirtParams
  >>> mc = McFlirtParams(path.join('mvpa2', 'data', 'bold_mc.par'))
  >>>
  >>> # simple plot using pylab (use pylab.show() or pylab.savefig()
  >>> # afterwards)
  >>> mc.plot()
  >>>
  >>> # merge the correction parameters into the dataset itself
  >>> for param in mc:
  ...     ds.sa['mc_' + param] = mc[param]
  >>>
  >>> # detrend some dataset with mc params as additonal regressors
  >>> from mvpa2.mappers.detrend import poly_detrend
  >>> res = poly_detrend(ds, opt_regs=['mc_x', 'mc_y', 'mc_z',
  ...                                  'mc_rot1', 'mc_rot2', 'mc_rot3'])
  >>> # 'res' contains all regressors and their associated weights

All FSL bindings are located in the `mvpa2.misc.fsl`_ module.

.. _FSL: http://www.fmrib.ox.ac.uk
.. _mvpa2.misc.fsl: api/mvpa2.misc.fsl-module.html

