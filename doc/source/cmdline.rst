.. -*- mode: rst -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:

.. _chap_cmdline:

**********************
Command line interface
**********************

Since version 2.3 PyMVPA includes a command line interface that allows for
using PyMVPA in shell scripts and other non-Python environments. When
installed, PyMVPA offers a single :command:`pymvpa2` command through which all
supported functionality is made available. Analogous to the Python interface,
the command line interface is broken down into modules that individually serve
a specific purpose, and can be combined into more complex analysis pipelines.
Each module is exposed as a sub-command of :command:`pymvpa2`. All sub-commands
are documented individually below.

Both the primary command :command:`pymvpa2`, as well as all sub-commands
support specific options that need to be given in the right location on the
command line. The basic structure of any command line is this::

  $ pymvpa2 [{{primary options}}] {{sub-command}} [{{secondary options}}]

where ``[]`` indicates an optional segment, ``primary options`` are any options
for the main :command:`pymvpa2` command, and ``secondary options`` are options
of a sub-command. For example::

  $ pymvpa2 --help

will yield the documentation of the main command, but::

  $ pymvpa2 info --help

will yield the documentation of the ``info`` sub-command.

Documentation of the main ``pymvpa2`` command
=============================================

The documentation of the :command:`pymvpa2` command (accessible via ``--help``)
includes a list of all available sub-commands on a particular system.

.. toctree::

   generated/cmd_pymvpa2

Sub-command documentation
=========================

.. toctree::

   generated/cmd_mkds
   generated/cmd_mkevds
   generated/cmd_info
   generated/cmd_dump
   generated/cmd_preproc
   generated/cmd_hyperalign
   generated/cmd_clfcv

Example scripts
===============

.. toctree::

   cmdline/datasets

Create a new PyMVPA command (for developers)
============================================


write me
