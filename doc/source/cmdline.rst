.. -*- mode: rst -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:

.. _chap_cmdline:

**********************
Command line interface
**********************

Since version 2.3 PyMVPA includes a command line interface that allows for
using PyMVPA in shell scripts and other non-Python environments.  The scope of
the command line interface is to expose the most commonly used building blocks
of PyMVPA and connect them using a series of commands, where intermediate
results are stored in HDF5 files. It is not intended to provide the same
flexibility as the Python API, but aims to avoid boilerplate code for the most
commonly used analysis and processing strategies.

At present, the command line interface is still in its early development stages
and future API changes cannot be ruled out, and further extensions are expected
for future releases.

When installed, PyMVPA offers a single :command:`pymvpa2` command through which
all supported functionality is made available. Analogous to the Python
interface, the command line interface is broken down into modules that
individually serve a specific purpose, and can be combined into more complex
analysis pipelines.  Each module is exposed as a sub-command of
:command:`pymvpa2`. All sub-commands are documented individually below.

Both the primary command :command:`pymvpa2`, as well as all sub-commands
support specific options that need to be given in the right location on the
command line. The basic structure of any command line is this::

  $ pymvpa2 [{{primary options}}] {{sub-command}} [{{secondary options}}]

where ``[]`` indicates an optional segment, ``primary options`` are any options
for the main :command:`pymvpa2` command, and ``secondary options`` are options
of a sub-command. For example::

  $ pymvpa2 --help

will yield the documentation of the main command, but::

  $ pymvpa2 mkds --help

will yield the documentation of the ``mkds`` sub-command.


Documentation of the main ``pymvpa2`` command
=============================================

The documentation of the :command:`pymvpa2` command (accessible via ``--help``)
includes a list of all available sub-commands on a particular system.

.. toctree::

   generated/cmd_pymvpa2


Sub-command documentation
=========================

Create, modify and convert datasets
-----------------------------------

.. toctree::

   generated/cmd_mkds
   generated/cmd_mkevds
   generated/cmd_select
   generated/cmd_preproc
   generated/cmd_dump
   generated/cmd_describe

Perform analyses
----------------

.. toctree::

   generated/cmd_crossval
   generated/cmd_searchlight

Auxilliary command
------------------

.. toctree::

   generated/cmd_info
   generated/cmd_exec
   generated/cmd_atlaslabeler
   generated/cmd_ofmotionqc
   generated/cmd_ttest

.. _cmdline_example_scripts:

Example scripts
===============

Here are a few executable example that come with the PyMVPA source code and
demonstrate how to use PyMVPA's command line interface in actual scripts.

.. toctree::

   cmdline/start_easy
   cmdline/fmri_analyses
   cmdline/query_pymvpa


For developers
==============

Create a new PyMVPA command
---------------------------

For now just a few notes:

- No positional arguments, only options

  The majority of all commands (can) have very complex argument lists.
  Positional arguments are harder to identify, and only offer a flat list
  for structured input, without the possibility to specify nested list
  like input.

- whenever possible use (and improve) common option definitions from
  mvpa2.cmdline.helpers

- An option specifying an output location should be called -o/-output-...

- Whenever a dataset needs to be loaded and there is no special reason to do
  anything fancy, ``arg2ds()`` should be used to load it.

  This will allow all relevant command to vstack input datasets on the fly,
  and significantly shapes the "standard" workflow, as data can be keep in fine
  grained structures avoiding the need to produce tailored datasets for any
  particular operation.

- If a command requires multiple separate datasets multiple --input options
  should be used to specify them.
