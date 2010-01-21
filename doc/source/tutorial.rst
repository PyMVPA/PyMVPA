.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_tutorial:
.. index:: Tutorial

*******************************
Tutorial Introduction to PyMVPA
*******************************

The following chapters offer a tutorial introduction into PyMVPA. This
tutorial is only concerned with aspects directly related to PyMVPA.  It
does **not** teach basic Python_ programming.

Requirements
============

The PyMVPA course assumes some basic knowledge about programming in Python.
For a short self-assessment of your Python skills, please read the
following questions. If you have an approximate answer to each of them, you
can safely proceed to the tutorial. Otherwise, it is recommended that you
take a look at the Python documentation resources listed under `Recommended
Reading and Viewing`_.

.. _Python: http://www.python.org

* Are you using *spaces* or *tabs* for indentation?  Why is that important to
  know?
* What is the difference between `import numpy` and `from numpy import *`?
* What is the difference between a Python list and a tuple?
* What is the difference between a Python list and a Numpy `ndarray`?
* What is the difference between an *iterable* and a *generator* in Python?
* What is a *list comprehension*?
* What is a *callable*?
* What are `*args` and `**kwargs` usually used for?
* When would you use `?` or `??` in IPython?
* What is the difference between a *deep* copy and a *shallow* copy?
* What is a *derived class*?
* Is it always a problem whenever a Python *exception* is raised?

If you could not answer many questions: **Don't panic!** Python is known to
be an easy-to-learn language. If you are already proficient in *any* other
programming language, you can expect to be able to write fairly complex
Python programs after a weekend of training.


What Do I Need To Get Python Running
------------------------------------

Python is avialable for any machine that might be used for PyMVPA-based
processing. Any GNU/Linux distribution already comes with Python by
default. The Python website offers `installers for Windows and MacOS X`_.

.. _installers for Windows and MacOS X: http://www.python.org/download

However, PyMVPA can make use of many additional software packages to
enhance its functionality. Therefore it is preferrable to use a Python
distribution that offers are large variety of scientific Python packages.
For Windows, `Python(x,y)`_ matches these requirements.  For MacOS X, the
MacPorts_ project offers a large variety of Python packages (including
PyMVPA).

.. _Python(x,y): http://www.pythonxy.com
.. _MacPorts: http://www.macports.org

The ideal enviroment is, however, the Debian_ operating system. Debian
offers the largest selection of free and open-source software in the world,
and runs on almost any machine. Moreover, the NeuroDebian_ project provides
Debian packages for a number of popular neuroscience software package, such
as AFNI_ and FSL_.

.. _Debian: http://www.debian.org
.. _NeuroDebian: http://neuro.debian.net

For those who just want to quickly try PyMVPA, or do not want to deal with
installing multiple software package we recommend the `NeuroDebian Virtual
Machine`_. This is a virtual Debian installation that can be ran on Linux,
Windows, and MacOS X. It includes many Python packages, PyMVPA, and other
neuroscience software (including AFNI_ and FSL_).

.. _NeuroDebian Virtual Machine: http://neuro.debian.net/vm.html
.. _AFNI: http://afni.nimh.nih.gov/afni
.. _FSL: http://www.fmrib.ox.ac.uk/fsl



Recommended Reading and Viewing
-------------------------------

This section contains a commented list of useful resources, ranging from
basic Python programming to efficient scientific computing with Python.


Tutorial Introductions Into General Python Programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://en.wikibooks.org/wiki/Non-Programmer's_Tutorial_for_Python_2.0

  Basic from-scratch introduction into Python. This should give you the basics,
  even if you had *no* prior programming experience.

http://www.ibiblio.org/swaroopch/byteofpython/read/

  From the author:

    The aim is that if all you know about computers is how to save text files,
    then you can learn Python from this book. If you have previous programming
    experience, then you can also learn Python from this book.

  We recommend reading the PDF version that is a lot better formatted.

http://diveintopython.org

  A famous tutorial that served as the entry-point into Python for many people.
  However, it has a relatively steep learning curve, and also covers various
  topics which aren't in the focus of scientific computing.

http://docs.python.org/tutorial/

  Written by the creator of Python itself, this is a more comprehensive, but
  also more compressed tutorial that can serve as a reference. Recommended
  as resource for people with basic programming experience in *some* language.


Scientific Computing In Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python_ itself is "just" a generic programming language.  To employ Python
for scientific computing, where a common analysis deals with vast amounts of
numerical data, more specialized tools are needed -- and are provided by
the NumPy_ package.  PyMVPA makes extensive use of
NumPy data structures and functions, therefore we recommend you to get
familiar with it.

.. _NumPy: http://numpy.scipy.org


http://www.scipy.org/Tentative_NumPy_Tutorial

  Useful for a first glimpse at NumPy -- the basis for scientific computing in
  Python.

http://mathesaurus.sourceforge.net/

  Valuable resource for people coming from other languages and environments,
  such as Matlab.  This pages offers cheat sheets with the equivalents of
  commands and expressions in various languages, including Matlab, R and
  Python.

http://www.tramy.us/numpybook.pdf

  This is *the* comprehensive reference manual of the NumPy package. It gives
  answers to questions, yet to be asked.


Interactive Python Shell
~~~~~~~~~~~~~~~~~~~~~~~~

To make interactive use of Python more enjoyable and productive, we suggest
to explore an enhanced interactive environment for Python -- IPython_.

.. _IPython: http://ipython.scipy.org

http://fperez.org/papers/ipython07_pe-gr_cise.pdf

  An article from the author of IPython in the Computing in Science and Engineering
  journal, describing goals and basic features of IPython

http://showmedo.com/videotutorials/series?name=CnluURUTV

  Video tutorials from Jeff Rush walking you through basic and advanced
  features of IPython.  While doing that he also exposes basic constructs of
  Python, so you might like to watch this video whenever you already have
  basic programming experience with any programming language.

http://ipython.scipy.org/moin/Documentation

  IPython documentation page which references additional materials, such as
  the main IPython documentation which extensively covers features of IPython.



The PyMVPA Tutorial -- Table Of Contents
========================================

.. toctree::
   :maxdepth: 2

   tutorial1_start
   tutorial2_datasets
