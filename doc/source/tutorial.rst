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


Things to be aware of
=====================

* What is a *list comprehension*?
* What is the difference between a *deep* copy and a *shallow* copy?
* What is the difference between a Python list and a tuple?
* What is the difference between a Python list and a Numpy `ndarray`?
* What is the difference between an *iterable* and a *generator* in Python?
* What is the difference between `import numpy` and `from numpy import *`?
* What is a *callable*?
* What is a *derived class*?
* What `*args` and `**kwargs` are usually used for?
* Are you using *spaces* or *tabs* for indentation?  Why do we bother asking?
* When would you use *?* or *??* in IPython?
* Is it always a problem whenever a Python *exception* is raised?


Recommended Reading and Viewing
===============================

Tutorial Introductions Into General Python Programming
------------------------------------------------------

`http://en.wikibooks.org/wiki/Non-Programmer's_Tutorial_for_Python_2.0`

  Basic from-scratch introduction into Python. This should give you the basics,
  even if you had *no* prior programming experience.

http://www.ibiblio.org/swaroopch/byteofpython/read/

  From the author:

    The aim is that if all you know about computers is how to save text files,
    then you can learn Python from this book. If you have previous programming
    experience, then you can also learn Python from this book.

http://diveintopython.org

  A famous tutorial that served as the entry-point into Python for many people.
  However, it has a relatively steep learning curve, and also covers various
  topics which aren't in the focus of scientific computing.

http://docs.python.org/tutorial/

  Written by the creator of Python itself, this is a more comprehensive, but
  also more compressed tutorial that can also serve as a reference. Recommended
  as resource for people with basic programming experience in *some* language.


Scientific Computing In Python
------------------------------

Python itself is a generic programming language.  To employ Python for
scientific computing, where usual analysis deals with vast amounts of
numerical data, NumPy module was developed.  PyMVPA makes extensive use of
NumPy data structures and functions, therefore we recommend you to get
familiar with it.

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
------------------------

Being a scripting programming language, `python` itself provides a basic
interface for interactive scripting.  To make interactive use of Python more
enjoyable and productive, we suggest to explore an enhanced interactive
environment for Python -- IPython.

http://fperez.org/papers/ipython07_pe-gr_cise.pdf

  An article from IPython author in the Computing in Science and Engineering
  journal describing goals and basic features of IPython

http://showmedo.com/videotutorials/series?name=CnluURUTV

  Video tutorials from Jeff Rush walking you through basic and advanced
  features of IPython.  While doing that he also exposes basic constructs of
  Python, so you might like to watch this video whenever you already have
  basic programming experience with any programming language.

http://ipython.scipy.org/moin/Documentation

  IPython documentation page which references additional materials, such as
  the main IPython documentation which extensively covers features of IPython.



Tutorial Parts
==============

.. toctree::
   :maxdepth: 2

   tutorial1_start
   tutorial2_datasets
