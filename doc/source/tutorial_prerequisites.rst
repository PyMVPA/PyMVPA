.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_prerequisites:

**********************
Tutorial Prerequisites
**********************

The PyMVPA tutorial assumes some basic knowledge about programming in Python.
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

PyMVPA code is compatible with Python 2.X series (more precisely >= 2.6).
Python 3.x is supported as well, but not as widely used (yet), and many
3rd-party Python modules are still lacking Python 3 support. For now, we
recommend Python 2.7 for production, but Python 2.6 should work equally well.

Any machine which has Python 2.X available can be used for PyMVPA-based
processing (see :ref:`chap_installation>` on how to deploy
PyMVPA on your system). Any GNU/Linux distribution already comes with Python
by default. The Python website offers `installers for Windows and MacOS X`_.

.. _installers for Windows and MacOS X: http://www.python.org/download

However, PyMVPA can make use of many additional software packages to
enhance its functionality. Therefore it is preferable to use a Python
distribution that offers are large variety of scientific Python packages.
For Windows, `Python(x,y)`_ matches these requirements.  For MacOS X, the
MacPorts_ project offers a large variety of Python packages (including
PyMVPA).

.. _Python(x,y): https://code.google.com/p/pythonxy/
.. _MacPorts: http://www.macports.org

The *ideal* environment is, however, the Debian_ operating system. Debian
offers the largest selection of free and open-source software in the world,
and runs on almost any machine. Moreover, the NeuroDebian_ project provides
Debian packages for a number of popular neuroscience software package, such
as AFNI_ and FSL_.

.. _Debian: http://www.debian.org
.. _NeuroDebian: http://neuro.debian.net

For those who just want to quickly try PyMVPA, or do not want to deal with
installing multiple software package we recommend the `NeuroDebian Virtual
Machine`_. This is a virtual Debian installation that can be deployed on Linux,
Windows, and MacOS X in a matter of minutes. It includes many Python packages,
PyMVPA, and other neuroscience software (including AFNI_ and FSL_).

.. _NeuroDebian Virtual Machine: http://neuro.debian.net/vm.html
.. _AFNI: http://afni.nimh.nih.gov/afni
.. _FSL: http://www.fmrib.ox.ac.uk/fsl



Recommended Reading and Viewing
-------------------------------

This section contains a recommended list of useful resources, ranging from
basic Python programming to efficient scientific computing with Python.


Tutorial Introductions Into General Python Programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://en.wikibooks.org/wiki/Non-Programmer's_Tutorial_for_Python_2.6

  Basic from-scratch introduction into Python. This should give you the basics,
  even if you had *no* prior programming experience.

http://swaroopch.com/notes/python/

  From the author:

    The aim is that if all you know about computers is how to save text files,
    then you can learn Python from this book. If you have previous programming
    experience, then you can also learn Python from this book.

  We recommend reading the PDF version that is a lot better formatted.

http://www.diveintopython.net

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

http://showmedo.com/videotutorials/series?name=CnluURUTV

  Video tutorials from Jeff Rush walking you through basic and advanced
  features of IPython.  While doing that he also exposes basic constructs of
  Python, so you might like to watch this video whenever you already have
  basic programming experience with any programming language.

http://ipython.org/documentation.html

  IPython documentation page which references additional materials, such as
  the main IPython documentation which extensively covers features of IPython.

http://ipython.org/notebook.html

  IPython notebook provides an interactive programming environment right in
  your browser.  Our tutorial is available not only as static web-pages but
  also as IPython notebooks, thus making learning and tinkering with code
  convenient and fun.  Familiarizing with principles of ipython notebooks and
  keyboard shortcuts would be beneficial.


Multivariate Analysis of Neuroimaging Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a constantly growing number of interesting articles related to the
field -- visit :ref:`chap_references` for an extended but not exhaustive list
of related publications.  For a quick introduction into the topic read
:ref:`Pereira et. al. 2009 <PMB09>`.  For the generic reference on machine
learning methods we would recommend a great text book :ref:`The Elements of
Statistical Learning: Data Mining, Inference, and Prediction <HTF09>` by
`Trevor Hastie`_, `Robert Tibshirani`_, and `Jerome Friedman`_ , PDF of which
was generously made available online_ free of charge.  For an overview of
recent advances in computational approaches for modeling and decoding of
stimulus and cognitive spaces we recommend video recordings from `Neural
Computation 2011 Workshop at Dartmouth College
<http://haxbylab.dartmouth.edu/meetings/ncworkshop11.html>`_.

.. _online:
.. _The Elements of Statistical Learning\: Data Mining, Inference, and Prediction: http://www-stat.stanford.edu/~tibs/ElemStatLearn/
.. _Trevor Hastie: http://www-stat.stanford.edu/~hastie/
.. _Robert Tibshirani: http://www-stat.stanford.edu/~tibs/
.. _Jerome Friedman: http://www-stat.stanford.edu/~jhf/
