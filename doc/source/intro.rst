.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_intro:

************
Introduction
************

.. index:: MVPA

PyMVPA is a Python_ module intended to ease pattern classification
analysis of large datasets. It provides high-level abstraction of typical
processing steps and a number of implementations of some popular algorithms.
While it is not limited to neuroimaging data it is eminently suited for such
datasets. PyMVPA is truly free software (in every respect) and additionally
requires nothing but free software to run. Theoretically PyMVPA should run
on anything that can run a Python_ interpreter, although the proof is yet to
come.

PyMVPA stands for *Multivariate Pattern Analysis* in Python_.


What this Manual is NOT
=======================

.. index:: textbook, review, API reference, examples

This manual does not make an attempt to be a comprehensive introduction into
machine learning *theory*. There is a wealth of high-quality text books about
this field available.  Two very good examples are: `Pattern Recognition and Machine
Learning`_ by `Christopher M. Bishop`_, and :ref:`The Elements of Statistical Learning:
Data Mining, Inference, and Prediction <HTF09>` by `Trevor Hastie`_, `Robert
Tibshirani`_, and `Jerome Friedman`_ (PDF was generously made available online_
free of charge).

There is a growing number of introductory papers about the application of
machine learning algorithms to (f)MRI data. A very high-level overview about
the basic principles is available in :ref:`Mur et al. (2009) <MBK09>`. A more
detailed tutorial covering a wide variety of aspects is provided in
:ref:`Pereira et al. (2009) <PMB09>`. Two reviews by :ref:`Norman et al.
(2006) <NPD+06>` and :ref:`Haynes and Rees (2006) <HR06>` give a broad overview
about the literature.

This manual also does not describe every technical bit and piece of the PyMVPA
package, but is instead focused on the user perspective. Developers should have
a look at the `API documentation`_, which is a detailed, comprehensive and
up-to-date description of the whole package. Users looking for an overview of
the public programming interface of the framework are referred to the
:ref:`chap_modref`. The :ref:`chap_modref` is similar to the API reference, but hides
overly technical information, which are only relevant for people intending to
extend the framework by adding more functionality.

More examples and usage patterns extending the ones described here can be taken
from the examples shipped with the PyMVPA source distribution (`doc/examples/`;
some of them are also available in the :ref:`chap_examples` chapter of this manual)
or even the unit test battery, also part of the source distribution (in the
`tests/` directory).

.. _API Documentation: api/index.html
.. _Christopher M. Bishop: http://research.microsoft.com/~cmbishop/
.. _Pattern Recognition and Machine Learning: http://research.microsoft.com/~cmbishop/PRML

.. _online:
.. _The Elements of Statistical Learning\: Data Mining, Inference, and Prediction: http://www-stat.stanford.edu/~tibs/ElemStatLearn/
.. _Trevor Hastie: http://www-stat.stanford.edu/~hastie/
.. _Robert Tibshirani: http://www-stat.stanford.edu/~tibs/
.. _Jerome Friedman: http://www-stat.stanford.edu/~jhf/

.. _history:

.. index:: history, MVPA toolbox for Matlab, license, free software

A bit of History
================

The roots of PyMVPA date back to early 2005. At that time it was a C++ library
(no Python_ yet) developed by Michael Hanke and Sebastian Krüger, intended to
make it easy to apply artificial neural networks to pattern recognition
problems.

During a visit to `Princeton University`_ in spring 2005, Michael Hanke
was introduced to the `MVPA toolbox`_ for `Matlab
<http://buchholz.hs-bremen.de/aes/aes_matlab.gif>`_, which had several
advantages over a C++ library. Most importantly it was easier to use. While a
user of a C++ library is forced to write a significant amount of front-end
code, users of the MVPA toolbox could simply load their data and start
analyzing it, providing a common interface to functions drawn from a variety
of libraries.

.. _Princeton University: http://www.princeton.edu
.. _MVPA toolbox: https://code.google.com/p/princeton-mvpa-toolbox/

However, there are some disadvantages when writing a toolbox in Matlab. While
users in general benefit from the powers of Matlab, they are at the same time
bound to the goodwill of a commercial company. That this is indeed a problem
becomes obvious when one considers the time when the vendor of Matlab was not
willing to support the Mac platform. Therefore even if the MVPA toolbox is
`GPL-licensed`_ it cannot fully benefit from the enormous advantages of the
free software development model environment (free as in free speech, not only
free beer).

.. _GPL-licensed: http://www.gnu.org/copyleft/gpl.html

For these reasons, Michael thought that a successor to the C++ library
should remain truly free software, remain fully object-oriented (in contrast
to the MVPA toolbox), but should be at least as easy to use and extensible
as the MVPA toolbox.

After evaluating some possibilities Michael decided that `Python`_ is the most
promising candidate that was fully capable of fulfilling the intended
development goal. Python is a very powerful language that magically combines
the possibility to write really fast code and a simplicity that allows one to
learn the basic concepts within a few days.

.. index:: RPy, PyMatlab

One of the major advantages of Python is the availability of a huge amount of
so called *modules*. Modules can include extensions written in a hardcore
language like C (or even FORTRAN) and therefore allow one to incorporate
high-performance code without having to leave the Python
environment. Additionally some Python modules even provide links to other
toolkits. For example `RPy`_ allows to use the full functionality of R_ from
inside Python. Even Matlab can be used via some Python modules (see PyMatlab_
for an example).

.. _RPy: http://rpy.sourceforge.net/
.. _R: http://www.r-project.org
.. _PyMatlab: http://code.google.com/p/pymatlab/

After the decision for Python was made, Michael started development with a
simple k-Nearest-Neighbor classifier and a cross-validation class. Using
the mighty NumPy_ package made it easy to support data of any dimensionality.
Therefore PyMVPA can easily be used with 4d fMRI dataset, but equally well
with EEG/MEG data (3d) or even non-neuroimaging datasets.

.. index:: NIfTI

By September 2007 PyMVPA included support for reading and writing datasets
from and to the `NIfTI format`_, kNN and Support Vector Machine classifiers,
as well as several analysis algorithms (e.g. searchlight and incremental
feature search).

.. _NIfTI format: http://nifti.nimh.nih.gov/

During another visit in Princeton in October 2007 Michael met with `Yaroslav
Halchenko`_ and `Per B. Sederberg`_. That incident and the following
discussions and hacking sessions of Michael and Yaroslav lead to a major
refactoring of the PyMVPA codebase, making it much more flexible/extensible,
faster and easier than it has ever been before.

.. _Yaroslav Halchenko: http://www.onerussian.com/
.. _Per B. Sederberg: http://www.princeton.edu/~persed/


.. index:: citation, PyMVPA poster


How to cite PyMVPA
==================

.. include:: howtocite.txt


.. include:: contributors.txt

.. include:: link_names.txt
