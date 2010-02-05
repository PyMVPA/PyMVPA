.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
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

.. _Python: http://www.python.org


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
:ref:`Pereira et al. (in press) <PMB+IP>`. Two reviews by :ref:`Norman et al.
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
.. _MVPA toolbox: http://www.csbmb.princeton.edu/mvpa/

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

.. _NumPy: http://numpy.scipy.org/
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


Authors & Contributors
======================

.. include:: authors.rst


How to cite PyMVPA
==================

Below is a list of all publications about PyMVPA that have been published so
far (in chronological order). If you use PyMVPA in your research please cite
the one that matches best.

Peer-reviewed publications
--------------------------

Hanke, M., Halchenko, Y. O., Haxby, J. V., and Pollmann, S. (accepted) *Statistical learning analysis in neuroscience: aiming for transparency*. Frontiers in Neuroscience.
  Focused review article emphasizing the role of transparency to facilitate
  adoption and evaluation of statistical learning techniques in neuroimaging
  research.


Hanke, M., Halchenko, Y. O., Sederberg, P. B., Olivetti, E., Fründ, I., Rieger, J. W., Herrmann, C. S., Haxby, J. V., Hanson, S. J. and Pollmann, S. (2009) `PyMVPA\: a unifying approach to the analysis of neuroscientific data`_. Frontiers in Neuroinformatics, 3:3.
  Demonstration of PyMVPA capabilities concerning multi-modal or
  modality-agnostic data analysis.

.. _PyMVPA\: a unifying approach to the analysis of neuroscientific data: http://dx.doi.org/10.3389/neuro.11.003.2009


Hanke, M., Halchenko, Y. O., Sederberg, P. B., Hanson, S. J., Haxby, J. V. & Pollmann, S. (2009). `PyMVPA: A Python toolbox for multivariate pattern analysis of fMRI data`_. Neuroinformatics, 7, 37-53.
  First paper introducing fMRI data analysis with PyMVPA.

.. _PyMVPA\: A Python toolbox for multivariate pattern analysis of fMRI data: http://dx.doi.org/10.1007/s12021-008-9041-y


Posters
-------

Hanke, M., Halchenko, Y. O., Sederberg, P. B., Hanson, S. J., Haxby, J. V. & Pollmann, S. (2008). `PyMVPA: A Python toolbox for machine-learning based data analysis.`_
  Poster emphasizing PyMVPA's capabilities concerning multi-modal data analysis
  at the annual meeting of the Society for Neuroscience, Washington, 2008.

Hanke, M., Halchenko, Y. O., Sederberg, P. B., Hanson, S. J., Haxby, J. V. & Pollmann, S. (2008). `PyMVPA: A Python toolbox for classifier-based data analysis.`_
  First presentation of PyMVPA at the conference *Psychologie und Gehirn*
  [Psychology and Brain], Magdeburg_, 2008. This poster received the poster
  prize of the *German Society for Psychophysiology and its Application*.

.. _PyMVPA\: A Python toolbox for classifier-based data analysis.: http://www.pymvpa.org/files/PyMVPA_PuG2008.pdf
.. _PyMVPA\: A Python toolbox for machine-learning based data analysis.: http://www.pymvpa.org/files/PyMVPA_SfN2008.pdf
.. _Magdeburg: http://www.magdeburg.de/



Acknowledgements
================

We are greatful to the developers and contributers of NumPy_, SciPy_ and
IPython_ for providing an excellent Python-based computing environment.

Additionally, as PyMVPA makes use of a lot of external software packages (e.g.
classifier implementations), we want to acknowledge the authors of the
respective tools and libraries (e.g. LIBSVM_ or Shogun_) and thank them for
developing their packages as free and open source software.

Finally, we would like to express our acknowledgements to the `Debian project`_
for providing us with hosting facilities for mailing lists and source code
repositories. But most of all for developing the *universal operating system*.

.. Please add some notes when you think that you should give credits to someone
   that enables or motivates you to work on PyMVPA ;-)

.. _Debian project: http://www.debian.org
.. _SciPy: http://www.scipy.org/
.. _Shogun: http://www.shogun-toolbox.org
.. _IPython: http://ipython.scipy.org
.. _LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
