.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:


PyMVPA is a Python_ module intended to ease pattern classification analyses of
large datasets. In the neuroimaging contexts such analysis techniques are also
known as :term:`decoding` or :term:`MVPA` analysis. PyMVPA provides high-level
abstraction of typical processing steps and a number of implementations of some
popular algorithms.  While it is not limited to the neuroimaging domain, it is
eminently suited for such datasets.  PyMVPA is truly free software (in every
respect) and additionally requires nothing but free-software to run.

.. _Python: http://www.python.org

PyMVPA stands for **M**\ ulti\ **V**\ ariate **P**\ attern **A**\ nalysis
(:term:`MVPA`) in **Py**\ thon.

PyMVPA is developed inside the `Debian Experimental Psychology Project`_. This
website, the source code repository and download services are hosted on 
Alioth_, a service that is kindly provided by the `Debian project`_.

.. _Debian Experimental Psychology Project: http://pkg-exppsy.alioth.debian.org
.. _Alioth: http://alioth.debian.org
.. _Debian project: http://www.debian.org


News
====

PyMVPA 0.4.5 is out [01 Oct 2010]
  This release brings a set of bug fixes. See the
  :ref:`changelog <chap_changelog>` for details.

.. raw:: html

   <dt></dt><dd><script type="text/javascript" src="http://www.ohloh.net/p/16363/widgets/project_users_logo.js"></script></dd>

PyMVPA Extravaganza 2009 at Dartmouth College [30th Nov -- 4th Dec]
  :ref:`Read more <chap_workshop_2009fall>` about the topics and achievements.


First publication from outside the PyMVPA team employing PyMVPA [19 Jul 2009]
  :ref:`Sun et al. (2009) <SET+09>`: *Elucidating an MRI-Based Neuroanatomic
  Biomarker for Psychosis: Classification Analysis Using Probabilistic Brain
  Atlas and Machine Learning Algorithms.*

.. _pydocweb: https://code.launchpad.net/~pauli-virtanen/scipy/pydocweb

Documentation
=============

For users
---------

* :ref:`User Documentation <contents>` [PDF-manual_] (**the** documentation).
* :ref:`Installation Instructions <chap_installation>`
* :ref:`FAQ <chap_faq>` (short answers to common problems)
* :ref:`Module Reference <chap_modref>` (user-oriented reference)
* :ref:`Bibliography <chap_references>` (references to interesting literature)
* :ref:`Development Changelog <chap_changelog>` [:ref:`Movie version
  <chap_code_swarm>`] (see what has changed)

.. _PDF-manual: PyMVPA-Manual.pdf

* :ref:`exampledata` (single subject dataset from :ref:`Haxby et al., 2001
  <HGF+01>`)

.. comment to separate the two lists

* :ref:`genindex` (access by keywords)
* :ref:`search` (online and offline full-text search)


For developers
--------------

* :ref:`Developer Guidelines <chap_devguide>` [PDF-guide_] (information for people
  contributing code)
* `API Reference`_ (comprehensive and up-to-date information about the details
  of the implementation)

.. _PDF-guide: PyMVPA-DevGuide.pdf
.. _API Reference: api/index.html


License
=======

PyMVPA is free-software (beer and speech) and covered by the `MIT License`_.
This applies to all source code, documentation, examples and snippets inside
the source distribution (including this website). Please see the
:ref:`appendix of the manual <chap_license>` for the copyright statement and the
full text of the license.

.. _MIT License: http://www.opensource.org/licenses/mit-license.php
.. _appendix of the manual: manual.html#license



Download
========

Binary packages
---------------

Binary packages are available for:

* Debian and Ubuntu (:ref:`installation instructions <install_debian>`)
    PyMVPA is an `official Debian package`_ (`python-mvpa`).
    Additionally, backports for some Debian and Ubuntu releases are also
    available. Please read the `package repository instructions`_ to learn
    about how to obtain them.

* RPM-based GNU/Linux distributions (:ref:`installation instructions <install_rpm>`)
    RPM packages are provided through the `OpenSUSE Build Service`_. The
    currently supported distributions include: CentOS 5, Fedora 9-12,
    RedHat Enterprise Linux 5, OpenSUSE 11.0 up to 11.2 (but also OpenSUSE
    Factory). The build service supports RPM-package repositories
    (`SUSE-related`_ and `Fedora, Redhat and CentOS-related`_) and
    `1-click-installations`_.

* MacOS X (:ref:`installation instructions <install_macos>`)
    PyMVPA is available from the MacPorts_ framework.

* Windows (:ref:`installation instructions <install_win>`)
    An installer for Python 2.5 is available from the `download area`_.

If there are no binary packages for your particular operating system or
platform, you need to compile your own. The manual contains :ref:`instructions
<buildfromsource>` to build PyMVPA in various environments.

.. _MacPorts: http://www.macports.org
.. _official Debian package: http://packages.debian.org/python-mvpa
.. _package repository instructions: http://neuro.debian.net/#how-to-use-this-repository
.. _SUSE-related: http://download.opensuse.org/repositories/home:/hankem:/suse/
.. _Fedora, Redhat and CentOS-related: http://download.opensuse.org/repositories/home:/hankem:/rh5/
.. _1-click-installations: http://software.opensuse.org/search?baseproject=ALL&p=1&q=python-mvpa
.. _OpenSUSE Build Service: https://build.opensuse.org/


Source code
-----------

Source code tarballs of PyMVPA releases are available from the `download
area`_. Alternatively, one can also download a tarball of the latest
development snapshot_ (i.e. the current state of the *master* branch of the
PyMVPA source code repository).

To get access to both the full PyMVPA history and the latest
development code, the PyMVPA Git_ repository is publicly available. To view the
repository, please point your webbrowser to gitweb:
http://github.com/PyMVPA/PyMVPA

To clone (aka checkout) the PyMVPA repository simply do:

::

  git clone http://github.com/PyMVPA/PyMVPA

After a short while you will have a `PyMVPA` directory below your current
working directory, that contains the PyMVPA repository.

More detailed instructions on :ref:`installation requirements <requirements>`
and on how to :ref:`build PyMVPA from source <buildfromsource>` are provided
in the manual.


.. _download area: http://alioth.debian.org/frs/?group_id=30954
.. _Git: http://git.or.cz/
.. _snapshot:  http://github.com/PyMVPA/PyMVPA/archives/master


Support
=======

If you have problems installing the software or questions about usage,
documentation or something else related to PyMVPA, you can post to the PyMVPA
mailing list (preferred) or contact the authors on IRC:

:Mailing list: pkg-exppsy-pymvpa@lists.alioth.debian.org [subscription_,
               archive_]
:IRC: #exppsy on OTFC/Freenode

All users should subscribe to the mailing list. PyMVPA is still a young project
that is under heavy development. Significant modifications (hopefully
improvements) are very likely to happen frequently. The mailing list is the
preferred way to announce such changes. The mailing list archive can also be
searched using the *mailing list archive search* located in the sidebar of the
PyMVPA home page.

.. _subscription: http://lists.alioth.debian.org/mailman/listinfo/pkg-exppsy-pymvpa
.. _archive: http://lists.alioth.debian.org/pipermail/pkg-exppsy-pymvpa/



Publications
============

.. include:: publications.rst


Authors & Contributors
======================

.. include:: authors.rst


Similar or Related Projects
===========================

.. in alphanumerical order

There are a number other projects with -- in comparison to
PyMVPA -- partially overlapping features or a similar purpose.
Some of their functionality is already available through and
within the PyMVPA framework. *Only* free software projects are listed
here.

* 3dsvm_: AFNI_ plugin to apply support vector machine classifiers to fMRI data.

* Elefant_: Efficient Learning, Large-scale Inference, and Optimization
  Toolkit.  Multi-purpose open source library for machine learning.

* MDP_:
  Python data processing framework. MDP_ provides various algorithms.
  *PyMVPA makes use of MDP's PCA and ICA implementations.*

* `MVPA Toolbox`_: Matlab-based toolbox to facilitate multi-voxel pattern
  analysis of fMRI neuroimaging data.

* NiPy_: Project with growing functionality to analyze brain imaging data. NiPy_
  is heavily connected to SciPy and lots of functionality developed within
  NiPy becomes part of SciPy.

* OpenMEEG_: Software package for low-frequency bio-electromagnetism
  solving forward problems in the field of EEG and MEG.
  OpenMEEG includes Python bindings.

* Orange_: Powerful general-purpose data mining software. Orange also has Python
  bindings.

* `PyMGH/PyFSIO`_: Python IO library to for FreeSurfer's `.mgh` data format.

* PyML_: Interactive object oriented framework for machine learning
  written in Python. PyML focuses on SVMs and other kernel methods.

* PyNIfTI_: Read and write NIfTI images from within Python.
  *PyMVPA uses PyNIfTI to access MRI datasets.*

* `scikits.learn`_: Python module integrating classic machine learning
  algorithms in the tightly-knit world of scientific Python packages.

* Shogun_: Comprehensive machine learning toolbox with bindings to various
  programming languages.
  *PyMVPA can optionally use implementations of Support Vector Machines from
  Shogun.*

.. _3dsvm: http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dsvm.html
.. _AFNI: http://afni.nimh.nih.gov/
.. _Elefant: http://elefant.developer.nicta.com.au
.. _Shogun: http://www.fml.tuebingen.mpg.de/raetsch/projects/shogun
.. _Orange: http://magix.fri.uni-lj.si/orange
.. _PyML: http://pyml.sourceforge.net
.. _MDP: http://mdp-toolkit.sourceforge.net
.. _MVPA Toolbox: http://www.csbmb.princeton.edu/mvpa/
.. _NiPy: http://neuroimaging.scipy.org
.. _PyMGH/PyFSIO: http://code.google.com/p/pyfsio
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti
.. _OpenMEEG: http://openmeeg.gforge.inria.fr
.. _scikits.learn: http://scikit-learn.sourceforge.net
