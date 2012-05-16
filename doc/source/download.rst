.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. index:: Download
.. _chap_download:

****************
Obtaining PyMVPA
****************

Binary packages
===============

Binary packages are available for:

* Debian and Ubuntu (:ref:`installation instructions <install_debian>`)
    `Official Debian package`_ (``python-mvpa2``).
    Backports for Debian and Ubuntu releases are available from the
    `NeuroDebian project`_.

..
  * RPM-based GNU/Linux distributions (:ref:`installation instructions <install_rpm>`)
      RPM packages are provided through the `OpenSUSE Build Service`_. It offers
      package repositories and `1-click-installations`_. Currently, we offer RPMs
      for:
  
      * CentOS_ 5
      * Fedora_ 9 (and later releases)
      * OpenSUSE_ 11.0 (and later releases)
      * RedHat_ Enterprise Linux 5
  
  * MacOS X (:ref:`installation instructions <install_macos>`)
      PyMVPA is available from the MacPorts_ framework (at the moment only
      previous 0.4 series.  .

* Windows (:ref:`installation instructions <install_win>`) An installer for
    Python 2.6 is available from the `github download area`_.  Also installers
    of PyMVPA for Python 2.5, 2.6 and 2.7 are available from the `Unofficial
    Windows binaries collection of Christoph Gohlke`_.

If there are no binary packages for your particular operating system or
platform, you have two opportunities.  You could compile PyMVPA (which is not
strictly necessary anyways unless you need to use SVM or SMLR). The manual
contains :ref:`instructions <buildfromsource>` to build PyMVPA in various
environments (Windows, Fedora, OpenSuse, ...).  Alternatively you could use
`NeuroDebian Virtual Appliance`_ which allows in a matter of minutes to get a
full featured (Neuro)Debian GNU/Linux distribution running **on top** of your
system, thus providing easy access to over 30,000 software products many of
which are related to machine learning and neuroscience.

.. _NeuroDebian Virtual Appliance: http://neuro.debian.net/vm.html
.. _MacPorts: http://www.macports.org/ports.php?by=name&substr=pymvpa
.. _official Debian package: http://packages.debian.org/python-mvpa2
.. _NeuroDebian project: http://neuro.debian.net
.. _OpenSUSE: http://download.opensuse.org/repositories/home:/hankem:/suse/
.. _CentOS: http://download.opensuse.org/repositories/home:/hankem:/rh5/
.. _Fedora: http://download.opensuse.org/repositories/home:/hankem:/rh5/
.. _Redhat: http://download.opensuse.org/repositories/home:/hankem:/rh5/
.. _1-click-installations: http://software.opensuse.org/search?baseproject=ALL&p=1&q=python-mvpa
.. _OpenSUSE Build Service: https://build.opensuse.org/


Source code
===========

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

  git clone git://github.com/PyMVPA/PyMVPA.git

After a short while you will have a ``PyMVPA`` directory below your current
working directory, that contains the PyMVPA repository.

More detailed instructions on :ref:`installation requirements <requirements>`
and on how to :ref:`build PyMVPA from source <buildfromsource>` are provided
in the manual.


.. _download area: http://alioth.debian.org/frs/?group_id=30954
.. _github download area: https://github.com/PyMVPA/PyMVPA/downloads
.. _Git: http://git.or.cz/
.. _snapshot: http://github.com/PyMVPA/PyMVPA/archives/master
.. _Unofficial Windows binaries collection of Christoph Gohlke: http://www.lfd.uci.edu/~gohlke/pythonlibs/
