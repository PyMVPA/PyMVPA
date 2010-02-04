.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
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
===========

Source code tarballs of PyMVPA releases are available from the `download
area`_. Alternatively, one can also download a tarball of the latest
development snapshot_ (i.e. the current state of the *master* branch of the
PyMVPA source code repository).

To get access to both the full PyMVPA history and the latest
development code, the PyMVPA Git_ repository is publicly available. To view the
repository, please point your webbrowser to gitweb:
http://git.debian.org/?p=pkg-exppsy/pymvpa.git

To clone (aka checkout) the PyMVPA repository simply do:

::

  git clone git://git.debian.org/git/pkg-exppsy/pymvpa.git

After a short while you will have a `pymvpa` directory below your current
working directory, that contains the PyMVPA repository.

More detailed instructions on :ref:`installation requirements <requirements>`
and on how to :ref:`build PyMVPA from source <buildfromsource>` are provided
in the manual.


.. _download area: http://alioth.debian.org/frs/?group_id=30954
.. _Git: http://git.or.cz/
.. _snapshot:  http://git.debian.org/?p=pkg-exppsy/pymvpa.git;a=snapshot;h=refs/heads/master;sf=tgz
