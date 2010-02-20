.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:


PyMVPA is a Python_ package intended to ease statistical learning analyses of
large datasets. It offers an extensible framework with a high-level interface
to a broad range of algorithms for classification, regression, feature
selection, data import and export. While it is not limited to the neuroimaging
domain, it is eminently suited for such datasets. PyMVPA is truly free software
(in every respect) and additionally requires nothing but free-software to run.

PyMVPA stands for **M**\ ulti\ **V**\ ariate **P**\ attern **A**\ nalysis
(:term:`MVPA`) in **Py**\ thon.

.. _Python: http://www.python.org



.. raw:: html

 <table style="border-style:none;overflow:scroll">
  <tr>
   <td>
    <a href="download.html">
     <img alt="Download" src="_static/download_icon.jpg" title="Download and Installation" />
    </a>
   </td>
   <td>
    <a href="tutorial.html">
     <img alt="Tutorial" src="_static/tutorial_icon.jpg" title="Tutorial for Beginners" />
    </a>
   </td>
   <td>
    <a href="docoverview.html">
     <img alt="Documentation" src="_static/documentation_icon.jpg" title="Documentation" />
    </a>
   </td>
   <td>
    <a href="support.html">
     <img alt="Support" src="_static/support_icon.jpg" title="Getting Support" />
    </a>
   </td>
  </tr>
  <tr>
    <td style="text-align:center"><a href="download.html">Download</a></td>
    <td style="text-align:center"><a href="tutorial.html">Tutorial</a></td>
    <td style="text-align:center"><a href="docoverview.html">Documentation</a></td>
    <td style="text-align:center"><a href="support">Support</a></td>
  </tr>
 </table>


News
====

.. raw:: html

 <script src="http://widgets.twimg.com/j/2/widget.js"></script>
 <script>
 new TWTR.Widget({
   version: 2,
   type: 'profile',
   rpp: 4,
   interval: 6000,
   width: 'auto',
   height: 300,
   theme: {
     shell: {
       background: '#e6e6e6',
       color: '#000000'
     },
     tweets: {
       background: '#ffffff',
       color: '#525052',
       links: '#993e4d'
     }
   },
   features: {
     scrollbar: false,
     loop: false,
     live: false,
     hashtags: true,
     timestamp: true,
     avatars: true,
     behavior: 'all'
   }
 }).render().setUser('pymvpa').start();
 </script>


License
=======

PyMVPA is free-software (beer and speech) and covered by the `MIT License`_.
This applies to all source code, documentation, examples and snippets inside
the source distribution (including this website). Please see the
:ref:`appendix of the manual <chap_license>` for the copyright statement and the
full text of the license.

.. _MIT License: http://www.opensource.org/licenses/mit-license.php
.. _appendix of the manual: manual.html#license



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

* MDP_
  Python data processing framework. MDP_ provides various algorithms.
  *PyMVPA makes use of MDP's PCA and ICA implementations.*

* `MVPA Toolbox`_: Matlab-based toolbox to facilitate multi-voxel pattern
  analysis of fMRI neuroimaging data.

* NiPy_: Project with growing functionality to analyze brain imaging data. NiPy_
  is heavily connected to SciPy and lots of functionality developed within
  NiPy becomes part of SciPy.

* OpenMEEG_: Software package for low-frequency bio-electromagnetism including
  the EEG/MEG forward and inverse problems. OpenMEEG includes Python bindings.

* Orange_: Powerful general-purpose data mining software. Orange also has Python
  bindings.

* PROBID_: Matlab-based GUI pattern recognition toolbox for MRI data.

* `PyMGH/PyFSIO`_: Python IO library to for FreeSurfer's `.mgh` data format.

* PyML_: PyML is an interactive object oriented framework for machine learning
  written in Python. PyML focuses on SVMs and other kernel methods.

* PyNIfTI_: Read and write NIfTI images from within Python.
  *PyMVPA uses PyNIfTI to access MRI datasets.*

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
.. _PROBID: http://www.brainmap.co.uk/probid.htm
.. _PyMGH/PyFSIO: http://code.google.com/p/pyfsio
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti
.. _OpenMEEG: http://www-sop.inria.fr/odyssee/software/OpenMEEG
