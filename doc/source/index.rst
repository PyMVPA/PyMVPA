.. -*- mode: rst -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:


PyMVPA is a Python_ package intended to ease statistical learning analyses of
large datasets. It offers an extensible framework with a high-level interface
to a broad range of algorithms for classification, regression, feature
selection, data import and export. It is designed to integrate well with
related software packages, such as scikit-learn_, and MDP_. While it is not
limited to the neuroimaging domain, it is eminently suited for such datasets.
PyMVPA is free software and requires nothing but free-software to run.

PyMVPA stands for **M**\ ulti\ **V**\ ariate **P**\ attern **A**\ nalysis
(:term:`MVPA`) in **Py**\ thon.

.. _Python: http://www.python.org

.. raw:: html

 <table style="border-style:none;overflow:scroll">
  <tr>
   <td>
    <a href="installation.html">
     <img alt="Installation" src="_static/download_icon.jpg" title="Download and Installation" />
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
    <td style="text-align:center"><a href="installation.html">Installation</a></td>
    <td style="text-align:center"><a href="tutorial.html">Tutorial</a></td>
    <td style="text-align:center"><a href="docoverview.html">Documentation</a></td>
    <td style="text-align:center"><a href="support.html">Support</a></td>
  </tr>
 </table>

News
====

.. raw:: html

 <a class="twitter-timeline" href="https://twitter.com/pymvpa"
    data-widget-id="434978943293083648"
    data-link-color="#820430"
    height="150px"
    data-show-replies="false"
    data-chrome="noheader nofooter transparent">Tweets by @pymvpa</a>
 <script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+"://platform.twitter.com/widgets.js";fjs.parentNode.insertBefore(js,fjs);}}(document,"script","twitter-wjs");</script>

.. _twitter: http://twitter.com/pymvpa


Contributing
============

We welcome all kinds of contributions, and you do **not need to be a
programmer** to contribute! If you have some feature in mind that is missing,
some example use case that you want to share, you spotted a typo in the
documentation, or you have an idea how to improve the user experience all
together -- do not hesitate and :ref:`contact us <chap_support>`. We will then
figure out how your contribution can be best incorporated. Any contributor will
be acknowledged and will appear in the list of people who have helped to
develop PyMVPA on the front-page of the `pymvpa.org <http://www.pymvpa.org>`_.

License
=======

PyMVPA is free-software (beer and speech) and covered by the `MIT License`_.
This applies to all source code, documentation, examples and snippets inside
the source distribution (including this website). Please see the
:ref:`appendix of the manual <chap_license>` for the copyright statement and the
full text of the license.

.. _MIT License: http://www.opensource.org/licenses/mit-license.php
.. _appendix of the manual: manual.html#license


How to cite PyMVPA
==================

.. include:: howtocite.txt

.. contributors.txt also would include link_names.txt
.. include:: contributors.txt


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

* PROBID_: Matlab-based GUI pattern recognition toolbox for MRI data.

* `PyMGH/PyFSIO`_: Python IO library to for FreeSurfer's `.mgh` data format.

* PyML_: Interactive object oriented framework for machine learning
  written in Python. PyML focuses on SVMs and other kernel methods.

* PyNIfTI_: Read and write NIfTI images from within Python.
  *PyMVPA uses PyNIfTI to access MRI datasets.*

* `scikit-learn`_: Python module integrating classic machine learning
  algorithms in the tightly-knit world of scientific Python packages.

* Shogun_: Comprehensive machine learning toolbox with bindings to various
  programming languages.
  *PyMVPA can optionally use implementations of Support Vector Machines from
  Shogun.*

.. toctree::
   :hidden:

   manual
   mvpa_guidelines
   release_notes_0.5
   release_notes_0.6
   workshops/2009-fall
