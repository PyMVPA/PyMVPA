.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:

.. _chap_modref:

****************
Module Reference
****************

This module reference extends the manual with a comprehensive overview of the
currently available functionality, that is built into PyMVPA. However, instead
of a full list including every single line of the PyMVPA code base, this
reference limits itself to the relevant pieces of the application programming
interface (API) that are of particular interest to users of this framework.

Each module in the package is documented by a general summary of its
purpose and the list of classes and functions it provides.

Entry Point
===========

.. autosummary::
   :toctree: generated

   mvpa


.. the rest of the modules are relative to the top-level
.. currentmodule:: mvpa

Basic Facilities
=================

.. autosummary::
   :toctree: generated

   base
   base.collections
   base.config
   base.dochelpers
   base.externals
   base.hdf5
   base.info
   base.report
   base.types
   base.verbosity


Datasets: Input, Output, Storage and Preprocessing
==================================================

.. autosummary::
   :toctree: generated

   base.dataset
   datasets.base
   datasets.mri
   datasets.eventrelated
   datasets.eep
   datasets.miscfx
   datasets.splitters


Mappers: Data Transformations
=============================

.. autosummary::
   :toctree: generated

   mappers
   mappers.base
   mappers.flatten
   mappers.boxcar
   mappers.mdp_adaptor
   mappers.procrustean
   mappers.projection
   mappers.prototype
   mappers.fx
   mappers.som
   mappers.lle
   mappers.svd
   mappers.wavelet
   mappers.detrend
   mappers.zscore


Classifiers and Errors
======================

.. autosummary::
   :toctree: generated

   clfs.base
   clfs.meta
   clfs.blr
   clfs.enet
   clfs.glmnet
   clfs.gnb
   clfs.gpr
   clfs.knn
   clfs.lars
   clfs.plr
   clfs.ridge
   clfs.smlr
   clfs.svm
   clfs.sg
   clfs.libsvmc
   clfs.distance
   clfs.similarity
   clfs.stats
   clfs.transerror
   clfs.warehouse


Kernels
-------

.. autosummary::
   :toctree: generated

   kernels
   kernels.base
   kernels.libsvm
   kernels.np
   kernels.sg


Measures: Searchlights and Sensitivties
=======================================

.. autosummary::
   :toctree: generated

   measures.base
   measures.anova
   measures.corrcoef
   measures.corrstability
   measures.ds
   measures.glm
   measures.irelief
   measures.noiseperturbation
   measures.searchlight


Feature Selection
=================

.. autosummary::
   :toctree: generated

   featsel.base
   featsel.ifs
   featsel.rfe
   featsel.helpers


Additional Algorithms
=====================

.. autosummary::
   :toctree: generated

   algorithms.cvtranserror
   algorithms.hyperalignment


Miscellaneous
=============

.. autosummary::
   :toctree: generated

   atlases
   misc.args
   misc.attributes
   misc.attrmap
   misc.cmdline
   misc.data_generators
   misc.errorfx
   misc.exceptions
   misc.fx
   misc.neighborhood
   misc.param
   misc.sampleslookup
   misc.state
   misc.stats
   misc.support
   misc.transformers
   misc.vproperty


Basic Plotting Utilities
------------------------

.. autosummary::
   :toctree: generated

   misc.plot
   misc.plot.base
   misc.plot.erp
   misc.plot.lightbox
   misc.plot.topo


3rd-party Interfaces
--------------------

.. autosummary::
   :toctree: generated

   misc.bv
   misc.bv.base
   misc.fsl
   misc.fsl.base
   misc.fsl.flobs
   misc.fsl.melodic
   misc.io
   misc.io.base
   misc.io.hamster
   misc.io.meg

