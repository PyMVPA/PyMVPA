.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_glossary:

********
Glossary
********

The literature concerning the application of multivariate pattern analysis
procedures to neuro-scientific datasets contains a lot of specific terms to
refer to procedures or types of data, that are of particular importance.
Unfortunately, sometimes various terms refer to the same construct and even
worse these terms do not necessarily match the terminology used in the machine
learning literature. The following glossary is an attempt to map the various
terms found in the literature to the terminology used in this manual.


.. glossary::

  Block-averaging
    Averaging all samples recorded during a block of continuous stimulation in a
    block-design fMRI experiment. The rationale behind this technique is, that
    a averaging might lead to an improved signal-to-noise ratio. However,
    averaging further decreases the number of samples in a dataset, which is
    already very low in typical fMRI datasets, especially in comparison to the
    number of features/voxels. Block-averaging might nevertheless improve the
    classifier performance, *if* it indeed improves signal-to-noise *and* the
    respective classifier benefits more from few high-quality samples than
    from a larger set of lower-quality samples.


  Chunk
    A chunk is a group of samples. In PyMVPA chunks define *independent* groups
    of samples (note: the groups are independent from each other, not the
    samples in each particular group). This information is important in the
    context of a cross-validation procedure, as it is required to measure the
    classifier performance on independent test datasets to be able to compute
    unbiased generalization estimates. This is of particular importance in the
    case of fMRI data, where two successively recorded volumes cannot be
    considered as independent measurements. This is due to the significant
    temporal forward contamination of the hemodynamic response whose correlate
    is measured by the MR scanner.


  Dataset
    In PyMVPA a dataset is the combination of samples, their ...


  Decoding
    This term is usually used to refer to the application of machine learning or
    pattern recognition techniques to brainimaging datasets, and therefore is
    another term for :term:`MVPA`. Sometimes also 'brain-reading' is used as
    another alternative.


  Epoch
    Sometimes used to refer to a group of successively acquired samples, and,
    thus, related to a :term:`chunk`.


  Example
    Another term for :term:`sample`.


  Feature
    This is a name for a variable in the :term:`dataset`.


  fMRI
    This abbrevation stands for *functional magnetic resonance imaging*.


  Label
    A label associates each :term:`sample` in the :term:`dataset` with
    a certain category, experimental condition or, in case of a regression
    problem, with some metric variable. The label therefore defines the model
    that a classifier has to learn. The labels also provide the "true"
    model value when computing classifier errors.


  MVPA
    This term originally stems from the authors of the Matlab MVPA toolbox, and
    in that context stands for *multi-voxel pattern analysis* (see :ref:`Norman
    et al., 2006 <NPD+06>`). PyMVPA obviously adopted this acronym. However, as
    PyMVPA is explicitly designed to operate on non-fMRI data as well, the
    'voxel' term is not appropriate and therefore MVPA in this context stands
    for the more general term *multivariate pattern analysis*.


  Processing object
   Most objects dealing with data are implemented as processing objects. Such
   objects are instantiated *once*, with all appropriate parameters configured
   as desired. When created, they can be used multiple time by simply calling
   them with new data.


  Sample
    A sample a vector with observations for all :term:`feature` variables.


  Sensitivity
    The sensitivity is a score assigned to a particular :term:`feature` with
    respect to its impact on a classifier's decision. The sensitivity is
    often available from a classifier's :term:`weight vector`. There are some
    more scores which are similar to a sensitivity in terms of indicating the
    "importance" of a particular feature -- examples are a univariate
    :ref:`anova` score or a :ref:`noise_perturbation` measure.



  Sensitivity Map
    A vector of several sensitivity scores -- one for each feature in a
    dataset.


  Spatial Discrimination Map (SDM)
    This is another term for a :term:`sensitivity map`, used in e.g.
    :ref:`Wang et al. (2007) <WCW+07>`.


  Statistical Discrimination Map (SDM)
    This is another term for a :term:`sensitivity map`, used in e.g.
    :ref:`Sato et al. (2008) <SMM+08>`, where instead of raw sensitivity
    significance testing result is assigned.


  Time-compression
    This usually refers to the :term:`block-averaging` of samples from a
    block-design fMRI dataset.


  Weight Vector
    See :term:`sensitivity`.
