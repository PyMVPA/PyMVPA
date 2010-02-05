.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
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

  Classifier
    A model that maps an arbitrary feature space into a discrete set of
    labels.

  Cross-validation
    A technique to assess the :term:`generalization` of the constructed model
    by the analysis of accuracy of the model predictions on presumably
    independent dataset.

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

  Dataset attribute
    An arbitrary auxiliary information that is stored in a dataset.

  Decoding
    This term is usually used to refer to the application of machine learning or
    pattern recognition techniques to brainimaging datasets, and therefore is
    another term for :term:`MVPA`. Sometimes also 'brain-reading' is used as
    another alternative.

  Epoch
    Sometimes used to refer to a group of successively acquired samples, and,
    thus, related to a :term:`chunk`.

  Examplar
    Another term for :term:`sample`.

  Feature
    A variable that represents a dimension in a :term:`dataset`. This might be
    the output of a single sensor, such as a voxel, or a refined measure
    reflecting specific aspect of data, such as a specific spectral
    component.

  Feature attribute
    Analogous to a :term:`sample attribute`, this is a per-feature vector of
    auxiliary information that is stored in a dataset.

  Feature Selection
    A technique that targets detection of features relevant to a given
    problem, so that their selection improves generalization of the
    constructed model.

  fMRI
    This abbrevation stands for *functional magnetic resonance imaging*.

  Generalization
    An ability of a model to perform reliably well on any novel data in
    the given domain.

  Label
    A label is a special case of a :term:`target` for specifying descrete
    categories of :term:`samples` in a classification analyses.

  Machine Learning
    A field of Computer Science that aims at constructing methods, such
    as classifiers, to integrate available knowledge extracted from
    existing data.

  MVPA
    This term originally stems from the authors of the Matlab MVPA toolbox, and
    in that context stands for *multi-voxel pattern analysis* (see :ref:`Norman
    et al., 2006 <NPD+06>`). PyMVPA obviously adopted this acronym. However, as
    PyMVPA is explicitly designed to operate on non-fMRI data as well, the
    'voxel' term is not appropriate and therefore MVPA in this context stands
    for the more general term *multivariate pattern analysis*.

  Neural Data Modality
    A reflection of neural activity collected using some available
    instrumental method (\eg EEG, :term:`fMRI`).

  Processing object
   Most objects dealing with data are implemented as processing objects. Such
   objects are instantiated *once*, with all appropriate parameters configured
   as desired. When created, they can be used multiple time by simply calling
   them with new data.

  Sample
    A sample a vector with observations for all :term:`feature` variables.

  Sample attribute
    A per-sample vector of auxiliary information that is stored in a
    dataset. This could, for example, be a vector identifying specific
    :term:`chunk`\ s of samples.

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

  Statistical Learning
    A field of science related to :term:`machine learning` which aims at
    exploiting statistical properties of data to construct robust models, and to
    assess their convergence and :term:`generalization` performances.

  Target
    A target associates each :term:`sample` in the :term:`dataset` with
    a certain category, experimental condition or, in case of a regression
    problem, with some metric variable. The target defines the model
    for a supervised learning algorithm. The targets also provide the "ground
    truth" for assessing the model's generalization performance.

  Time-compression
    This usually refers to the :term:`block-averaging` of samples from a
    block-design fMRI dataset.

  Weight Vector
    See :term:`sensitivity`.
