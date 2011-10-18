.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
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
    block-design :term:`fMRI` experiment. The rationale behind this technique is, that
    averaging might lead to an improved signal-to-noise ratio. However,
    averaging further decreases the number of samples in a dataset, which is
    already very low in typical fMRI datasets, especially in comparison to the
    number of features/voxels. Block-averaging might nevertheless improve the
    classifier performance, *if* it indeed improves signal-to-noise *and* the
    respective classifier benefits more from few high-quality samples than
    from a larger set of lower-quality samples.

  Classifier
    A model that maps an arbitrary feature space into a discrete set of
    labels.

  Meta-classifier
    An internal to PyMVPA term to describe a classifier which is usually a
    proxy to the main classifier which it wraps to provide additional data
    preprocessing (e.g. feature selection) before actually training and/or
    testing of the wrapped classifier.

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

  Conditional Attribute
    An attribute of a :term:`learner` which might be enabled or disabled,
    grouped within ``.ca`` attributes collection.  If enabled, it might cause
    additional computation and memory consumption, so the "heaviest"
    conditional attributes are disabled by default.

  Confusion Matrix
    Visualization of the :term:`generalization` performance of a
    :term:`classifier`.  Each row of the matrix represents the instances in a
    predicted class, while each column represents the :term:`sample`\s in an
    actual (target) class.  Each cell provides a count of how many
    :term:`sample`\s of the target class were (mis)classifier into the
    corresponding class.  In PyMVPA instances of
    :class:`~mvpa2.clfs.transerror.ConfusionMatrix` class provide not only
    confusion matrix itself but a bulk of additional statistics.

  Dataset
    In PyMVPA a dataset is the combination of samples, and their
    :term:`Dataset attribute`\s.

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

  Exemplar
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
    This acronym stands for *functional magnetic resonance imaging*.

  Generalization
    An ability of a model to perform reliably well on any novel data in
    the given domain.

  Label
    A label is a special case of a :term:`target` for specifying discrete
    categories of :term:`sample`\s in a classification analyses.

  Learner
    A model that upon training given some data (:term:`sample`\s and may be
    :term:`target`\s) develops an ability to map an arbitrary :term:`feature`
    space of :term:`sample`\s into another space.  If :term:`target`\s were
    provided, such learner is called :term:`supervised` and tries to achieve
    mapping into the space of :term:`target`\s.  If the target space defined by
    a set of discrete set of labels, such learner is called a
    :term:`classifier`.

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
    instrumental method (e.g., EEG, :term:`fMRI`).

  Processing object
    Most objects dealing with data are implemented as processing objects. Such
    objects are instantiated *once*, with all appropriate parameters
    configured as desired. When created, they can be used multiple times by
    simply calling them with new data.

  Sample
    A sample is a vector with observations for all :term:`feature` variables.

  Sample attribute
    A per-sample vector of auxiliary information that is stored in a
    dataset. This could, for example, be a vector identifying specific
    :term:`chunk`\ s of samples.

  Sensitivity
    A sensitivity is a score assigned to each :term:`feature` with respect to
    its impact on the performance of the learner.  So, for a classifier,
    sensitivity of a feature might describe its influence on :term:`generalization`
    performance of the classifier.  In case of linear classifiers, it could
    simply be coefficients of separating hyperplane given by :term:`weight
    vector`. There exist additional scores which are similar to sensitivities
    in terms of indicating the "importance" of a particular feature --
    examples are a univariate :ref:`anova` score or a
    :ref:`noise_perturbation` measure.

  Sensitivity Map
    A vector of several sensitivity scores -- one for each feature in a
    dataset.

  Spatial Discrimination Map (SDM)
    This is another term for a :term:`sensitivity map`, used in e.g.
    :ref:`Wang et al. (2007) <WCW+07>`.

  Statistical Discrimination Map (SDM)
    This is another term for a :term:`sensitivity map`, used in e.g.
    :ref:`Sato et al. (2008) <SMM+08>`, where instead of raw sensitivity
    the result of significance testing is assigned.

  Statistical Learning
    A field of science related to :term:`machine learning` which aims at
    exploiting statistical properties of data to construct robust models, and to
    assess their convergence and :term:`generalization` performances.

  Supervised
    Is a :term:`learner` which obtains both :term:`sample`\s data and
    :term:`target`\s within a :term:`training dataset`.

  Target
    A target associates each :term:`sample` in the :term:`dataset` with a
    certain category, experimental condition or, in case of a regression
    problem, with some metric variable.  In case of supervised learning
    algorithm targets define the model to be trained, and provide the "ground
    truth" for assessing the model's :term:`generalization` performance.

  Time-compression
    This usually refers to the :term:`block-averaging` of samples from a
    block-design fMRI dataset.

  Training Dataset
    :term:`Dataset` which is used for training of the :term:`learner`.

  Testing Dataset
    :term:`Dataset` which is used to assess the :term:`generalization` of the
    :term:`learner`.

  Weight Vector
    See :term:`Sensitivity`.
