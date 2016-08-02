.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, OpenFMRI
.. _chap_tutorial_openfmri:

********************************
 Working with OpenFMRI.org data
********************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_openfmri.ipynb>`_]

Working with data from other researchers can be hard. There are lots of ways to
collect data, and even more ways to store it on a hard drive. This variability
turns discovering the structure of a "foreign" dataset into a research project
of its own.

Standardization is one way to make this easier and the OpenFMRI_ project has
proposed a scheme for structuring (task) fMRI dataset in order to facilitate
automated analysis. While there are other approaches to standardization, the
`layout proposed by OpenFMRI`_ is appealing, because it offers a good balance
between the level of standardization and the required effort to achieve it.

.. _OpenFMRI: http://www.openfmri.org
.. _layout proposed by openfmri: https://openfmri.org/content/data-organization

PyMVPA offers convenient tools to work with dataset that are (somewhat)
compliant with the OpenFMRI structure. So independent of whether you plan on
sharing your data or not, it may make sense to adopt these conventions, when
working with PyMVPA. Take a look at this tutorial and make up your mind whether
there is something about this convenience that you like. As a bonus, if you
have your dataset formated for OpenFMRI already, it becomes technically trivial
to share it on openfmri.org later on -- for free. Here is how it looks like to
work with an OpenFMRI dataset, starting with the bare necessities:

>>> from os.path import join as opj
>>> import mvpa2
>>> from mvpa2.datasets.sources import OpenFMRIDataset

Assuming you downloaded and extracted a dataset from OpenFMRI.org into the
current directory, you will have a sub-directory (for example ``ds105`` if you
picked the `Haxby et al, (2001) data`_) that contains all files of the data
release. In order to have PyMVPA access this data, we simply have to create a
handler that is pointed to this sub-directory. In order to spare you the 2GB
download just to run this tutorial, we are using a minified version of that
dataset in this demo which already comes with PyMVPA.

.. _Haxby et al, (2001) data: https://openfmri.org/dataset/ds000105

>>> path = opj(mvpa2.pymvpa_dataroot , 'haxby2001')
>>> of = OpenFMRIDataset(path)

Through this handler we can access lots of information about this dataset.
Let's start with what this dataset is all about.

>>> print of.get_task_descriptions()
{1: 'object viewing'}

We can immediately see that the dataset is concerned with a single task *object
viewing* The descriptions are always returned as a dictionary that maps the
task ID (an integer number) to a verbal description. This is done, because a
dataset can contain data for more than one task.

Other descriptive information, such as the number and IDs of the subjects in the
dataset, as well as other supporting information specified in the
``scan_key.txt`` meta data file are also available:

>>> print of.get_subj_ids()
[1, 'phantom']
>>> of.get_scan_properties()
{'TR': '2.5'}

As you can see, subject IDs don't have to be numerical.

So far, the information we retrieved was rather simple and the advantages of
being able to access them through an API will not become obvious until one
starts working with a lot of datasets simultaneously. So let's take a look at
some functionality that is more useful in the context of a single dataset.

For task fMRI, we are almost always interested in information about the
stimulation model, i.e. when was any particular subject exposed to which
experiment conditions. All this information is readily available. Here is how
you get the number and IDs of all contained model specifications:

>>> of.get_model_ids()
[1]
>>> of.get_model_descriptions()
{1: 'visual object categories'}

This particular dataset contains a single model specification. With its
numerical ID we can query more information about the model:

>>> conditions = of.get_model_conditions(1)
>>> print conditions
[{'task': 1, 'id': 1, 'name': 'house'}, {'task': 1, 'id': 2, 'name': 'scrambledpix'}, {'task': 1, 'id': 3, 'name': 'cat'}, {'task': 1, 'id': 4, 'name': 'shoe'}, {'task': 1, 'id': 5, 'name': 'bottle'}, {'task': 1, 'id': 6, 'name': 'scissors'}, {'task': 1, 'id': 7, 'name': 'chair'}, {'task': 1, 'id': 8, 'name': 'face'}]
>>> # that was not human readable -> make prettier
>>> print [c['name'] for c in conditions]
['house', 'scrambledpix', 'cat', 'shoe', 'bottle', 'scissors', 'chair', 'face']

We can easily get a list of the condition names and their association with a
particular task. And with the task ID we can query the dataset for the number
(and IDs) of all related BOLD run fMRI images.

>>> print of.get_task_bold_run_ids(1)
{1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

If there would be actual data available for the ``phantom`` subject, we would
see it in the output too.

With this information we can access almost any item in this dataset that is
related to task fMRI. Take a look at
:meth:`~mvpa2.datasets.sources.openfmri.OpenFMRIDataset.get_bold_run_image`,
:meth:`~mvpa2.datasets.sources.openfmri.OpenFMRIDataset.get_bold_run_dataset`,
and the other methods in order to explore the possibilities.  After looking at
all the raw information available in a dataset, let's take a look at some
high-level functionality that is more interesting when actually working with a
task fMRI dataset.

For any supervised analysis strategy, for example a classification analysis, it
is necessary to assign labels to data points. In PyMVPA, this is done by
creating a dataset with (at least) one sample attribute containing the labels
-- one for each sample in the dataset. The
:meth:`~mvpa2.datasets.sources.openfmri.OpenFMRIDataset.get_model_bold_dataset`
method is a convenient way of generating such a dataset directly from the
OpenFMRI specification. As you'll see in a second, this methods uses any
relevant information contained in the OpenFMRI specification and we only need
to fill in the details of how exactly we want the PyMVPA dataset to be created.
So here is a complete example:

>>> from mvpa2.datasets.eventrelated import fit_event_hrf_model
>>> ds = of.get_model_bold_dataset(
...          model_id=1,
...          subj_id=1,
...          flavor='25mm',
...          mask=opj(path, 'sub001', 'masks', '25mm', 'brain.nii.gz'),
...          modelfx=fit_event_hrf_model,
...          time_attr='time_coords',
...          condition_attr='condition')

So let's take this bit of code apart in order to understand what it is doing.
When calling ``get_model_bold_dataset()``, we specify the model ID and subject
ID, as well as the "flavor" of data we are interested in. Think of the flavor
as different variants of the same raw fMRI time series (e.g. different set of
applied preprocessing steps). We are using the "25mm" flavor, which is our
minified variant of the original dataset, down-sampled to voxels with 25 mm edge
length.  Based on this information, the relevant stimulus model specifications
are discovered and data files for the associated subject are loaded. This
method could be called in a loop to, subsequently, load data for all available
subjects. In addition, we specify a mask image file to exclude non-brain voxels.
Often these masks do not come with a data release and have to be created first.

Now for the important bits: The ``modelfx`` argument takes a, so-called,
factory method that can transform a time series dataset (each sample in the
dataset is a time point at that stage) into the desired type of sample (or
observation). In this example, we have used
:func:`~mvpa2.datasets.eventrelated.fit_event_hrf_model` that is designed to
perform modeling of each stimulation event contained in the OpenFMRI
specification. PyMVPA ships with three principal transformation methods that
can be used here: :func:`~mvpa2.datasets.eventrelated.fit_event_hrf_model`,
:func:`~mvpa2.datasets.eventrelated.extract_boxcar_event_samples` and
:func:`~mvpa2.datasets.eventrelated.assign_conditionlabels`. The difference
between the three is that the latter simply assignes conditions labels to the
time point samples of a time series dataset, whereas the former two can do more
complex transformations, such as temporal compression, or model fitting.  Note,
that is is possible to implement custom transformation functions for
``modelfx``, but all common use cases should be supported by the three functions
that already come with PyMVPA.

All subsequent argument are passed on to the ``modelfx``. In this example, we
requested all events of the same condition to be modeled by a regressor that is
based on a canonical hemodynamic response function (this requires the
specification of a dataset attribute that encodes the timing of a time series
samples; ``time_attr``).

>>> print ds
<Dataset: 96x129@float64, <sa: chunks,condition,regressors,run,subj>, <fa: voxel_indices>, <a: add_regs,imgaffine,imghdr,imgtype,mapper,voxel_dim,voxel_eldim>>

This all led to an output dataset with 96 samples, one sample per each of the
eight condition in each of the 12 runs.

>>> print ds.sa.condition
['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe']
>>> print ds.sa.chunks
[ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  3
  3  3  3  3  3  3  3  4  4  4  4  4  4  4  4  5  5  5  5  5  5  5  5  6  6
  6  6  6  6  6  6  7  7  7  7  7  7  7  7  8  8  8  8  8  8  8  8  9  9  9
  9  9  9  9  9 10 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11]


Each value in the sample matrix corresponds to the estimated model parameter
(or weight) for the associated voxel. Model fitting is performed individually
per each run. The model regressors, as well as numerous other bits of
information are available in the returned dataset.

Depending on the type of preprocessing that was applied to this data flavor,
the dataset ``ds`` may be ready for immediate analysis, for example in
a cross-validated classification analysis. If further preprocessing steps
are desired, the ``preproc_ds`` argument of
:meth:`~mvpa2.datasets.sources.openfmri.OpenFMRIDataset.get_model_bold_dataset`
provides an interface for applying additional transformations, such as temporal
filtering, to the time series data of each individual BOLD fMRI run.
