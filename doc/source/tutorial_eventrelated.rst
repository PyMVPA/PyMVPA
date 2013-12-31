.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, event-related fMRI
.. _chap_tutorial_eventrelated:

***********************************************************
Part 7: "When Is The Signal" -- Event-related Data Analysis
***********************************************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_eventrelated.ipynb>`_]

In all previous tutorial parts we have analyzed the same fMRI data. We
analyzed it using a number of different strategies, but they all had one
thing in common: A sample in each dataset was always a single fMRI volume.
Sometimes, we have limited ourselves to just a specific region of interest,
sometimes we averaged many fMRI volumes into a single one. In all cases,
however, a feature always corresponded to a voxel in the fMRI volume and
appeared only once in the dataset.

In this part we are going to extend the analysis beyond the spatial
dimensions and will consider *time* as another aspect of our data.
This is a common thing to do, for example, in ERP-analyses of EEG data.
Here we are going to employ a similar approach in our well-known example
data -- this time selecting a subset of ventral temporal regions.

>>> from mvpa2.tutorial_suite import *
>>> ds = get_raw_haxby2001_data(roi=(36,38,39,40))
>>> print ds.shape
(1452, 941)

As we know, this dataset consists of 12 concatenated experiment sessions.
Every session had a stimulation block spanning multiple fMRI volumes for
each of the eight stimulus categories. Stimulation blocks were separated by
rest periods. What we want to do now is to look at the spatio-temporal
signal across our region of interest and the full duration of the
stimulation blocks. In other words, we want to perform a sensitivity
analysis revealing the spatio-temporal distribution of
classification-relevant information.

In this kind of analysis, we consider each stimulation block an
:term:`event` and we need to create a representative sample for every one
of them. In the context of an event-related fMRI classification analysis, the
literature offers three principal techniques:

1. Choose a single representative volume.
2. Compress all relevant volumes into a single one (averaging or parametric fit).
3. Consider the full event-related time series.

Obviously, only the third approach can possibly provide us with a temporal
sensitivity profile, hence we will choose this path.

Event-related Pre-processing Is Not Event-related
-------------------------------------------------

For an event-related analysis, most of the processing is done on data
samples that are somehow derived from a set of events. The rest of the data
could be considered irrelevant. However, some preprocessing is only
meaningful when performed on the full time series and not on the segmented
event samples. An example is the detrending that typically needs to be done
on the original, continuous time series.

In its current shape our dataset consists of 1452 samples that represent
contiguous fMRI volumes. At this stage we can easily perform linear
detrending. Again, we are going to do it per each experiment run (the
dataset has runs encoded in the ``chunks`` sample attribute), since we do
not assume a contiguous linear trend throughout the whole time series.

>>> poly_detrend(ds, polyord=1, chunks_attr='chunks')

Let's make a copy of the de-trended dataset that we can use later on for
some visualization.

>>> orig_ds = ds.copy()

We still need to normalize each feature (i.e. a voxel at this point). In
this case we are, again, going to Z-score them, using the mean and standard
deviation from the experiment's rest condition. The resulting values might
be interpreted as "activation scores". We are again doing it per each run.

>>> zscore(ds, chunks_attr='chunks', param_est=('targets', 'rest'))


From Timeseries To Events
-------------------------

After detrending and normalization, we can now segment the time series into
a set of events. To achieve this we have to compile a list of event
definitions first. An event is defined by *onset*, *duration* and
potentially a number of additional properties, such as stimulus condition
or recording session number.

In this example we will simply convert the block-design setup defined by
the samples attributes into a list of events. With
:func:`~mvpa2.datasets.eventrelated.find_events`, PyMVPA provides a
function to convert sequential attributes into event lists. In our dataset,
we have the stimulus conditions of each volume sample available in the
``targets`` sample attribute.

>>> events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
>>> print len(events)
204
>>> for e in events[:4]:
...    print e
{'chunks': 0.0, 'duration': 6, 'onset': 0, 'targets': 'rest'}
{'chunks': 0.0, 'duration': 9, 'onset': 6, 'targets': 'scissors'}
{'chunks': 0.0, 'duration': 6, 'onset': 15, 'targets': 'rest'}
{'chunks': 0.0, 'duration': 9, 'onset': 21, 'targets': 'face'}

We are feeding not only the ``targets`` to the function, but also the
``chunks`` attribute, since we do not want to have events spanning multiple
recording sessions. :func:`~mvpa2.datasets.eventrelated.find_events`
sequentially parses all provided attributes and records an event whenever the
value in *any* of the attributes changes. The generated event definition is a
dictionary that contains:

1. Onset of the event as an index in the sequence (in this example this is a
   volume id)
2. Duration of the event in "number of sequence elements" (i.e. number of
   volumes). The duration is determined by counting the number of identical
   attribute combinations following an event onset.
3. Attribute combination of this event, i.e. the actual values of all given
   attributes at the particular position.

Let's limit ourselves to ``face`` and ``house`` stimulation blocks for now.
We can easily filter out all other events.

>>> events = [ev for ev in events if ev['targets'] in ['house', 'face']]
>>> print len(events)
24
>>> for e in events[:4]:
...    print e
{'chunks': 0.0, 'duration': 9, 'onset': 21, 'targets': 'face'}
{'chunks': 0.0, 'duration': 9, 'onset': 63, 'targets': 'house'}
{'chunks': 1.0, 'duration': 9, 'onset': 127, 'targets': 'face'}
{'chunks': 1.0, 'duration': 9, 'onset': 213, 'targets': 'house'}
>>> np.unique([e['duration'] for e in events])
array([9])

All of our events are of the same length, 9 consecutive fMRI volumes. Later
on we would like to view the temporal sensitivity profile from *before* until
*after* the stimulation block, hence we should extend the duration of the
events a bit.

>>> event_duration = 13
>>> for ev in events:
...     ev['onset'] -= 2
...     ev['duration'] = event_duration

The next and most important step is to actually segment the original
time series dataset into event-related samples. PyMVPA offers
:func:`~mvpa2.datasets.eventrelated.eventrelated_dataset` as a function to
perform this conversion. Let's just do it, it only needs the original
dataset and our list of events.

>>> # alt: `evds = load_tutorial_results('ds_haxby2001_blkev_facehouse')`
>>> evds = eventrelated_dataset(ds, events=events)
>>> len(evds) == len(events)
True
>>> evds.nfeatures == ds.nfeatures * event_duration
True

.. h5save('results/ds_haxby2001_blkev_facehouse.hdf5', ds)

.. exercise::

  Inspect the ``evds`` dataset. It has a fairly large number of attributes
  -- both for samples and for features. Look at each of them and think
  about what it could be useful for.

At this point it is worth looking at the dataset's mapper -- in particular at
the last two items in the chain mapper that have been added during the
conversion into events.

>>> print evds.a.mapper[-2:]
<Chain: <Boxcar: bl=13>-<Flatten>>

.. exercise::

  Reverse-map a single sample through the last two items in the chain
  mapper. Inspect the result and make sure it doesn't surprise. Now,
  reverse-map multiple samples at once and compare the result. Is this what
  you would expect?

The rest of our analysis is business as usual and is quickly done.  We want to
perform a cross-validation analysis of an SVM classifier. We are not
primarily interested in its performance, but in the weights it assigns to
the features. Remember, each feature is now voxel-at-time-point, so we get a
chance of looking at the spatio-temporal profile of classification-relevant
information in the data. We will nevertheless enable computing of a confusion
matrix, so we can assure ourselves that the classifier is performing
reasonably well, because only a generalizing model is worth
inspecting, as otherwise it overfits and the assigned weights
could be meaningless.

>>> sclf = SplitClassifier(LinearCSVMC(),
...                        enable_ca=['stats'])
>>> sensana = sclf.get_sensitivity_analyzer()
>>> sens = sensana(evds)

.. exercise::

  Check that the classifier achieves an acceptable accuracy. Is it
  enough above chance level to allow for an interpretation of the
  sensitivities?

.. exercise::

  Using what you have learned in the last tutorial part: Combine the
  sensitivity maps for all splits into a single map. Project this map into
  the original dataspace. What is the shape of that space? Store the
  projected map into a NIfTI file and inspect it using an MRI viewer.
  Viewer needs to be capable of visualizing time series (hint: for FSLView
  the time series image has to be opened first)!


A Plotting Example
------------------

We have inspected the spatio-temporal profile of the sensitivities using
some MRI viewer application, but we can also assemble an informative figure
right here. Let's compose a figure that shows the original peri-stimulus
time series, the effect of normalization, as well as the corresponding
sensitivity profile of the trained SVM classifier. We are going to do that
for two example voxels, whose coordinates we might have derived from
inspecting the full map.

>>> example_voxels = [(28,25,25), (28,23,25)]

The plotting will be done by the popular matplotlib_ package.

.. _matplotlib: http://matplotlib.sourceforge.net/

First, we plot the original signal after initial detrending. To do this, we
apply the same time series segmentation to the original detrended dataset
and plot the mean signal for all face and house events for both of our
example voxels. The code below will create the plot using matplotlib's
``pylab`` interface (imported as ``pl``). If you are familiar with Matlab's
plotting facilities, this shouldn't be hard to read.

.. note::
   ``_ =`` is used in the examples below simply to absorb output of plotting
   functions.  You do not have to swallow output in your interactive sessions.

>>> # linestyles and colors for plotting
>>> vx_lty = ['-', '--']
>>> t_col = ['b', 'r']

>>> # whole figure will have three rows -- this is the first
>>> _ = pl.subplot(311)
>>> # for each of the example voxels
>>> for i, v in enumerate(example_voxels):
...     # get a slicing array matching just to current example voxel
...     slicer = np.array([tuple(idx) == v for idx in ds.fa.voxel_indices])
...     # perform the timeseries segmentation just for this voxel
...     evds_detrend = eventrelated_dataset(orig_ds[:, slicer], events=events)
...     # now plot the mean timeseries and standard error
...     for j, t in enumerate(evds.uniquetargets):
...         l = plot_err_line(evds_detrend[evds_detrend.sa.targets == t].samples,
...                           fmt=t_col[j], linestyle=vx_lty[i])
...         # label this plot for automatic legend generation
...         l[0][0].set_label('Voxel %i: %s' % (i, t))
>>> # y-axis caption
>>> _ = pl.ylabel('Detrended signal')
>>> # visualize zero-level
>>> _ = pl.axhline(linestyle='--', color='0.6')
>>> # put automatic legend
>>> _ = pl.legend()
>>> _ = pl.xlim((0,12))

In the next figure row we do exactly the same again, but this time for the
normalized data.

>>> _ = pl.subplot(312)
>>> for i, v in enumerate(example_voxels):
...     slicer = np.array([tuple(idx) == v for idx in ds.fa.voxel_indices])
...     evds_norm = eventrelated_dataset(ds[:, slicer], events=events)
...     for j, t in enumerate(evds.uniquetargets):
...         l = plot_err_line(evds_norm[evds_norm.sa.targets == t].samples,
...                           fmt=t_col[j], linestyle=vx_lty[i])
...         l[0][0].set_label('Voxel %i: %s' % (i, t))
>>> _ = pl.ylabel('Normalized signal')
>>> _ = pl.axhline(linestyle='--', color='0.6')
>>> _ = pl.xlim((0,12))

Finally, we plot the associated SVM weight profile for each peri-stimulus
time-point of both voxels. For easier selection we do a little trick and
reverse-map the sensitivity profile through the last mapper in the
dataset's chain mapper (look at ``evds.a.mapper`` for the whole chain).
This will reshape the sensitivities into ``cross-validation fold x volume x
voxel features``.

>>> _ = pl.subplot(313)
>>> # L1 normalization of sensitivity maps per split to make them
>>> # comparable
>>> normed = sens.get_mapped(FxMapper(axis='features', fx=l1_normed))
>>> smaps = evds.a.mapper[-1].reverse(normed)

>>> for i, v in enumerate(example_voxels):
...     slicer = np.array([tuple(idx) == v for idx in ds.fa.voxel_indices])
...     smap = smaps.samples[:,:,slicer].squeeze()
...     l = plot_err_line(smap, fmt='ko', linestyle=vx_lty[i], errtype='std')
>>> _ = pl.xlim((0,12))
>>> _ = pl.ylabel('Sensitivity')
>>> _ = pl.axhline(linestyle='--', color='0.6')
>>> _ = pl.xlabel('Peristimulus volumes')

That was it. Perhaps you are scared by the amount of code. Please note that
it could have been done shorter, but this way allows for plotting any other voxel
coordinate combination as well. matplotlib also allows for saving this figure in
SVG_ format, allowing for convenient post-processing in Inkscape_ -- a
publication quality figure is only minutes away.

.. _SVG: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _Inkscape: http://www.inkscape.org/

.. figure:: pics/ex_eventrelated.*
   :align: center

   Sensitivity profile for two example voxels for *face* vs. *house*
   classification on event-related fMRI data from ventral temporal cortex.

.. exercise::

  What can we say about the properties of the example voxel's signal from
  the peri-stimulus plot?


If That Was Easy...
-------------------

This demo showed an event-related data analysis. Although we have performed
it on fMRI data, an analogous analysis can be done for any time series based
data in an almost identical fashion. Moreover, if a dataset has information
about acquisition time (e.g. like the ones created by
:func:`~mvpa2.datasets.mri.fmri_dataset`)
:func:`~mvpa2.datasets.eventrelated.eventrelated_dataset()` can also convert
event-definition in real time, making it relatively easy to "convert"
experiment design logfiles into event lists. In this case there would be no
need to run a function like
:func:`~mvpa2.datasets.eventrelated.find_events`, but instead they could be
directly specified and passed to
:func:`~mvpa2.datasets.eventrelated.eventrelated_dataset()`.

At the end of this tutorial part we want to take a little glimpse on the power
of PyMVPA for "multi-space" analysis. From the :ref:`previous tutorial part
<chap_tutorial_searchlight>` we know how to do searchlight analyses and it was
promised that there is more to it than what we already saw. And here it is:

>>> cvte = CrossValidation(LinearCSVMC(), NFoldPartitioner(),
...                        postproc=mean_sample())
>>> sl = Searchlight(cvte,
...                  IndexQueryEngine(voxel_indices=Sphere(1),
...                                   event_offsetidx=Sphere(2)),
...                  postproc=mean_sample())
>>> res = sl(evds)

Have you been able to deduce what this analysis will do? Clearly, it is some
sort of searchlight, but it doesn't use
:func:`~mvpa2.measures.searchlight.sphere_searchlight`. Instead, it utilizes
:class:`~mvpa2.measures.searchlight.Searchlight`. Yes, you are correct this is
a spatio-temporal searchlight. The searchlight focus travels along all possible
locations in our ventral temporal ROI, but at the same time also along the
peristimulus time segment covered by the events. The spatial searchlight extent
is the center voxel and its immediate neighbors and the temporal dimension
comprises of two additional time-points in each direction. The result is again
a dataset. Its shape is compatible with the mapper of ``evds``, hence it can
also be back-projected into the original 4D fMRI brain space.

:class:`~mvpa2.measures.searchlight.Searchlight` is a powerful class that
allows for complex runtime ROI generation. In this case it uses an
:class:`~mvpa2.misc.neighborhood.IndexQueryEngine` to look at certain
feature attributes in the dataset to compose sphere-shaped ROIs in two
spaces at the same time. This approach is very flexible and can be
extended with additional query engines to algorithms of almost arbitrary
complexity.


.. there is something that prevents us from mapping the whole dataset

>>> ts = res.a.mapper.reverse1(1 - res.samples[0])
>>> ni = nb.Nifti1Image(ts, ds.a.imghdr.get_best_affine(),
...                         ds.a.imghdr).to_filename('ersl.nii')

After all is done
-----------------

.. We need to remove generated files so daily tests pass

After you are done and want to tidy up after yourself, you can easily remove
unneeded generated files from within Python:

>>> os.unlink('ersl.nii')
