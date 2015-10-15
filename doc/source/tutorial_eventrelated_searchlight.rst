.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, event-related fMRI, searchlight
.. _chap_tutorial_eventrelated_searchlight:

********************************
 Multi-dimensional Searchlights
********************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_eventrelatedi_searchlight.ipynb>`_]

This is a little addendum to :ref:`chap_tutorial_eventrelated` where we want to
combine what we have learned about event-related data analysis and, at the same
time, take a little glimpse on the power of PyMVPA for "multi-space" analysis.

First let's re-create the dataset with the spatio-temporal features from
:ref:`chap_tutorial_eventrelated`:

>>> from mvpa2.tutorial_suite import *
>>> ds = get_raw_haxby2001_data(roi=(36,38,39,40))
>>> poly_detrend(ds, polyord=1, chunks_attr='chunks')
>>> zscore(ds, chunks_attr='chunks', param_est=('targets', 'rest'))
>>> events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
>>> events = [ev for ev in events if ev['targets'] in ['house', 'face']]
>>> event_duration = 13
>>> for ev in events:
...     ev['onset'] -= 2
...     ev['duration'] = event_duration
>>> evds = eventrelated_dataset(ds, events=events)

From the :ref:`chap_tutorial_searchlight` we know how to do searchlight
analyses and it was promised that there is more to it than what we already saw.
And here it is:

>>> cvte = CrossValidation(GNB(), NFoldPartitioner(),
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
>>> # need to put the time axis last for export to NIfTI
>>> ts = np.rollaxis(ts, 0, 4)
>>> ni = nb.Nifti1Image(ts, ds.a.imgaffine).to_filename('ersl.nii')

.. We need to remove generated files so daily tests pass

After you are done and want to tidy up after yourself, you can easily remove
unneeded generated files from within Python:

>>> os.unlink('ersl.nii')
