.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_searchlight:

******************************
Part 5: The Mighty Searchlight
******************************


ds = get_haxby2001_data_alternative(roi=0)
clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
terr = TransferError(clf)
cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
sl = sphere_searchlight(cvte, postproc=mean_sample())
res=sl(ds)
map2nifti(ds, 1 - res.samples).save('sl.nii.gz')


.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.measures.searchlight.Searchlight
