#!/bin/sh

set -e
set -u

# BOILERPLATE

# where is the data; support standard env variable switch
dataroot=${MVPA_DATA_ROOT:-"mvpa2/data"}

# where to place output; into tmp by default
outdir=${MVPA_EXAMPLE_WORKDIR:-}
have_tmpdir=0
if [ -z "$outdir" ]; then
  outdir=$(mktemp -d)
  have_tmpdir=1
fi

#% EXAMPLE START
#% Working with datasets
#% =====================

#% load an fMRI dataset with some attributes for each volume
#% only include voxels that are non-zero in a mask image
pymvpa2 mkds --mri-data "$dataroot"/bold.nii.gz \
            --add-sa-attr "$dataroot"/attributes_literal.txt  \
            --mask "$dataroot"/mask.nii.gz \
            -o "$outdir"/bold_ds.hdf5

pymvpa2 preproc --chunks chunks \
                --poly-detrend 1 \
                -i "$outdir"/bold_ds.hdf5 \
                -o "$outdir"/preproced.hdf5

pymvpa2 select --samples-by-attr targets eq face or targets eq house \
               -i "$outdir"/preproced.hdf5 \
               -o "$outdir"/facehouse.hdf5

pymvpa2 crossval --learner 'SMLR(lm=1.0)' \
                 --partitioner oddeven:chunks \
                 --errorfx mean_mismatch_error \
                 --avg-datafold-results \
                 -i "$outdir"/facehouse.hdf5 \
                 -o "$outdir"/crossval_results.hdf5

echo -n "Error for cross-validation problem: "
pymvpa2 dump -s -i "$outdir"/crossval_results.hdf5

#% EXAMPLE END

pymvpa2 eval -i "$outdir"/crossval_results.hdf5 \
               -e 'assert ds.shape == (1,1)' \
               -e 'assert ds.samples[0,0] < 0.1'

# cleanup if working in tmpdir
[ $have_tmpdir = 1 ] && rm -rf $outdir || true
