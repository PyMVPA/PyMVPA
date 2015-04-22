#!/bin/sh

set -e
set -u

# BOILERPLATE

# where is the data; support standard env variable switch
dataroot=${MVPA_DATA_ROOT:-"mvpa2/data"}

# where to place output; into tmp by default
outdir=${MVPA_EXAMPLE_WORKDIR:-}
if [ -z "$outdir" ]; then
  outdir=$(mktemp -d)
  # cleanup if working in tmpdir upon failure/exit
  trap "rm -rf \"$outdir\"" TERM INT EXIT
fi

#% EXAMPLE START

#% A simple start (on the command line)
#% ====================================

#% This script is the exact equivalent of the :ref:`example_start_easy` example,
#% but using the command line interface.

#% First we load an fMRI dataset with some attributes for each volume, only
#% considering voxels that are non-zero in a mask image.

pymvpa2 mkds --mri-data "$dataroot"/bold.nii.gz \
            --add-sa-attr "$dataroot"/attributes_literal.txt  \
            --mask "$dataroot"/mask.nii.gz \
            -o "$outdir"/bold_ds.hdf5

#% Next we remove linear trends by polynomial regression for each voxel and
#% each chunk (recording run) of the dataset individually.

pymvpa2 preproc --chunks chunks \
                --poly-detrend 1 \
                -i "$outdir"/bold_ds.hdf5 \
                -o "$outdir"/preproced.hdf5

#% For this example we are only interested in data samples that correspond
#% to the ``face`` or to the ``house`` condition.

pymvpa2 select --samples-by-attr targets eq face or targets eq house \
               -i "$outdir"/preproced.hdf5 \
               -o "$outdir"/facehouse.hdf5

#% The setup for our cross-validation analysis include the selection of a
#% classifier from the "warehouse", a partitioning scheme, and an error function
#% to convert literal predictions into a quantitative performance metric.

pymvpa2 crossval --learner 'SMLR(lm=1.0)' \
                 --partitioner oddeven:chunks \
                 --errorfx mean_mismatch_error \
                 --avg-datafold-results \
                 -i "$outdir"/facehouse.hdf5 \
                 -o "$outdir"/crossval_results.hdf5

#% The resulting dataset contains the computed accuracy.

echo -n "Error for cross-validation problem: "
pymvpa2 dump -s -i "$outdir"/crossval_results.hdf5

#% EXAMPLE END

pymvpa2 exec -i "$outdir"/crossval_results.hdf5 \
               -e 'assert ds.shape == (1,1)' \
               -e 'assert ds.samples[0,0] < 0.1'

