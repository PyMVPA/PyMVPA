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

# EXAMPLE START

# load an fMRI dataset with some attributes for each volume
# only include voxels that are non-zero in a mask image
pymvpa2 mkds --mri-data "$dataroot"/bold.nii.gz \
            --add-sa-attr "$dataroot"/attributes_literal.txt  \
            --mask "$dataroot"/mask.nii.gz \
            -o "$outdir"/bold_ds.hdf5

# spectral filter, dataset with TR=2.5s
# remove waves with periods longer than 400s and shorter than 10s
# keep waves with periods longer than 15s and shorter than 200s
pymvpa2 preproc --filter-passband 0.005 0.067 \
                --filter-stopband 0.0025 0.1 \
                --sampling-rate 0.4 \
                -o "$outdir"/spec_filtered.hdf5 \
                "$outdir"/bold_ds.hdf5

# EXAMPLE END

# cleanup if working in tmpdir
[ $have_tmpdir = 1 ] && rm -rf $outdir || true
