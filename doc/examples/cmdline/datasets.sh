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

# add another dummy datasets sample attribute from the command line
pymvpa2 mkds --add-sa one 1 -o "$outdir"/ones.hdf5 -i "$outdir"/bold_ds.hdf5

# dump the just added attribute in txt formt and count its elements
pymvpa2 dump --sa one -f txt -i "$outdir"/ones.hdf5 | wc -l

# show a summary of the dataset content
pymvpa2 dsinfo -i "$outdir"/bold_ds.hdf5 | grep '^Dataset'

# create a simple CSV table on the fly to define some "events" with attributes
cat << EOT > "$outdir"/events.csv
"vol","attr","part"
0,"leafy",1
10,"bald",1
100,"furry",2
EOT

# convert the previous time series dataset into an event-related one
# using the 'vol' column in the CSV table to define event onsets
pymvpa2 mkevds --csv-events  "$outdir"/events.csv --onset-column vol \
               -o "$outdir"/evds.hdf5 \
               --offset 3 \
               --duration 2 \
               --event-compression mean \
               -i "$outdir"/bold_ds.hdf5

pymvpa2 pytest -i "$outdir"/evds.hdf5 -e 'assert(len(dss[0]) == 3)'

# EXAMPLE END

# cleanup if working in tmpdir
[ $have_tmpdir = 1 ] && rm -rf $outdir || true
