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

# what was the name of the debug channel for 'searchlight' analyses?
pymvpa2 info --debug | grep -i searchlight

# what version of nibabel is used by pymvpa?
pymvpa2 info --externals |grep nibabel

# create a description of the computing environment that can be posted on the
# pymvpa mailing list to make a bug report more informative
pymvpa2 info > $outdir/mysystem.txt

# EXAMPLE END

# cleanup if working in tmpdir
[ $have_tmpdir = 1 ] && rm -rf $outdir || true
