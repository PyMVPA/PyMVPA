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

#% Generate a quick QC plot from motion estimates
#% ==============================================

pymvpa2 plotmotionqc \
    -s "$dataroot"/haxby2001/sub*/BOLD/task001_run*/bold_moest.txt \
    --savefig "$outdir"/motion.png

#% EXAMPLE END
