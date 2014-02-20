#!/bin/sh

set -e
set -u

# BOILERPLATE

# where is the data; support standard env variable switch
dataroot=${MVPA_DATA_ROOT:-"datadb/tutorial_data/tutorial_data/data"}

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
pymvpa2 mkds \
   --mri-data "$dataroot"/bold.nii.gz \
   --mask "$dataroot"/mask_brain.nii.gz \
   --add-sa-attr "$dataroot"/attributes.txt  \
   --add-vol-attr hoc "$dataroot"/mask_hoc.nii.gz \
   --add-vol-attr gm "$dataroot"/mask_gray.nii.gz \
   --add-vol-attr vt "$dataroot"/mask_vt.nii.gz \
   --add-fsl-mcpar "$dataroot"/bold_mc.par \
   --hdf5-compression gzip \
   -o "$outdir"/bold.hdf5

#% Obtain a summary of the dataset content
pymvpa2 describe -i "$outdir"/bold.hdf5

echo "Number of ROIs in the Harvard-Oxford cortial atlas: "
pymvpa2 dump --fa hoc -f txt -i "$outdir"/bold.hdf5 | sort | uniq | wc -l


pymvpa2 preproc --poly-detrend 0 \
                --detrend-regrs mc_x mc_y mc_z mc_rot1 mc_rot2 mc_rot3 \
                --filter-passband 0.005 0.067 \
                --filter-stopband 0.0025 0.1 \
                --sampling-rate 0.4 \
                --chunks chunks \
                --hdf5-compression gzip \
                -o "$outdir"/bold_mcf.hdf5 \
                -i "$outdir"/bold.hdf5

pymvpa2 select \
    --samples-by-attr targets eq face or targets eq house \
    --features-by-attr vt gt 0 \
    -i "$outdir"/bold_mcf.hdf5 \
    -o "$outdir"/faceshouses_inVT.hdf5

pymvpa2 crossval \
    --learner 'SMLR(lm=1.0)' \
    --partitioner n-1 \
    --errorfx mean_match_accuracy \
    --avg-datafold-results \
    -i "$outdir"/faceshouses_inVT.hdf5 \
    -o "$outdir"/xval_faces_vs_houses_inVT.hdf5

pymvpa2 select \
    --samples-by-attr targets eq face or targets eq house \
    --hdf5-compression gzip \
    -i "$outdir"/bold_mcf.hdf5 \
    -o "$outdir"/faceshouses_brain.hdf5

# --scatter-rois is for demo speed-up only
pymvpa2 --dbg-channel SLC searchlight \
    --payload cv \
    --neighbors 3 \
    --scatter-rois 3 \
    --nproc 2 \
    --roi-attr gm \
    --cv-learner 'SMLR(lm=1.0)' \
    --cv-partitioner oddeven:chunks \
    --cv-errorfx mean_match_accuracy \
    --cv-avg-datafold-results \
    --hdf5-compression gzip \
    -i "$outdir"/faceshouses_brain.hdf5 \
    -o "$outdir"/sl_faces_vs_houses_brain.hdf5

pymvpa2 dump -s \
    -f nifti \
    -i "$outdir"/sl_faces_vs_houses_brain.hdf5 \
    -o "$outdir"/sl_faces_vs_houses_brain_ACC.nii.gz

for roi in $(seq 48); do
    echo "Doing ROI $roi"

    pymvpa2 select --features-by-attr hoc eq $roi \
    -i "$outdir"/faceshouses_brain.hdf5 \
    -o "$outdir"/roi_tmp.hdf5

    resultds="${outdir}/xval_faces_vs_houses_inROI${roi}.hdf5"

    pymvpa2 crossval \
        --learner 'SMLR(lm=1.0)' \
        --partitioner n-1 \
        --errorfx mean_match_accuracy \
        --avg-datafold-results \
        -i "$outdir"/roi_tmp.hdf5 \
        -o $resultds 2> /dev/null | grep "ACC%"
done



# EXAMPLE END

#pymvpa2 pytest -i "$outdir"/evds.hdf5 -e 'assert(len(dss[0]) == 3)'
#pymvpa2 pytest -i "$outdir"/short_ds.hdf5 -e 'assert(len(dss[0]) == 500)'

# cleanup if working in tmpdir
[ $have_tmpdir = 1 ] && rm -rf $outdir || true
