#!/bin/sh

set -e
set -u

# BOILERPLATE

# where is the data; support standard env variable switch
dataroot=${MVPA_DATA_ROOT:-"datadb/tutorial_data/tutorial_data/data"}

# where to place output; into tmp by default
outdir=${MVPA_EXAMPLE_WORKDIR:-}

# which classifier we will use through out the analyses
clf=${MVPA_EXAMPLE_CLF:-'SMLR(lm=1.0)'}

have_tmpdir=0
if [ -z "$outdir" ]; then
  outdir=$(mktemp -d)
  have_tmpdir=1
fi

#% EXAMPLE START

#% Full-scale fMRI data analysis using pattern classification
#% ==========================================================

#% This script demonstrates a complete classification analysis as it could be
#% found in actual research projects. It is a representative example of the
#% functionality accessible from the command line interface -- when it was
#% originally release with PyMVPA 2.3.

#% We start by creating a dataset as a collection of information from various
#% sources. fMRI data is loaded from a 4D NIfTI image, while only voxel with
#% non-zero value in a mask image are kept. Each volume is associated with some
#% attribute values that are read from a text file. In addition, a number of
#% mask additional mask images are included as feature attributes, in order to
#% be able to conveniently group voxel/features into ROIs. Lastly, motion
#% estimates are included as per-volume attributes. The resulting dataset
#% is stored in a compressed HDF5 file.

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

#% The describe command generates a terse summary of the dataset.

pymvpa2 describe -i "$outdir"/bold.hdf5

#% One of the additional feature attributed was down-sampled and aligned
#% brain parcelation of the Harvard-Oxford cortical atlas. The ``dump`` command
#% can be used to extract any dataset component and convert it into a variety
#% of formats -- including plain text -- for further processing with standard
#% UNIX console tools.

echo "Number of ROIs in the Harvard-Oxford cortial atlas: "
pymvpa2 dump --fa hoc -f txt -i "$outdir"/bold.hdf5 | sort | uniq | wc -l

#% The ``preproc`` command offers a few selected preprocessing procedures.  Here
#% we first regress out the motion parameter estimate time-courses that were
#% included as sample attributes in the dataset. And finally, we perform spectral
#% filtering using a Butterworth with selected pass and stop bands.
#% As with every processing step, the result is stored as a dataset in an HDF5
#% file.

pymvpa2 preproc --poly-detrend 0 \
                --detrend-regrs mc_x mc_y mc_z mc_rot1 mc_rot2 mc_rot3 \
                --filter-passband 0.005 0.067 \
                --filter-stopband 0.0025 0.1 \
                --sampling-rate 0.4 \
                --chunks chunks \
                --hdf5-compression gzip \
                -o "$outdir"/bold_mcf.hdf5 \
                -i "$outdir"/bold.hdf5

#% For the subsequent classification analysis we are only interested in two
#% out of all available experimental conditions, and only in voxels that are
#% part of the "VT" mask. Sub-selection of dataset content
#% is supported by indices, as well as a simple expressions.

pymvpa2 select \
    --samples-by-attr targets eq face or targets eq house \
    --features-by-attr vt gt 0 \
    -i "$outdir"/bold_mcf.hdf5 \
    -o "$outdir"/faceshouses_inVT.hdf5

#% The easiest way to perform a classification analysis is the selection of a
#% pre-crafted classifier instance from the "warehouse". A variety of data
#% partitioning schemes is available -- we select "leave-one-out" -- by default
#% operating on the ``chunks`` sample attribute of the dataset. Selection of error
#% functions is possible by name, or by providing a Python script with a custom
#% implementation. Many commands and options of the command line interface can be
#% fed and extended with custom Python scripts.

pymvpa2 crossval \
    --learner "$clf" \
    --partitioner n-1 \
    --errorfx mean_match_accuracy \
    --avg-datafold-results \
    -i "$outdir"/faceshouses_inVT.hdf5 \
    -o "$outdir"/xval_faces_vs_houses_inVT.hdf5

#% After this first ROI-based classification analysis, we are now aiming for
#% a very similar classification analysis that, in contrast, is done in a
#% "searchlight" -- a traveling ROI analysis throught the entire brain.
#% Hence we create a new dataset, again with only face and house data samples,
#% but this time including all voxels.

pymvpa2 select \
    --samples-by-attr targets eq face or targets eq house \
    --hdf5-compression gzip \
    -i "$outdir"/bold_mcf.hdf5 \
    -o "$outdir"/faceshouses_brain.hdf5

#% The ``searchlight`` command can be used to compute arbitrary metric in this
#% fashion, but has built-in support for cross-validated classification analyses
#% (``--payload``).  Spherical ROI with a radius of 4 voxels (``--neighbors``) will
#% be generated, centered on gray-matter voxels only (``--roi-attr``). The
#% computation will be parallelized with up to two concurrent processes.

#% [Note: The ``--scatter-rois`` option is only present to speed up computation
#%  and can be removed in order to obtain a dense result map.]

pymvpa2 --dbg-channel SLC searchlight \
    --payload cv \
    --neighbors 4 \
    --scatter-rois 5 \
    --roi-attr gm \
    --nproc 2 \
    --cv-learner "$clf" \
    --cv-partitioner oddeven:chunks \
    --cv-errorfx mean_match_accuracy \
    --cv-avg-datafold-results \
    --cv-permutations 2 \
    --hdf5-compression gzip \
    -i "$outdir"/faceshouses_brain.hdf5 \
    -o "$outdir"/sl_faces_vs_houses_brain.hdf5

#% Using the ``dump`` command, results can be stored in various formats,
#% including NIfTI. Saving as NIfTI automatically takes care of projecting
#% back results into the 3D voxel space.

pymvpa2 dump -s \
    -f nifti \
    -i "$outdir"/sl_faces_vs_houses_brain.hdf5 \
    -o "$outdir"/sl_faces_vs_houses_brain_ACC.nii.gz
pymvpa2 dump --fa null_prob \
    -f nifti \
    -i "$outdir"/sl_faces_vs_houses_brain.hdf5 \
    -o "$outdir"/sl_faces_vs_houses_brain_NP.nii.gz

#% An alternative to a searchlight with its often arbitrary ROI shapes and
#% boundaries is an iterative ROI analysis -- cycling through a number of
#% ROIs that are defined by localizers or an atlas. Here we perform the
#% cross-validated classification analysis shown above on all areas defined
#% in the Harvard-Oxford cortical atlas and present in our data.

hoc_rois=( $(pymvpa2 exec -i "$outdir"/bold.hdf5 -e 'print(" ".join(map(str, dss[0].fa["hoc"].unique)))') )
echo "ROIs of the Harvard-Oxford cortial atlas present in the data: ${hoc_rois[*]}"

for roi in ${hoc_rois[*]}; do
    echo -en " ROI $roi\t"

#% Select corresponding voxels.

    pymvpa2 select --features-by-attr hoc eq $roi \
        -i "$outdir"/faceshouses_brain.hdf5 \
        -o "$outdir"/roi_tmp.hdf5

#% Report number of voxels present in the given ROI.

    nfeatures=$(pymvpa2 exec -i "$outdir"/roi_tmp.hdf5 -e "print(dss[0].nfeatures)")
    resultds="${outdir}/xval_faces_vs_houses_inROI${roi}.hdf5"

    echo -en "$nfeatures voxels\taccuracy="

#% And run the cross validation, finally printing the overall accuracy
#% as a result on the console.

    pymvpa2 crossval \
        --learner "$clf" \
        --partitioner n-1 \
        --errorfx mean_match_accuracy \
        --avg-datafold-results \
        -i "$outdir"/roi_tmp.hdf5 \
        -o $resultds | awk '/ACC%/{printf "%.2f%%\n", $2}'

    [ -z "${MVPA_TESTS_QUICK:-}" ] || break  # reserved for testing
done

#% EXAMPLE END

# cleanup if working in tmpdir
[ $have_tmpdir = 1 ] && rm -rf $outdir || true
