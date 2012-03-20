"""
Hyperalignment for between-subject analysis
===========================================

.. index:: hyperalignment, between-subject classification

Multivariate pattern analysis (MVPA) reveals how the brain represents
fine-scale information. Its power lies in its sensitivity to subtle pattern
variations that encode this fine-scale information but that also presents a
hurdle for group analyses due to between-subject variability of both anatomical
& functional architectures. :ref:`Haxby et al. (2011) <HGC+11>` recently
proposed a method of aligning subjects' brain data in a high-dimensional
functional space and showed how to build a common model of ventral temporal
cortex that captures visual object category information. They tested their
model by successfully performing between-subject classification of category
information.  Moreover, when they built the model using a complex naturalistic
stimulation (a feature film), it even generalized to other independent
experiments even after removing any occurrences of the experimental stimuli
from the movie data.

In this example we show how to perform Hyperalignment within a single
experiment. We will compare between-subject classification after hyperalignment
to between-subject classification on anatomically aligned data (currently the
most typical approach), and within-subject classification performance.


Analysis setup
--------------

"""

from mvpa2.suite import *


"""Using a Linear SVM classifier for all classifications."""

clf = LinearCSVMC()

"""
Select nf voxels in each subject based on a simple
OneWayAnova() with proper cross-validation.
"""

nf = 30
fselector = FixedNElementTailSelector(nf, tail='upper', mode='select',sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                        enable_ca=['sensitivities'])
fsclf = FeatureSelectionClassifier( clf, sbfs)
fscvte = CrossValidation(fsclf, NFoldPartitioner(), errorfx=mean_match_accuracy,
            postproc=mean_sample())

cvte = CrossValidation(clf, NFoldPartitioner(attr='subject'), 
            errorfx=mean_match_accuracy)

"""
Loading the preprocessed Face & Object datasets 
[Ref: Haxby et. al. Neuron 2011]
"""

print "Loading the data..."
filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data', 
    "hyperalignment_tutorial_data.hdf5.gz")
ds_all = h5load(filepath)

"""
This is a list of 10 datasets corresponding to 10 subjects.
Each dataset, after preprocessing, has 7 samples per run for each of 8runs.
"""

nsubjs = len(ds_all)
ncats = len(ds_all[0].UT)
nruns = len(ds_all[0].UC)
print "A subject's dataset:"
print ds_all[0]
print "Stimulus categories:", ds_all[0].UT

"""
Between subject classification using anatomically aligned data
--------------------------------------------------------------

"""

bsc_mni_results = []
for test_run in range(nruns):
    # partitioner + splitter?
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]

    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]

    """
    Repeat between-subject classification with MNI-aligned voxels.
    Note that we use average F-Scores to select the same nf voxels across subjects
    """

    fscores_mni = np.mean(np.asarray(vstack(fscores)), axis=0)
    ds_fs_mni = [sd[:,fselector(fscores_mni)] for i,sd in enumerate(ds_test)]
    for i,sd in enumerate(ds_fs_mni):
        sd.sa['subject'] = np.repeat(i,sd.nsamples)
    ds_fs_mni = vstack(ds_fs_mni)
    bsc_mni_results.append( cvte(ds_fs_mni))

bsc_mni_results = hstack(bsc_mni_results)


"""
Between subject classification with Hyperalignment(TM)
------------------------------------------------------

For each test run, we extract the data from all subjects excluding
the test run, and perform ANOVA-based voxel selection. We then perform
Hyperalignment of these selected voxels from all the subjects.
"""

bsc_hyper_results = []
for test_run in range(nruns):
    # partitioner + splitter?
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]

    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]

    """
    Selecting voxels based on ANOVA of our training data.
    """

    # sensitivitybasedfeatureselection?
    ds_fs = [sd[:,fselector(fscores[i])] for i,sd in enumerate(ds_train)]

    """
    Setting up Hyperalignment with default parameters.
    """

    hyper = Hyperalignment()

    """
    Computing hyperalignment parameters is as simple as calling 
    hyperalignment class with a list of datasets with same number of samples.
    This returns a list of mappers corresponding to subjects in the same order
    as the list of datasets we passed in.
    """

    mappers = hyper(datasets=ds_fs)

    """
    Applying hyperalignment parameters is similar to applying any mapper in 
    PyMVPA. We start by selecting the voxels that we used to derive the hyperalignment
    parameters.
    """

    ds_fs = [sd[:,fselector(fscores[i])] for i,sd in enumerate(ds_test)]

    """
    We apply the tranformations by running the test dataset through forward() function of the mapper.
    """

    ds_hyper_all = [ mappers[i].forward(ds_) for i,ds_ in enumerate(ds_fs)]

    """
    Now, we have a list of datasets corrsponding to our subjects which are now all in a common space 
    derived from the training data.
    """

    for i,sd in enumerate(ds_hyper_all):
        sd.sa['subject'] = np.repeat(i, len(sd))
    ds_hyper_all = vstack(ds_hyper_all)
    bsc_hyper_results.append(cvte(ds_hyper_all))

bsc_hyper_results = hstack(bsc_hyper_results)

"""
Within-subject classification
-----------------------------

Performing within-subject classification using a Anova based 
FeatureSelectionClassifier
"""

#Within-subject classification
wsc = []
print "Performing within-subject classification"
wsc = [ fscvte(sd) for sd in ds_all]
wsc = hstack(wsc)

"""
Comparing the results
---------------------

Reporting the results of within-subject and between-subject classification analyses.
"""

print "Average within subject classification accuracy:",
print np.mean(wsc),"+/-",np.std(wsc) / np.sqrt(nsubjs - 1)
print "Average between-subject classfication accuracy (Anatomically aligned):",
print np.mean(bsc_mni_results),"+/-",np.std(np.mean(bsc_mni_results,axis=1))/np.sqrt(nsubjs-1)
print "Average between-subject classfication accuracy (Hyperaligned):",
print np.mean(bsc_hyper_results),"+/-",np.std(np.mean(bsc_hyper_results,axis=1))/np.sqrt(nsubjs-1)
