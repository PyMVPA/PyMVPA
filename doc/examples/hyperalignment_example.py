"""
Hyperalignment Example
===========================

.. index:: hyperalignment, between-subject classification

This is a simple example showing how to use Hyperalignment
within an experiment to compute between-subject classification
of hyperaligned data with proper cross-validation.

"""

from mvpa2.suite import *

""" Using a Linear SVM classifier for all classifications.
"""
clf = LinearCSVMC()
""" Select nf voxels in each subject based on a simple
OneWayAnova() with proper cross-validation.
"""
nf = 250
fselector = FixedNElementTailSelector(nf, tail='upper', mode='select',sort=False)
fsclf = FeatureSelectionClassifier( clf, SensitivityBasedFeatureSelection(OneWayAnova(), fselector) )
cvte = CrossValidation(fsclf, NFoldPartitioner())
""" We need to setup a TransferMeasure to manually do cross-validation
over different folds, since we need to run hyperaglinment for each fold and exlude both the test
subject and the test run from the training data for the classifier.
"""
tacc = TransferMeasure(clf, Splitter(attr='bsc_split',attr_values=[0, 1]), 
                        postproc= lambda x:float(np.sum(x.samples.T==x.targets))/x.nsamples )
""" Loading the preprocessed Face & Object datasets [Ref: Haxby et. al. Neuron 2011]
"""
print "Loading the data..."
filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data', 
    "hyperalignment_tutorial_data.hdf5.gz")
ds_all = h5load(filepath)
""" This is a list of 10 datasets corresponding to 10 subjects.
Each dataset, after preprocessing, has 7 samples per run for each of 8runs.
"""
nsubjs = len(ds_all)
ncats = len(ds_all[0].UT)
nruns = len(ds_all[0].UC)
print "A subject's dataset:"
print ds_all[0]
print "Stimulus categories:", ds_all[0].UT
"""
Performing within-subject classification using a Anova based 
FeatureSelectionClassifier
"""
#Within-subject classification
wsc = []
print "Performing within-subject classification"
wsc = [ cvte(sd) for sd in ds_all]
wsc = hstack(wsc)
print "Within subject classification accuracies:"
print 1.0 - np.mean(wsc), ,"+/-",np.std(1.0-np.mean(wsc,axis=0))/np.sqrt(nsubjs-1)
"""
For each test run, we extract the data from all subjects excluding
the test run, and perform ANOVA-based voxel selection. We then perform
Hyperalignment of these selected voxels from all the subjects.
"""
bsc_hyper_results = []
bsc_mni_results = []
for test_run in range(nruns):
    print "######################"
    print "Testing on run %i" %(test_run)
    print "######################"
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = deepcopy(ds_all)
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    """
    Selecting voxels based on ANOVA of our training data.
    """
    ds_fs = [sd[:,fselector(fscores[i])] for i,sd in enumerate(ds_train)]
    """
    Setting up Hyperalignment with default parameters.
    """
    print "Computing transformations for "
    hyper = Hyperalignment()
    """
    Computing hyperalignment parameters is as simple as calling 
    hyperalignment class with a list of datasets with same number of samples.
    This returns a list of mappers corresponding to subjects in the same order
    as the list of datasets we passed in.
    """
    mappers = hyper(datasets=ds_fs)
    print "Hyperalignment parameters computed"
    """
    Applying hyperalignment parameters is similar to applying any mapper in 
    PyMVPA. We start by selecting the voxels that we used to derive the hyperalignment
    parameters.
    """
    print "Applying hyperalignment to test data"
    #ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_test]
    ds_fs = [sd[:,fselector(fscores[i])] for i,sd in enumerate(ds_test)]
    """
    We apply the tranformations by running the test dataset through forward() function of the mapper.
    """
    ds_hyper_all = [ mappers[i].forward(ds_) for i,ds_ in enumerate(ds_fs)]
    """
    Now, we have a list of datasets corrsponding to our subjects which are now all in a common space 
    derived from the training data.
    """
    print "Adding subject info"
    for i,sd in enumerate(ds_hyper_all):
        sd.sa['subject'] = np.repeat(i,sd.nsamples)
    ds_hyper_all = vstack(ds_hyper_all)
    ds_hyper_all.sa['bsc_split'] = np.zeros(nruns*nsubjs*ncats, dtype=np.int)
    for test_subj in range(nsubjs):
        ds_hyper_all_train = ds_hyper_all[ np.where( np.logical_and( ds_hyper_all.sa.chunks != test_run, ds_hyper_all.sa.subject != test_subj)) ]
        ds_hyper_all_test  = ds_hyper_all[ np.where( np.logical_and( ds_hyper_all.sa.chunks == test_run, ds_hyper_all.sa.subject == test_subj)) ]
        ds_hyper_all_train.sa.bsc_split = np.zeros((nsubjs-1)*(nruns-1)*ncats,dtype=np.int)
        ds_hyper_all_test.sa.bsc_split = np.ones(ncats,dtype=np.int)
        bsc_hyper_results.append( tacc(vstack([ds_hyper_all_train, ds_hyper_all_test])))
    """ Repeat between-subject classification with MNI-aligned voxels.
    Note that we use average F-Scores to select the same nf voxels across subjects
    """
    fscores_mni = np.mean(np.asarray(vstack(fscores)), axis=0)
    ds_fs_mni = [sd[:,fselector(fscores_mni)] for i,sd in enumerate(ds_test)]
    for i,sd in enumerate(ds_fs_mni):
        sd.sa['subject'] = np.repeat(i,sd.nsamples)
    ds_fs_mni = vstack(ds_fs_mni)
    ds_fs_mni.sa['bsc_split'] = np.zeros(nruns*nsubjs*ncats, dtype=np.int)
    for test_subj in range(nsubjs):
        ds_fs_mni_train = ds_fs_mni[ np.where( np.logical_and( ds_fs_mni.sa.chunks != test_run, ds_fs_mni.sa.subject != test_subj)) ]
        ds_fs_mni_test  = ds_fs_mni[ np.where( np.logical_and( ds_fs_mni.sa.chunks == test_run, ds_fs_mni.sa.subject == test_subj)) ]
        ds_fs_mni_train.sa.bsc_split = np.zeros((nsubjs-1)*(nruns-1)*ncats,dtype=np.int)
        ds_fs_mni_test.sa.bsc_split = np.ones(ncats,dtype=np.int)
        bsc_mni_results.append( tacc(vstack([ds_fs_mni_train, ds_fs_mni_test])))

bsc_hyper_results = np.reshape(bsc_hyper_results, (nruns,nsubjs))
bsc_mni_results = np.reshape(bsc_mni_results, (nruns,nsubjs))
"""
Reporting the results of within-subject and between-subject classification analyses.
"""
print "Average within subject classification accuracy:",
print 1.0 - np.mean(wsc),"+/-",np.std(1.0 - np.mean(wsc,axis=0))/np.sqrt(nsubjs-1)
print "Average between-subject classfication accuracy (Anatomically aligned):",
print np.mean(bsc_mni_results),"+/-",np.std(np.mean(bsc_mni_results,axis=0))/np.sqrt(nsubjs-1)
print "Average between-subject classfication accuracy (Hypearligned):",
print np.mean(bsc_hyper_results),"+/-",np.std(np.mean(bsc_hyper_results,axis=0))/np.sqrt(nsubjs-1)
