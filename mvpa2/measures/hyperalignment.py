import numpy as np
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.measures.base import Measure
from mvpa2.featsel.base import StaticFeatureSelection
from mvpa2.datasets import Dataset
from scipy.linalg import LinAlgError
    
class HyperalignmentMeasure(Measure):
    is_trained=True
    def __init__(self, ndatasets=11, scale=0.0, index_attr='index', **kwargs):
        Measure.__init__(self, **kwargs)
        self.ndatasets = ndatasets
        self.scale = scale
        self._index_attr = index_attr
        
    def __call__(self, dataset):
        # create the dissimilarity matrix for the data in the input dataset
        ds = []
        nsamples = dataset.nsamples/self.ndatasets
        seed_index = np.where(dataset.fa.roi_seed)
        if self.scale>0.0:
            dist = np.sum(np.abs(dataset.fa.voxel_indices-dataset.fa.voxel_indices[seed_index]), axis=1)
            dist = np.exp(-(self.scale*dist/np.float(max(dist)) )**2)
            dataset.samples = dataset.samples*dist
        for i in range(self.ndatasets):
            ds.append(dataset[0+i*nsamples:nsamples*(i+1),])
        for ref_ds in range(self.ndatasets):
            try:
                hyper = Hyperalignment(zscore_common=True, ref_ds = ref_ds)
                mappers = hyper(datasets=ds)
                # Extract only the row/column corresponding to the center voxel
                mappers = [ np.squeeze(m.proj[:,seed_index]) for m in mappers]
                break
            except LinAlgError:
                print "SVD didn't converge. Trying with a new reference: %i" %(ref_ds+1)
                print "Nevermind, this is not a good idea for searchlight hyperalignment"
                raise
                break
                if ref_ds == self.ndatasets-1:
                    mappers = []
                    print "SVD didn't converge with any reference. We are screwed :("
                    raise
            else:
                print "We are Screwed..."
        
        
        return Dataset(samples=np.asanyarray([{'proj':mapper,'fsel':StaticFeatureSelection(dataset.fa[self._index_attr].value)} for mapper in mappers]))

