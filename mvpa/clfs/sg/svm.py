#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrap the libsvm package into a very simple class interface."""

__docformat__ = 'restructuredtext'

import numpy as N


# Rely on SG
# TODO: XXX dual-license under GPL for use of SG?
import shogun.Features
import shogun.Classifier
import shogun.Kernel
import shogun.Library


from mvpa.misc.param import Parameter
from mvpa.misc import warning

from mvpa.clfs.classifier import MulticlassClassifier
from mvpa.clfs._svmbase import _SVM
from mvpa.misc.state import StateVariable
from mvpa.algorithms.datameasure import Sensitivity


if __debug__:
    from mvpa.misc import debug

known_svm_impl = { "libsvm" : shogun.Classifier.LibSVM,
                   "gmnp" : shogun.Classifier.GMNPSVM,
                   #"mpd"  : shogun.Classifier.MPDSVM, # disable due to infinite looping on XOR
                   "gpbt" : shogun.Classifier.GPBTSVM,
                   "gnpp" : shogun.Classifier.GNPPSVM,
                   }

try:
    known_svm_impl["lightsvm"] = shogun.Classifier.SVMLight
except:
    warning("No LightSVM implementation is available in given shogun")


def _setdebug(obj, partname):
    """Helper to set level of debugging output for SG
    :Parameters:
      obj
        In SG debug output seems to be set per every object
      partname : basestring
        For what kind of object we are talking about... could be automated
        later on (TODO)
    """
    debugname = "SG_%s" % partname.upper()

    if __debug__ and debugname  in debug.active:
        debug("SG_", "Setting verbosity for shogun.%s instance to M_DEBUG" %
              partname)
        obj.io.set_loglevel(shogun.Kernel.M_DEBUG)
        # progress is enabled by default so don't bother
    else:
        debug("SG_", "Setting verbosity for shogun.%s instance to M_EMERGENCY" %
              partname + " and disabling progress reports")
        obj.io.set_loglevel(shogun.Kernel.M_EMERGENCY)
        try:
            obj.io.disable_progress()
        except:
            warning("Shogun version installed has no way to disable progress" +
                    " reports")


def _tosg(data):
    """Draft helper function to convert data we have into SG suitable format"""

    if __debug__:
        debug("SG_", "Converting data for shogun into RealFeatures")
    features = shogun.Features.RealFeatures(data.astype('double').T)
    if __debug__:
        debug("SG__", "Done converting data for shogun into RealFeatures")
    _setdebug(features, 'Features')
    return features


class SVM_SG_Modular(_SVM):
    """Support Vector Machine Classifier(s) based on Shogun

    This is a simple base interface
    """

    # init the parameter interface
    tube_epsilon = Parameter(1e-2,
                             min=1e-10,
                             descr='XXX Some kind of tolerance')

    num_threads = Parameter(1,
                            min=1,
                            descr='Number of threads to utilize')

    # XXX gamma is width in SG notation for RBF(Gaussian)
    _KERNELS = { "linear": (shogun.Kernel.LinearKernel,   ()),
                 "rbf" :   (shogun.Kernel.GaussianKernel, ('gamma',)),
                 "rbfshift" : (shogun.Kernel.GaussianShiftKernel, ('gamma', 'max_shift', 'shift_step')),
                 "sigmoid" : (shogun.Kernel.SigmoidKernel, ('cache_size', 'gamma', 'coef0')),
                }

    _KNOWN_PARAMS = [ 'C', 'epsilon' ]
    _KNOWN_KERNEL_PARAMS = [ ]

    def __init__(self,
                 kernel_type='linear',
                 svm_impl="libsvm",   # gpbt was failing on testAnalyzerWithSplitClassifier for some reason
                 **kwargs):
        # XXX Determine which parameters depend on each other and implement
        # safety/simplifying logic around them
        # already done for: nr_weight
        # thought: weight and weight_label should be a dict
        """This is the base class of all classifier that utilize so
        far just SVM classifiers provided by shogun.

        TODO Documentation if this all works ;-)
        """

        # init base class
        _SVM.__init__(self, kernel_type=kernel_type, **kwargs)

        self.__svm = None
        """Holds the trained svm."""

        # although there are some multiclass SVMs already in shogun,
        # we might rely on our own
        self.__mclf = None
        """Holds `multiclassClassifier` if such one is needed"""

        # assign default params
        # XXX taking abs for now since some implementations might freak out until we implement proper scaling
        # self.params.C = C

        if svm_impl.lower() in known_svm_impl:
            self.__svm_impl = svm_impl.lower()
        else:
            raise ValueError, "Unknown SVM implementation %s" % svm_impl

        # Need to store original data...
        # TODO: keep 1 of them -- just __traindata or __traindataset
        self.__traindataset = None

        # internal SG swig proxies
        self.__traindata = None
        self.__kernel = None


    def __str__(self):
        """Definition of the object summary over the object
        """
        res = "<SVM_SG_Modular#%d>" % id(self)
        return res # XXX

        sep = ""
        for k,v in self.param._params.iteritems():
            res += "%s%s=%s" % (sep, k, str(v))
            sep = ', '
        res += sep + "enable_states=%s" % (str(self.states.enabled))
        res += ")"
        return res

    def __repr__(self):
        # adjust representation a bit to report SVM backend
        id_ = super(SVM_SG_Modular, self).__repr__()
        return id_.replace("#", "/%s#" % self.__svm_impl)


    def _train(self, dataset):
        """Train SVM
        """
        self.untrain()
        self.__mclf = None
        svm_impl_class = None

        ul = dataset.uniquelabels
        ul.sort()
        if len(ul) > 2: # or (ul != [-1.0, 1.0]).any():
            if self.__svm_impl == 'libsvm':
                svm_impl_class = shogun.Classifier.LibSVMMultiClass
            elif self.__svm_impl == 'gmnp':
                svm_impl_class = shogun.Classifier.GMNPSVM
                # or just known_svm_impl[self.__svm_impl]
            else:
                warning("XXX Using built-in MultiClass classifier for now")
                mclf = MulticlassClassifier(self)
                mclf._train(dataset)
                self.__mclf = mclf
                return
        else:
            svm_impl_class = known_svm_impl[self.__svm_impl]

        # create training data
        if __debug__:
            debug("SG_", "Converting input data for shogun")

        self.__traindataset = dataset
        self.__traindata = _tosg(dataset.samples)

        # create labels
        #

        # OK -- we have to map labels since
        #  binary ones expect -1/+1
        #  Multiclass expect labels starting with 0, otherwise they puke
        #   when ran from ipython... yikes
        if __debug__:
            debug("SG_", "Creating labels instance")

        if len(ul) == 2:
            # assure that we have -1/+1
            self._labels_dict = {ul[0]:-1.0,
                                 ul[1]:+1.0}
        elif len(ul) < 2:
            raise ValueError, "we do not have 1-class SVM brought into pymvpa yet"
        else:
            # can't use plain enumerate since we need them swapped
            self._labels_dict = dict([ (ul[i], i) for i in range(len(ul))])

        # reverse labels dict for back mapping in _predict
        self._labels_dict_rev = dict([(x[1], x[0])
                                      for x in self._labels_dict.items()])

        if __debug__:
            debug("SG__", "Mapping labels using dict %s" % self._labels_dict)
        labels_ = N.array([ self._labels_dict[x] for x in dataset.labels ], dtype='double')
        labels = shogun.Features.Labels(labels_)
        _setdebug(labels, 'Labels')

        # create kernel
        # TODO: decide on how to handle kernel parameters more or less
        # appropriately
        #if len(self.__kernel_params)==1:

        kargs = []
        for arg in self._KERNELS[self._kernel_type_literal][1]:
            value = self.kernel_params[arg].value
            # XXX Unify damn automagic gamma value
            if arg == 'gamma' and value == 0.0:
                value = 1.0/len(ul)     # the same way is done in libsvm
            kargs += [value]

        if __debug__:
            debug("SG_",
                  "Creating kernel instance of %s giving arguments %s" %
                  (`self._kernel_type`, kargs))


        self.__kernel = self._kernel_type(self.__traindata, self.__traindata,
                                          *kargs)

        _setdebug(self.__kernel, 'Kernels')

        # create SVM
        if __debug__:
            debug("SG_", "Creating SVM instance of %s" % `svm_impl_class`)

        C = self.params.C
        if C<0:
            C = self._getDefaultC(dataset.samples)*abs(C)
            if __debug__:
                debug("SG_", "Default C for %s was computed to be %s" % (self.params.C, C))

        self.__svm = svm_impl_class(C, self.__kernel, labels)

        # Set optimization parameters
        self.__svm.set_epsilon(self.params.epsilon)
        self.__svm.set_tube_epsilon(self.params.tube_epsilon)
        self.__svm.parallel.set_num_threads(self.params.num_threads)

        _setdebug(self.__svm, 'SVM')

        # train
        if __debug__:
            debug("SG", "Training SG_SVM %s %s on data with labels %s" %
                  (self._kernel_type, self.params, dataset.uniquelabels))

        self.__svm.train()

        # train
        if __debug__:
            debug("SG_", "Done training SG_SVM %s on data with labels %s" %
                  (self._kernel_type, dataset.uniquelabels))
            if "SG__" in debug.active:
                trained_labels = self.__svm.classify().get_labels()
                debug("SG__", "Original labels: %s, Trained labels: %s" % (dataset.labels, trained_labels))

    def _predict(self, data):
        """Predict values for the data
        """

        if not self.__mclf is None:
            return self.__mclf._predict(data)

        if __debug__:
            debug("SG_", "Initializing kernel with training/testing data")

        testdata = _tosg(data)
        self.__kernel.init(self.__traindata, testdata)

        if __debug__:
            debug("SG_", "Classifying testing data")

        # doesn't do any good imho although on unittests helps tiny bit... hm
        #self.__svm.init_kernel_optimization()

        values_ = self.__svm.classify()
        values = values_.get_labels()

        if __debug__:
            debug("SG__", "Got values %s" % values)

        if len(self._labels_dict) == 2:
            predictions = 1.0-2*N.signbit(values)
        else:
            predictions = values

        # assure that we have the same type
        label_type = type(self._labels_dict.values()[0])

        # remap labels back adjusting their type
        predictions = [self._labels_dict_rev[label_type(x)]
                       for x in predictions]

        if __debug__:
            debug("SG__", "Tuned predictions %s" % predictions)

        # store state variable
        self.values = values

        # to avoid leaks with not yet properly fixed shogun
        try:
            testdata.free_features()
        except:
            pass

        return predictions


    def untrain(self):
        if __debug__:
            debug("SG__", "Untraining %s and destroying sg's SVM" % self)
        super(SVM_SG_Modular, self).untrain()

        # to avoid leaks with not yet properly fixed shogun

        # XXX make it nice... now it is just stable ;-)
        if not self.__mclf is None:
            self.__mclf.untrain()
            self.__mclf = None
        elif not self.__traindata is None:
            try:
                try:
                    self.__traindata.free_features()
                except:
                    pass
                self.__traindataset = None
                del self.__kernel
                self.__kernel = None
                del self.__traindata
                self.__traindata = None
                del self.__svm
                self.__svm = None
            except:
                pass

        if __debug__:
            debug("SG__", "Done untraining %s and destroying sg's SVM" % self)


    svm = property(fget=lambda self: self.__svm)
    """Access to the SVM model."""

    mclf = property(fget=lambda self: self.__mclf)
    """Multiclass classifier if it was used"""

    traindataset = property(fget=lambda self: self.__traindataset)
    """Dataset which was used for training

    TODO -- might better become state variable I guess"""



class LinearSVM(SVM_SG_Modular):

    def __init__(self, **kwargs):
        """
        """
        # init base class
        SVM_SG_Modular.__init__(self, kernel_type='linear', **kwargs)

    def getSensitivityAnalyzer(self, **kwargs):
        """Returns an appropriate SensitivityAnalyzer."""
        return ShogunLinearSVMWeights(self, **kwargs)



class LinearCSVMC(LinearSVM):
    def __init__(self, **kwargs):
        """
        """
        # init base class
        LinearSVM.__init__(self, **kwargs)



class RbfCSVMC(SVM_SG_Modular):
    """C-SVM classifier using a radial basis function kernel.
    """
    def __init__(self, C=1, **kwargs):
        """
        """
        # init base class
        SVM_SG_Modular.__init__(self, C=C, kernel_type='RBF', **kwargs)



class ShogunLinearSVMWeights(Sensitivity):
    """`Sensitivity` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """

    biases = StateVariable(enabled=True,
                           doc="Offsets of separating hyperplanes")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf: LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        # init base classes first
        Sensitivity.__init__(self, clf, **kwargs)


    def __sg_helper(self, svm):
        """Helper function to compute sensitivity for a single given SVM"""
        self.offsets = svm.get_bias()
        svcoef = N.matrix(svm.get_alphas())
        svnums = svm.get_support_vectors()
        svs = self.clf.traindataset.samples[svnums,:]
        res = (svcoef * svs).mean(axis=0).A1
        return res


    def _call(self, dataset):
        #from IPython.Shell import IPShellEmbed
        #ipshell = IPShellEmbed()
        #ipshell()
        #12: self.clf._SVM_SG_Modular__mclf.clfs[0].clf._SVM_SG_Modular__svm.get_bias()
        #19: alphas=self.clf._SVM_SG_Modular__mclf.clfs[0].clf._SVM_SG_Modular__svm.get_alphas()
        #20: svs=self.clf._SVM_SG_Modular__mclf.clfs[0].clf._SVM_SG_Modular__svm.get_support_vectors()

        # TODO: since multiclass is done internally - we need to check
        # here if self.clf.__mclf is not an instance of some out
        # Classifier and apply corresponding combiner of
        # sensitivities... think about it more... damn

        # XXX Hm... it might make sense to unify access functions
        # naming across our swig libsvm wrapper and sg access
        # functions for svm

        if not self.clf.mclf is None:
            anal = self.clf.mclf.getSensitivityAnalyzer()
            if __debug__:
                debug('SVM',
                      '! Delegating computing sensitivity to %s' % `anal`)
            return anal(dataset)

        svm = self.clf.svm
        if isinstance(svm, shogun.Classifier.MultiClassSVM):
            sens = []
            for i in xrange(svm.get_num_svms()):
                sens.append(self.__sg_helper(svm.get_svm(i)))
        else:
            sens = self.__sg_helper(svm)
        return N.array(sens)



#if __debug__:
#    if 'SG_PROGRESS' in debug.active:
#        debug('SG_PROGRESS', 'Allowing SG progress bars')
#    else:
#        if 
#import shogun.Library
#io = shogun.Library.IO()
#io.disable_progress()
