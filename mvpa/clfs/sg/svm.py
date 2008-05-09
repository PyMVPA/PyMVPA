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
from mvpa.misc.state import StateVariable
from mvpa.clfs.classifier import Classifier, MulticlassClassifier
from mvpa.algorithms.datameasure import Sensitivity


if __debug__:
    from mvpa.misc import debug

known_kernels = { "linear": shogun.Kernel.LinearKernel,
                  "rbf" :   shogun.Kernel.GaussianKernel }

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
        if __debug__:
            debug("SG_", "Setting verbosity for shogun.%s instance to M_DEBUG" %
                  partname)
        obj.io.set_loglevel(shogun.Kernel.M_DEBUG)
        try:
            obj.io.enable_progress()
        except:
            warning("Shogun version installed has no way to enable progress" +
                    " reports")
    else:
        if __debug__:
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



class SVM_SG_Modular(Classifier):
    """Support Vector Machine Classifier(s) based on Shogun

    This is a simple base interface
    """

    # init the parameter interface
    # TODO: should do that via __metaclass__ I guess -- collect all
    # parameters for a given class. And in general -- think about
    # class vs instance definition of them...
    params = Classifier.params.copy()

    params['eps'] = Parameter(1e-5,
                              min=0,
                              descr='tolerance of termination criterium')

    params['tune_eps'] = Parameter(1e-2,
                                   min=0,
                                   descr='XXX')

    params['C'] = Parameter(1.0,
                            min=1e-10,
                            descr='Trade-off parameter. High C -- ridig margin SVM')

    def __init__(self,
                 kernel_type='Linear',
                 kernel_params=[1.0],
                 svm_impl="libsvm",   # gpbt was failing on testAnalyzerWithSplitClassifier for some reason
                 C=1.0,
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
        Classifier.__init__(self, **kwargs)

        self.__svm = None
        """Holds the trained svm."""

        # although there are some multiclass SVMs already in shogun,
        # we might rely on our own
        self.__mclf = None
        """Holds `multiclassClassifier` if such one is needed"""

        # assign default params
        self.params = {}
        self.params.update(SVM_SG_Modular.params)

        self.params['C'].val = C

        if kernel_type.lower() in known_kernels:
            self.__kernel_type = known_kernels[kernel_type.lower()]
        else:
            raise ValueError, "Unknown kernel %s" % kernel_type

        if svm_impl.lower() in known_svm_impl:
            self.__svm_impl = svm_impl.lower()
        else:
            raise ValueError, "Unknown SVM implementation %s" % svm_impl

        self.__kernel_params = kernel_params

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
        if __debug__:
            debug("SG_",
                  "Creating kernel instance of %s" % `self.__kernel_type`)

        self.__kernel = self.__kernel_type(self.__traindata, self.__traindata,
                                           self.__kernel_params[0])
        _setdebug(self.__kernel, 'Kernels')

        # create SVM
        if __debug__:
            debug("SG_", "Creating SVM instance of %s" % `svm_impl_class`)

        self.__svm = svm_impl_class(self.params['C'].val, self.__kernel, labels)
        _setdebug(self.__svm, 'SVM')

        # train
        if __debug__:
            debug("SG", "Training SG_SVM %s on data with labels %s" %
                  (self.__kernel_type, dataset.uniquelabels))

        self.__svm.train()

        # train
        if __debug__:
            debug("SG__", "Done training SG_SVM %s on data with labels %s" %
                  (self.__kernel_type, dataset.uniquelabels))


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
        SVM_SG_Modular.__init__(self, kernel_type='Linear', **kwargs)


    def getSensitivityAnalyzer(self, **kwargs):
        """Returns an appropriate SensitivityAnalyzer."""
        return ShogunLinearSVMWeights(self, **kwargs)



class LinearCSVMC(LinearSVM):
    def __init__(self, C=1.0, **kwargs):
        """
        """
        # init base class
        LinearSVM.__init__(self, C=C, **kwargs)



class RbfCSVMC(SVM_SG_Modular):
    """C-SVM classifier using a radial basis function kernel.
    """
    def __init__(self, C=1.0, gamma=1.0, **kwargs):
        """
        """
        # init base class
        SVM_SG_Modular.__init__(self, C=C, kernel_type='RBF', kernel_params=[gamma], **kwargs)



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
