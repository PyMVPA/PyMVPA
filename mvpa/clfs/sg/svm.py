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
from mvpa.clfs.classifier import Classifier, MulticlassClassifier


if __debug__:
    from mvpa.misc import debug

known_kernels = { "linear": shogun.Kernel.LinearKernel,
                  "rbf" :   shogun.Kernel.GaussianKernel }

known_svm_impl = { "libsvm" :   shogun.Classifier.LibSVM,
                    "lightsvm" : shogun.Classifier.SVMLight }


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
        debug(debugname, "Setting verbosity for shogun.%s to M_INFO" %
              partname)
        obj.io.set_loglevel(shogun.Kernel.M_INFO)
    else:
        obj.io.set_loglevel(shogun.Kernel.M_EMERGENCY)


def _tosg(data):
    """Draft helper function to convert data we have into SG suitable format"""

    features = shogun.Features.RealFeatures(data.astype('double').T)
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
                 svm_impl="LibSVM",
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

        # assign default params
        self.params = {}
        self.params.update(SVM_SG_Modular.params)

        self.params['C'].val = C

        if kernel_type.lower() in known_kernels:
            self.__kernel_type = known_kernels[kernel_type.lower()]
        else:
            raise ValueError, "Unknown kernel %s" % kernel_type

        if svm_impl.lower() in known_svm_impl:
            self.__svm_impl = known_svm_impl[svm_impl.lower()]
        else:
            raise ValueError, "Unknown SVM implementation %s" % svm_impl

        self.__kernel_params = kernel_params


    def __repr__(self):
        """Definition of the object summary over the object
        """
        res = "<SVM_SG_Modular(...) #%d>" % id(self)
        return res # XXX

        sep = ""
        for k,v in self.param._params.iteritems():
            res += "%s%s=%s" % (sep, k, str(v))
            sep = ', '
        res += sep + "enable_states=%s" % (str(self.states.enabled))
        res += ")"
        return res


    def _train(self, dataset):
        """Train SVM
        """
        # although there are some multiclass SVMs already in shogun,
        # for now just rely on our own :-P
        self.__mclf = None

        ul = dataset.uniquelabels
        ul.sort()
        if len(ul) > 2 or (ul != [-1.0, 1.0]).any():
            warning("XXX Using built-in MultiClass classifier for now")
            mclf = MulticlassClassifier(self)
            mclf._train(dataset)
            self.__mclf = mclf
            return

        # create training data
        if __debug__:
            debug("SG_", "Converting input data for shogun")

        self.__traindata = _tosg(dataset.samples)

        # create labels
        #
        ## We need to convert to -1,+1 labels
        #ul = dataset.uniquelabels
        #dict_ = {ul[0]:-1.0,
        #         ul[1]:+1.0}
        #labels_ = N.array([ dict_[x] for x in dataset.labels ], dtype='double')
        #
        if __debug__:
            debug("SG_",
                  "Creating labels instance")
        labels = shogun.Features.Labels(dataset.labels.astype('double'))
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
            debug("SG_", "Creating SVM instance of %s" % `self.__svm_impl`)

        self.__svm = self.__svm_impl(self.params['C'].val, self.__kernel, labels)
        _setdebug(self.__svm, 'SVM')

        # train
        if __debug__:
            debug("SG", "Training SG_SVM %s on data with labels %s" %
                  (self.__kernel_type, dataset.uniquelabels))

        self.__svm.train()


    def _predict(self, data):
        """Predict values for the data
        """

        if not self.__mclf is None:
            return self.__mclf._predict(data)
        if __debug__:
            debug("SG_", "Initializing kernel with training/testing data")
        self.__kernel.init(self.__traindata, _tosg(data))
        if __debug__:
            debug("SG_", "Classifing testing data")
        values = self.__svm.classify().get_labels()

        self.values = values
        predictions = 1.0-2*N.signbit(values)
        return predictions


    def untrain(self):
        if __debug__:
            debug("SVM", "Untraining %s and destroying libsvm model" % self)
        super(SVM_SG_Modular, self).untrain()

        # XXX make it nice... now it is just stable ;-)
        if not self.__mclf is None:
            self.__mclf = None
        else:
            del self.__traindata
            del self.__kernel
            self.__traindata = None
            self.__kernel = None


    svm = property(fget=lambda self: self.__svm)
    """Access to the SVM model."""



class LinearSVM(SVM_SG_Modular):

    def __init__(self, **kwargs):
        """
        """
        # init base class
        SVM_SG_Modular.__init__(self, kernel_type='Linear', **kwargs)


class LinearCSVMC(LinearSVM):
    pass


class RbfCSVMC(SVM_SG_Modular):
    """C-SVM classifier using a radial basis function kernel.
    """
    def __init__(self, **kwargs):
        """
        """
        # init base class
        SVM_SG_Modular.__init__(self, kernel_type='RBF', **kwargs)

