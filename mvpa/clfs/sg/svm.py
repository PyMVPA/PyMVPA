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
import shogun.Regression
import shogun.Kernel
import shogun.Library


from mvpa.misc.param import Parameter
from mvpa.misc import warning

from mvpa.clfs.base import MulticlassClassifier
from mvpa.clfs._svmbase import _SVM
from mvpa.misc.state import StateVariable
from mvpa.misc.support import idhash
from mvpa.clfs.base import Classifier, MulticlassClassifier
from mvpa.measures.base import Sensitivity
from mvpa.base import externals


if __debug__:
    from mvpa.misc import debug

known_svm_impl = { "libsvm" : shogun.Classifier.LibSVM,
                   # fails to train though it should... disabled for now
                   #"gmnp" : shogun.Classifier.GMNPSVM,

                   # disabled due to infinite looping on XOR
                   #"mpd"  : shogun.Classifier.MPDSVM,

                   # fails to train for testAnalyzerWithSplitClassifier
                   #"gpbt" : shogun.Classifier.GPBTSVM,

                   # Failes some times with 'assertion Cache_Size > 2' though should work
                   # check later
                   #"gnpp" : shogun.Classifier.GNPPSVM,

                   # Regressions
                   "libsvr": shogun.Regression.LibSVR,
                   # also not that simple to make it 'generalize' as a binary classifier
                   # after proper 'binning'
                   #"svrlight": shogun.Regression.SVRLight,
                   #"krr": shogun.Regression.KRR
                   }

def _get_implementation(svm_impl, nl):
    if nl > 2:
        if svm_impl == 'libsvm':
            svm_impl_class = shogun.Classifier.LibSVMMultiClass
        elif svm_impl == 'gmnp':
            svm_impl_class = shogun.Classifier.GMNPSVM
        else:
            raise RuntimeError, \
                  "Shogun: Implementation %s doesn't handle multiclass " \
                  "data. Got labels %s. Use some other classifier" % \
                  (svm_impl, ul)
        if __debug__:
            debug("SG_", "Using %s for multiclass data of %s" %
                  (svm_impl_class, svm_impl))
    else:
            svm_impl_class = known_svm_impl[svm_impl]
    return svm_impl_class


if externals.exists('shogun.lightsvm'):
    known_svm_impl["lightsvm"] = shogun.Classifier.SVMLight

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

    switch = {True: (shogun.Kernel.M_DEBUG, 'M_DEBUG', "enable"),
              False: (shogun.Kernel.M_EMERGENCY, 'M_EMERGENCY', "disable")}

    key = __debug__ and debugname in debug.active

    sglevel, slevel, progressfunc = switch[key]

    if __debug__:
        debug("SG_", "Setting verbosity for shogun.%s instance: %s to %s" %
              (partname, `obj`, slevel))
    obj.io.set_loglevel(sglevel)
    try:
        exec "obj.io.%s_progress()" % progressfunc
    except:
        warning("Shogun version installed has no way to enable progress" +
                " reports")


def _tosg(data):
    """Draft helper function to convert data we have into SG suitable format

    TODO: Support different datatypes
    """

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

    # NOTE: gamma is width in SG notation for RBF(Gaussian)
    _KERNELS = { "linear": (shogun.Kernel.LinearKernel,   ()),
                 "rbf" :   (shogun.Kernel.GaussianKernel, ('gamma',)),
                 "rbfshift" : (shogun.Kernel.GaussianShiftKernel, ('gamma', 'max_shift', 'shift_step')),
                 "sigmoid" : (shogun.Kernel.SigmoidKernel, ('cache_size', 'gamma', 'coef0')),
                }

    _KNOWN_PARAMS = [ 'C', 'epsilon' ]
    _KNOWN_KERNEL_PARAMS = [ ]

    _clf_internals = _SVM._clf_internals + [ 'sg', 'retrainable' ]

    def __init__(self,
                 kernel_type='linear',
                 svm_impl="libsvm",   # gpbt was failing on testAnalyzerWithSplitClassifier for some reason
                 **kwargs):
        """This is the base class of all classifier that utilize so
        far just SVM classifiers provided by shogun.

        TODO Documentation if this all works ;-)
        """
        if svm_impl == 'krr':
            self._KNOWN_PARAMS = self._KNOWN_PARAMS[:] + ['tau']

        # init base class
        _SVM.__init__(self, kernel_type=kernel_type, **kwargs)

        self.__svm = None
        """Holds the trained svm."""

        # assign default params
        svm_impl = svm_impl.lower()
        if  svm_impl in known_svm_impl:
            self.__svm_impl = svm_impl
        else:
            raise ValueError, "Unknown SVM implementation %s" % svm_impl

        self._clf_internals.append(
            {True: 'multiclass', False:'binary'}[
            svm_impl in ['gmnp', 'libsvm']])
        if svm_impl in ['svrlight', 'libsvr', 'krr']:
            self._clf_internals += [ 'regression' ]

        # Need to store original data...
        # TODO: keep 1 of them -- just __traindata or __traindataset
        # For now it is needed for computing sensitivities
        self.__traindataset = None

        # internal SG swig proxies
        self.__traindata = None
        self.__kernel = None
        self.__testdata = None

        # if we do retraining -- store hashes
        # samples, labels, test_samples
        self.__idhash = [None, None, None]

        if __debug__:
            if 'RETRAIN' in debug.active:
                # XXX it is not clear though if idhash is faster than
                # simple comparison of (dataset != __traineddataset).any(),
                # but if we like to get rid of __traineddataset then we should
                # use idhash anyways

                # XXX now we keep 2 copies of the data -- __traineddataset
                #     has it in SG format... uff
                #
                # samples, labels, test_samples, trainsamples_intest
                self.__trained = [None, None, None]


    def __repr__(self):
        # adjust representation a bit to report SVM backend
        repr_ = super(SVM_SG_Modular, self).__repr__()
        return repr_.replace("(kern", "(svm_impl='%s', kern" % self.__svm_impl)


    def __wasChanged(self, descr, i, entry):
        """Check if given entry was changed from what known prior. If so -- store"""
        idhash_ = idhash(entry)
        changed = self.__idhash[i] != idhash_
        if __debug__ and 'RETRAIN' in debug.active:
            changed2 = entry != self.__trained[i]
            if isinstance(changed2, N.ndarray):
                changed2 = changed2.any()
            if changed != changed2:# and not changed:
                raise RuntimeError, \
                  'hashid found to be weak for %s. Though hashid %s!=%s %s, '\
                  'values %s!=%s %s' % \
                  (descr, idhash_, self.__idhash[i], changed,
                   entry, self.__trained[i], changed2)
            self.__trained[i] = entry
        if __debug__ and changed:
            debug('SG__', "Changed %s from %s to %s"
                      % (descr, self.__idhash[i], idhash_))
        self.__idhash[i] = idhash_
        return changed


    def _train(self, dataset):
        """Train SVM
        """
        # XXX might get up in hierarchy
        if self.retrainable:
            changed_params = self.params.whichSet()
            changed_kernel_params = self.kernel_params.whichSet()

        # XXX watchout
        # self.untrain()
        newkernel, newsvm = False, False
        if self.retrainable:
            if __debug__:
                debug('SG__', "IDHashes are %s" % (self.__idhash))
            changed_samples = self.__wasChanged('samples', 0, dataset.samples)
            changed_labels = self.__wasChanged('labels', 1, dataset.labels)
 
        ul = dataset.uniquelabels
        ul.sort()

        self.__traindataset = dataset

        # LABELS

        # OK -- we have to map labels since
        #  binary ones expect -1/+1
        #  Multiclass expect labels starting with 0, otherwise they puke
        #   when ran from ipython... yikes
        if __debug__:
            debug("SG_", "Creating labels instance")

        if 'regression' in self._clf_internals:
            labels_ = N.asarray(dataset.labels, dtype='double')
        else:
            if len(ul) == 2:
                # assure that we have -1/+1
                self._labels_dict = {ul[0]:-1.0,
                                     ul[1]:+1.0}
            elif len(ul) < 2:
                raise ValueError, "we do not have 1-class SVM brought into SG yet"
            else:
                # can't use plain enumerate since we need them swapped
                self._labels_dict = dict([ (ul[i], i) for i in range(len(ul))])

            # reverse labels dict for back mapping in _predict
            self._labels_dict_rev = dict([(x[1], x[0])
                                          for x in self._labels_dict.items()])

            # Map labels
            #
            # TODO: top level classifier should take care about labels
            # mapping if that is needed
            if __debug__:
                debug("SG__", "Mapping labels using dict %s" % self._labels_dict)
            labels_ = N.asarray([ self._labels_dict[x] for x in dataset.labels ], dtype='double')

        labels = shogun.Features.Labels(labels_)
        _setdebug(labels, 'Labels')


        # KERNEL
        if not self.retrainable or changed_samples or changed_kernel_params:
            # If needed compute or just collect arguments for SVM and for
            # the kernel
            kargs = []
            for arg in self._KERNELS[self._kernel_type_literal][1]:
                value = self.kernel_params[arg].value
                # XXX Unify damn automagic gamma value
                if arg == 'gamma' and value == 0.0:
                    value = self._getDefaultGamma(dataset)
                kargs += [value]

            if self.retrainable and __debug__:
                if changed_samples:
                    debug("SG_",
                          "Re-Creating kernel since samples has changed")

                if changed_kernel_params:
                    debug("SG_",
                          "Re-Creating kernel since params %s has changed" %
                          changed_kernel_params)

            # create training data
            if __debug__: debug("SG_", "Converting input data for shogun")
            self.__traindata = _tosg(dataset.samples)

            if __debug__:
                debug("SG_", "Creating kernel instance of %s giving arguments %s" %
                      (`self._kernel_type`, kargs))

            self.__kernel = self._kernel_type(self.__traindata, self.__traindata,
                                              *kargs)
            newkernel = True
            if self.retrainable:
                self.__kernel.set_precompute_matrix(True, True)
                self.__kernel_test = None
                self.__kernel_args = kargs
            _setdebug(self.__kernel, 'Kernels')

        # SVM
        C = self.params.C
        if C<0:
            C = self._getDefaultC(dataset.samples)*abs(C)
            if __debug__:
                debug("SG_", "Default C for %s was computed to be %s" %
                             (self.params.C, C))

        if not self.retrainable or self.__svm is None:
            # Choose appropriate implementation
            svm_impl_class = _get_implementation(self.__svm_impl, len(ul))

            if __debug__:
                debug("SG_", "Creating SVM instance of %s" % `svm_impl_class`)

            if self.__svm_impl in ['libsvr', 'svrlight']:
                # for regressions constructor a bit different
                self.__svm = svm_impl_class(C, self.params.epsilon, self.__kernel, labels)
            elif self.__svm_impl in ['krr']:
                self.__svm = svm_impl_class(self.params.tau, self.__kernel, labels)
            else:
                self.__svm = svm_impl_class(C, self.__kernel, labels)
                self.__svm.set_epsilon(self.params.epsilon)
            newsvm = True
            _setdebug(self.__svm, 'SVM')
            # Set optimization parameters
            if hasattr(self.__svm, 'set_tube_epsilon'):
                self.__svm.set_tube_epsilon(self.params.tube_epsilon)
            self.__svm.parallel.set_num_threads(self.params.num_threads)
        else:
            if __debug__:
                debug("SG_", "SVM instance is not re-created")
            if changed_labels:          # labels were changed
                self.__svm.set_labels(labels)
            if changed_params:
                raise NotImplementedError, \
                      "Implement handling of changing params of SVM"

        if self.retrainable:
            # we must assign it only if it is retrainable
            self.states.retrained = not newsvm or not newkernel

        # Train
        if __debug__:
            debug("SG", "%sTraining SG_SVM %s %s on data with labels %s" %
                  (("","Re-")[self.retrainable and self.states.retrained], self._kernel_type,
                   self.params, dataset.uniquelabels))

        self.__svm.train()

        # Report on training
        if __debug__:
            debug("SG_", "Done training SG_SVM %s on data with labels %s" %
                  (self._kernel_type, dataset.uniquelabels))
            if "SG__" in debug.active:
                trained_labels = self.__svm.classify().get_labels()
                debug("SG__", "Original labels: %s, Trained labels: %s" %
                              (dataset.labels, trained_labels))


    def _predict(self, data):
        """Predict values for the data
        """

        if __debug__:
            debug("SG_", "Initializing kernel with training/testing data")

        if self.retrainable:
            changed_testdata = self.__kernel_test is None or \
                               self.__wasChanged('test_samples', 2, data)

        if not self.retrainable or changed_testdata:
            testdata = _tosg(data)

        if not self.retrainable:
            # We can just reuse kernel used for training
            self.__kernel.init(self.__traindata, testdata)
        else:
            if changed_testdata:
                if __debug__:
                    debug("SG__",
                          "Re-creating testing kernel of %s giving "
                          "arguments %s" %
                          (`self._kernel_type`, self.__kernel_args))
                kernel_test = self._kernel_type(self.__traindata, testdata,
                                                *self.__kernel_args)
                _setdebug(kernel_test, 'Kernels')
                kernel_test_custom = shogun.Kernel.CustomKernel(self.__traindata, testdata)
                _setdebug(kernel_test, 'Kernels')
                self.__kernel_test = kernel_test_custom
                self.__kernel_test.set_full_kernel_matrix_from_full(
                    kernel_test.get_kernel_matrix())
            elif __debug__:
                debug("SG__", "Re-using testing kernel")

            assert(self.__kernel_test is not None)
            self.__svm.set_kernel(self.__kernel_test)

        if __debug__:
            debug("SG_", "Classifying testing data")

        # doesn't do any good imho although on unittests helps tiny bit... hm
        #self.__svm.init_kernel_optimization()
        values_ = self.__svm.classify()
        #if self.retrainable and not changed_testdata:
        #    import pydb
        #    pydb.debugger()
        values = values_.get_labels()

        if self.retrainable:
            # we must assign it only if it is retrainable
            self.states.retested = not changed_testdata
            if __debug__:
                debug("SG__", "Re-assigning learing kernel")
            # return back original kernel
            self.__svm.set_kernel(self.__kernel)

        if __debug__:
            debug("SG__", "Got values %s" % values)

        if ('regression' in self._clf_internals):
            predictions = values
        else:
            if len(self._labels_dict) == 2:
                predictions = 1.0 - 2*N.signbit(values)
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

        ## to avoid leaks with not yet properly fixed shogun
        if not self.retrainable:
            try:
                testdata.free_features()
            except:
                pass

        return predictions


    def untrain(self):
        super(SVM_SG_Modular, self).untrain()

        if not self.retrainable:
            if __debug__:
                debug("SG__", "Untraining %s and destroying sg's SVM" % self)

            self.__idhash = [None, None, None]  # samples, labels

            # to avoid leaks with not yet properly fixed shogun
            # XXX make it nice... now it is just stable ;-)
            if not self.__traindata is None:
                try:
                    try:
                        self.__traindata.free_features()
                    except:
                        pass
                    if __debug__:
                        if 'RETRAIN' in debug.active:
                            self.__trained = [None, None, None]
                    self.__traindataset = None
                    del self.__kernel
                    self.__kernel = None
                    self.__kernel_test = None
                    del self.__traindata
                    self.__traindata = None
                    del self.__svm
                    self.__svm = None
                except:
                    pass

            if __debug__:
                debug("SG__",
                      "Done untraining %(self)s and destroying sg's SVM",
                      msgargs=locals())
        elif __debug__:
            debug("SG__", "Not untraining %(self)s since it is retrainable",
                  msgargs=locals())


    def getSensitivityAnalyzer(self, **kwargs):
        """Returns an appropriate SensitivityAnalyzer."""
        if self._kernel_type_literal == 'linear':
            return ShogunLinearSVMWeights(self, **kwargs)
        else:
            raise NotImplementedError, 'Non-linear SVM sensitivity is not yet here'



    svm = property(fget=lambda self: self.__svm)
    """Access to the SVM model."""

    traindataset = property(fget=lambda self: self.__traindataset)
    """Dataset which was used for training

    TODO -- might better become state variable I guess"""



class LinearSVM(SVM_SG_Modular):

    def __init__(self, **kwargs):
        """
        """
        # init base class
        SVM_SG_Modular.__init__(self, kernel_type='linear', **kwargs)


# We don't have nu-SVM here
LinearCSVMC = LinearSVM


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
        # XXX Hm... it might make sense to unify access functions
        # naming across our swig libsvm wrapper and sg access
        # functions for svm
        svm = self.clf.svm
        if isinstance(svm, shogun.Classifier.MultiClassSVM):
            sens = []
            for i in xrange(svm.get_num_svms()):
                sens.append(self.__sg_helper(svm.get_svm(i)))
        else:
            sens = self.__sg_helper(svm)
        return N.asarray(sens)

