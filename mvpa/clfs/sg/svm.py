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

from sens import *

if __debug__:
    from mvpa.misc import debug




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
              False: (shogun.Kernel.M_ERROR, 'M_ERROR', "disable")}

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


class SVM(_SVM):
    """Support Vector Machine Classifier(s) based on Shogun

    This is a simple base interface
    """

    num_threads = Parameter(1,
                            min=1,
                            descr='Number of threads to utilize')

    # NOTE: gamma is width in SG notation for RBF(Gaussian)
    _KERNELS = { "linear": (shogun.Kernel.LinearKernel,   ('scale',), LinearSVMWeights),
                 "rbf" :   (shogun.Kernel.GaussianKernel, ('gamma',), None),
                 "rbfshift" : (shogun.Kernel.GaussianShiftKernel, ('gamma', 'max_shift', 'shift_step'), None),
                 "sigmoid" : (shogun.Kernel.SigmoidKernel, ('cache_size', 'gamma', 'coef0'), None),
                }

    _KNOWN_PARAMS = [ 'epsilon' ]
    _KNOWN_KERNEL_PARAMS = [ ]

    _clf_internals = _SVM._clf_internals + [ 'sg', 'retrainable' ]


    # Some words of wisdom from shogun author:
    # XXX remove after proper comments added to implementations
    """
    If you'd like to train linear SVMs use SGD or OCAS. These are (I am
    serious) the fastest linear SVM-solvers to date. (OCAS cannot do SVMs
    with standard additive bias, but will L2 reqularize it - though it
    should not matter much in practice (although it will give slightly
    different solutions)). Note that SGD has no stopping criterion (you
    simply have to specify the number of iterations) and that OCAS has a
    different stopping condition than svmlight for example which may be more
    tight and more loose depending on the problem - I sugeest 1e-2 or 1e-3
    for epsilon.

    If you would like to train kernel SVMs use libsvm/gpdt/svmlight -
    depending on the problem one is faster than the other (hard to say when,
    I *think* when your dataset is very unbalanced chunking methods like
    svmlight/gpdt are better), for smaller problems definitely libsvm.

    If you use string kernels then gpdt/svmlight have a special 'linadd'
    speedup for this (requires sg 0.6.2 - there was some inefficiency in the
    code for python-modular before that). This is effective for big datasets
    and (I trained on 10 million strings based on this).

    And yes currently we only implemented parallel training for svmlight,
    however all SVMs can be evaluated in parallel.
    """
    _KNOWN_IMPLEMENTATIONS = {
        "libsvm" : (shogun.Classifier.LibSVM, ('C',), ('multiclass', 'binary'), ''),
        "gmnp" : (shogun.Classifier.GMNPSVM, ('C',), ('multiclass', 'binary'), ''),
        "mpd"  : (shogun.Classifier.MPDSVM, ('C',), ('binary',), ''),
        "gpbt" : (shogun.Classifier.GPBTSVM, ('C',), ('binary',), ''),
        "gnpp" : (shogun.Classifier.GNPPSVM, ('C',), ('binary',), ''),

        ## TODO: Needs sparse features...
        # "svmlin" : (shogun.Classifier.SVMLin, ''),
        # "liblinear" : (shogun.Classifier.LibLinear, ''),
        # "subgradient" : (shogun.Classifier.SubGradientSVM, ''),
        ## good 2-class linear SVMs
        # "ocas" : (shogun.Classifier.SVMOcas, ''),
        # "sgd" : ( shogun.Classifier.SVMSGD, ''),

        # regressions
        "libsvr": (shogun.Regression.LibSVR, ('C', 'tube_epsilon',), ('regression',), ''),
        "krr": (shogun.Regression.KRR, ('tau',), ('regression',), ''),
        }


    def __init__(self,
                 kernel_type='linear',
                 **kwargs):
        """This is the base class of all classifier that utilize so
        far just SVM classifiers provided by shogun.

        TODO Documentation if this all works ;-)
        """

        svm_impl = kwargs.get('svm_impl', 'libsvm').lower()
        kwargs['svm_impl'] = svm_impl

        #if svm_impl == 'krr':
        #    self._KNOWN_PARAMS = self._KNOWN_PARAMS[:] + ['tau']
        #if svm_impl in ['svrlight', 'libsvr']:
        #    self._KNOWN_PARAMS = self._KNOWN_PARAMS[:] + ['tube_epsilon']

        # init base class
        _SVM.__init__(self, kernel_type=kernel_type, **kwargs)

        self.__svm = None
        """Holds the trained svm."""

        # Need to store original data...
        # TODO: keep 1 of them -- just __traindata or __traindataset
        # For now it is needed for computing sensitivities
        self.__traindataset = None

        # internal SG swig proxies
        self.__traindata = None
        self.__kernel = None
        self.__kernel_test = None
        self.__testdata = None

        # if we do retraining -- store hashes
        # samples, labels, test_samples
        self.__idhash = [None, None, None]

        if __debug__:
            if 'CHECK_RETRAIN' in debug.active:
                # XXX it is not clear though if idhash is faster than
                # simple comparison of (dataset != __traineddataset).any(),
                # but if we like to get rid of __traineddataset then we should
                # use idhash anyways

                # XXX now we keep 2 copies of the data -- __traineddataset
                #     has it in SG format... uff
                #
                # samples, labels, test_samples, trainsamples_intest
                self.__trained = [None, None, None]


    def __wasChanged(self, descr, i, entry):
        """Check if given entry was changed from what known prior. If so -- store"""
        idhash_ = idhash(entry)
        changed = self.__idhash[i] != idhash_
        if __debug__ and 'CHECK_RETRAIN' in debug.active:
            changed2 = entry != self.__trained[i]
            if isinstance(changed2, N.ndarray):
                changed2 = changed2.any()
            if changed != changed2 and not changed:
                raise RuntimeError, \
                  'idhash found to be weak for %s. Though hashid %s!=%s %s, '\
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
        if self.params.retrainable:
            changed_params = self.params.whichSet()
            changed_kernel_params = self.kernel_params.whichSet()

        # XXX watchout
        # self.untrain()
        newkernel, newsvm = False, False
        if self.params.retrainable:
            if __debug__:
                debug('SG__', "IDHashes are %s" % (self.__idhash))
            changed_samples = self.__wasChanged('samples', 0, dataset.samples)
            changed_labels = self.__wasChanged('labels', 1, dataset.labels)

        # LABELS

        ul = None
        self.__traindataset = dataset


        # OK -- we have to map labels since
        #  binary ones expect -1/+1
        #  Multiclass expect labels starting with 0, otherwise they puke
        #   when ran from ipython... yikes
        if __debug__:
            debug("SG_", "Creating labels instance")

        if 'regression' in self._clf_internals:
            labels_ = N.asarray(dataset.labels, dtype='double')
        else:
            ul = dataset.uniquelabels
            ul.sort()

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
        if not self.params.retrainable or changed_samples or changed_kernel_params:
            # If needed compute or just collect arguments for SVM and for
            # the kernel
            kargs = []
            for arg in self._KERNELS[self._kernel_type_literal][1]:
                value = self.kernel_params[arg].value
                # XXX Unify damn automagic gamma value
                if arg == 'gamma' and value == 0.0:
                    value = self._getDefaultGamma(dataset)
                kargs += [value]

            if self.params.retrainable and __debug__:
                if changed_samples:
                    debug("SG",
                          "Re-Creating kernel since samples has changed")

                if changed_kernel_params:
                    debug("SG",
                          "Re-Creating kernel since params %s has changed" %
                          changed_kernel_params)

            # create training data
            if __debug__: debug("SG_", "Converting input data for shogun")
            self.__traindata = _tosg(dataset.samples)

            if __debug__:
                debug("SG", "Creating kernel instance of %s giving arguments %s" %
                      (`self._kernel_type`, kargs))

            self.__kernel = self._kernel_type(self.__traindata, self.__traindata,
                                              *kargs)
            newkernel = True
            self.kernel_params.reset()  # mark them as not-changed
            _setdebug(self.__kernel, 'Kernels')

            if self.params.retrainable:
                self.__kernel.set_precompute_matrix(True, True)
                if __debug__:
                    debug("SG_", "Resetting test kernel for retrainable SVM")
                self.__kernel_test = None
                self.__kernel_args = kargs

        # TODO -- handle changed_params correctly, ie without recreating
        # whole SVM
        if not self.params.retrainable or self.__svm is None or changed_params:
            # SVM
            if self.params.isKnown('C'):
                C = self.params.C
                if C<0:
                    C = self._getDefaultC(dataset.samples)*abs(C)
                    if __debug__:
                        debug("SG_", "Default C for %s was computed to be %s" %
                              (self.params.C, C))

            # Choose appropriate implementation
            svm_impl_class = self.__get_implementation(ul)

            if __debug__:
                debug("SG", "Creating SVM instance of %s" % `svm_impl_class`)

            if self._svm_impl in ['libsvr', 'svrlight']:
                # for regressions constructor a bit different
                self.__svm = svm_impl_class(C, self.params.epsilon, self.__kernel, labels)
            elif self._svm_impl in ['krr']:
                self.__svm = svm_impl_class(self.params.tau, self.__kernel, labels)
            else:
                self.__svm = svm_impl_class(C, self.__kernel, labels)
                self.__svm.set_epsilon(self.params.epsilon)
            self.params.reset()  # mark them as not-changed
            newsvm = True
            _setdebug(self.__svm, 'SVM')
            # Set optimization parameters
            if self.params.isKnown('tube_epsilon') and \
                   hasattr(self.__svm, 'set_tube_epsilon'):
                self.__svm.set_tube_epsilon(self.params.tube_epsilon)
            self.__svm.parallel.set_num_threads(self.params.num_threads)
        else:
            if __debug__:
                debug("SG_", "SVM instance is not re-created")
            if changed_labels:          # labels were changed
                self.__svm.set_labels(labels)
            if newkernel:               # kernel was replaced
                self.__svm.set_kernel(self.__kernel)
            if changed_params:
                raise NotImplementedError, \
                      "Implement handling of changing params of SVM"

        if self.params.retrainable:
            # we must assign it only if it is retrainable
            self.states.retrained = not newsvm or not newkernel

        # Train
        if __debug__:
            debug("SG", "%sTraining %s on data with labels %s" %
                  (("","Re-")[self.params.retrainable and self.states.retrained], self,
                   dataset.uniquelabels))

        self.__svm.train()

        # Report on training
        if (__debug__ and 'SG__' in debug.active) or \
           self.states.isEnabled('training_confusion'):
            trained_labels = self.__svm.classify().get_labels()
        else:
            trained_labels = None

        if __debug__:
            debug("SG_", "Done training SG_SVM %s on data with labels %s" %
                  (self._kernel_type, dataset.uniquelabels))
            if "SG__" in debug.active:
                debug("SG__", "Original labels: %s, Trained labels: %s" %
                              (dataset.labels, trained_labels))

        # Assign training confusion right away here since we are ready
        # to do so.
        # XXX TODO use some other state variable like 'trained_labels' and
        #     use it within base Classifier._posttrain to assign predictions
        #     instead of duplicating code here
        if self.states.isEnabled('training_confusion'):
            self.states.training_confusion = self._summaryClass(
                targets=dataset.labels,
                predictions=trained_labels)

    def _predict(self, data):
        """Predict values for the data
        """

        if self.params.retrainable:
            changed_testdata = self.__wasChanged('test_samples', 2, data) or \
                               self.__kernel_test is None

        if not self.params.retrainable or changed_testdata:
            testdata = _tosg(data)

        if not self.params.retrainable:
            if __debug__:
                debug("SG__",
                      "Initializing SVMs kernel of %s with training/testing samples"
                      % self)
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
                _setdebug(kernel_test_custom, 'Kernels')
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
        if values_ is None:
            raise RuntimeError, "We got empty list of values from %s" % self

        values = values_.get_labels()

        if self.params.retrainable:
            # we must assign it only if it is retrainable
            self.states.retested = not changed_testdata
            if __debug__:
                debug("SG__", "Re-assigning learing kernel. Retested is %s"
                      % self.states.retested)
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
        if not self.params.retrainable:
            try:
                testdata.free_features()
            except:
                pass

        return predictions


    def untrain(self):
        super(SVM, self).untrain()
        if not self.params.retrainable:
            if __debug__:
                debug("SG__", "Untraining %(clf)s and destroying sg's SVM",
                      msgargs={'clf':self})

            self.__idhash = [None, None, None]  # samples, labels

            # to avoid leaks with not yet properly fixed shogun
            # XXX make it nice... now it is just stable ;-)
            if not self.__traindata is None:
                if True:
                # try:
                    if self.__kernel is not None:
                        del self.__kernel
                        self.__kernel = None

                    if self.__kernel_test is not None:
                        del __kernel_test
                        self.__kernel_test = None

                    if self.__svm is not None:
                        del self.__svm
                        self.__svm = None

                    if self.__traindata is not None:
                        # Let in for easy demonstration of the memory leak in shogun
                        #for i in xrange(10):
                        #    debug("SG__", "cachesize pre free features %s" %
                        #          (self.__svm.get_kernel().get_cache_size()))
                        self.__traindata.free_features()
                        del self.__traindata
                        self.__traindata = None

                    if __debug__:
                        if 'CHECK_RETRAIN' in debug.active:
                            self.__trained = [None, None, None]

                    self.__traindataset = None


                #except:
                #    pass

            if __debug__:
                debug("SG__",
                      "Done untraining %(self)s and destroying sg's SVM",
                      msgargs=locals())
        elif __debug__:
            debug("SG__", "Not untraining %(self)s since it is retrainable",
                  msgargs=locals())


    def __get_implementation(self, ul):
        if 'regression' in self._clf_internals or len(ul) == 2:
            svm_impl_class = SVM._KNOWN_IMPLEMENTATIONS[self._svm_impl][0]
        else:
            if self._svm_impl == 'libsvm':
                svm_impl_class = shogun.Classifier.LibSVMMultiClass
            elif self._svm_impl == 'gmnp':
                svm_impl_class = shogun.Classifier.GMNPSVM
            else:
                raise RuntimeError, \
                      "Shogun: Implementation %s doesn't handle multiclass " \
                      "data. Got labels %s. Use some other classifier" % \
                      (self._svm_impl, self.__traindataset.uniquelabels)
            if __debug__:
                debug("SG_", "Using %s for multiclass data of %s" %
                      (svm_impl_class, self._svm_impl))

        return svm_impl_class


    svm = property(fget=lambda self: self.__svm)
    """Access to the SVM model."""

    traindataset = property(fget=lambda self: self.__traindataset)
    """Dataset which was used for training

    TODO -- might better become state variable I guess"""



# Conditionally make some of the implementations available if they are
# present in the present shogun
for name, item, params, descr in \
        [('lightsvm', "shogun.Classifier.SVMLight", "('C',), ('binary',)",
          "SVMLight classification http://svmlight.joachims.org/"),
         ('svrlight', "shogun.Regression.SVRLight", "('C','tube_epsilon',), ('regression',)",
          "SVMLight regression http://svmlight.joachims.org/")]:
    if externals.exists('shogun.%s' % name):
        exec "SVM._KNOWN_IMPLEMENTATIONS[\"%s\"] = (%s, %s, \"%s\")" % (name, item, params, descr)
