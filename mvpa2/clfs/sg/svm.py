# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrap the libsvm package into a very simple class interface."""

__docformat__ = 'restructuredtext'


_DEV__doc__ = """

TODOs:
 * dual-license under GPL for use of SG?
 * for recent versions add ability to specify/parametrize normalization
   scheme for the kernel, and reuse 'scale' now for the normalizer
 * Add support for simplified linear classifiers (which do not require
   storing all training SVs/samples to make classification in predict())
"""

import numpy as np

from mvpa2 import _random_seed

# Rely on SG
from mvpa2.base import externals, warning
if externals.exists('shogun', raise_=True):
    import shogun.Features
    import shogun.Classifier
    import shogun.Regression
    #import shogun.Kernel
    import shogun.Library
    from mvpa2.kernels.sg import SGKernel, LinearSGKernel
    # set the default kernel here, to be able to import this module
    # when building the docs without SG
    _default_kernel_class_ = LinearSGKernel

    # Figure out debug IDs once and for all
    if hasattr(shogun.Classifier, 'M_DEBUG'):
        _M_DEBUG = shogun.Classifier.M_DEBUG
        _M_ERROR = shogun.Classifier.M_ERROR
        _M_GCDEBUG = None
    elif hasattr(shogun.Classifier, 'MSG_DEBUG'):
        _M_DEBUG = shogun.Classifier.MSG_DEBUG
        _M_ERROR = shogun.Classifier.MSG_ERROR
    else:
        _M_DEBUG, _M_ERROR = None, None
        warning("Could not figure out debug IDs within shogun. "
                "No control over shogun verbosity would be provided")
    # Highest level
    if hasattr(shogun.Classifier, 'MSG_GCDEBUG'):
        _M_GCDEBUG = shogun.Classifier.MSG_GCDEBUG
    else:
        _M_GCDEBUG = None

else:
    # set a fake default kernel here, to be able to import this module
    # when building the docs without SG
    _default_kernel_class_ = None


import operator

from mvpa2.base.param import Parameter
from mvpa2.misc.attrmap import AttributeMap

from mvpa2.clfs.base import accepts_dataset_as_samples, \
     accepts_samples_as_dataset
from mvpa2.base.learner import FailedToTrainError
from mvpa2.clfs.meta import MulticlassClassifier
from mvpa2.clfs._svmbase import _SVM
from mvpa2.base.state import ConditionalAttribute
from mvpa2.measures.base import Sensitivity

from sens import *

from mvpa2.support.due import due, BibTeX

if __debug__:
    from mvpa2.base import debug


def seed(random_seed):
    if __debug__:
        debug('SG', "Seeding shogun's RNG with %s" % random_seed)
    try:
        # reuse the same seed for shogun
        shogun.Library.Math_init_random(random_seed)
    except Exception, e:
        warning('Shogun cannot be seeded due to %s' % (e,))

seed(_random_seed)

def _setdebug(obj, partname):
    """Helper to set level of debugging output for SG
    Parameters
    ----------
    obj
      In SG debug output seems to be set per every object
    partname : str
      For what kind of object we are talking about... could be automated
      later on (TODO)
    """
    if _M_DEBUG is None:
        return
    debugname = "SG_%s" % partname.upper()

    switch = {True: (_M_DEBUG, 'M_DEBUG', "enable"),
              False: (_M_ERROR, 'M_ERROR', "disable"),
              'GCDEBUG': (_M_GCDEBUG, 'M_GCDEBUG', "enable")}

    if __debug__:
        if 'SG_GC' in debug.active:
            key = 'GCDEBUG'
        else:
            key = debugname in debug.active
    else:
        key = False

    sglevel, slevel, progressfunc = switch[key]

    if __debug__ and 'SG_' in debug.active:
        debug("SG_", "Setting verbosity for shogun.%s instance: %s to %s" %
              (partname, `obj`, slevel))
    if sglevel is not None:
        obj.io.set_loglevel(sglevel)
    if __debug__ and 'SG_LINENO' in debug.active:
        try:
            obj.io.enable_file_and_line()
        except AttributeError, e:
            warning("Cannot enable SG_LINENO debug target for shogun %s"
                    % externals.versions['shogun'])
    try:
        exec "obj.io.%s_progress()" % progressfunc
    except:
        warning("Shogun version %s has no way to enable progress" +
                " reports" % externals.versions['shogun'])


# Still in use by non-kernel classifiers, e.g. SVMOcas
def _tosg(data):
    """Draft helper function to convert data we have into SG suitable format

    TODO: Remove once kernels are implemented here (or, possibly for non-kernel
    solvers, modify?)
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
    __default_kernel_class__ = _default_kernel_class_
    num_threads = Parameter(1,
                            min=1,
                            doc='Number of threads to utilize')

    _KNOWN_PARAMS = [ 'epsilon' ]

    __tags__ = _SVM.__tags__ + [ 'sg', 'retrainable' ]

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
    _KNOWN_SENSITIVITIES={'linear':LinearSVMWeights,
                          }
    _KNOWN_IMPLEMENTATIONS = {}
    if externals.exists('shogun', raise_=True):
        _KNOWN_IMPLEMENTATIONS = {
            "libsvm" : (shogun.Classifier.LibSVM, ('C',),
                       ('multiclass', 'binary'),
                        "LIBSVM's C-SVM (L2 soft-margin SVM)"),
            "gmnp" : (shogun.Classifier.GMNPSVM, ('C',),
                     ('multiclass', 'binary'),
                      "Generalized Nearest Point Problem SVM"),
            # XXX should have been GPDT, shogun has it fixed since some version
            "gpbt" : (shogun.Classifier.GPBTSVM, ('C',), ('binary',),
                      "Gradient Projection Decomposition Technique for " \
                      "large-scale SVM problems"),
            "gnpp" : (shogun.Classifier.GNPPSVM, ('C',), ('binary',),
                      "Generalized Nearest Point Problem SVM"),

            ## TODO: Needs sparse features...
            # "svmlin" : (shogun.Classifier.SVMLin, ''),
            # "liblinear" : (shogun.Classifier.LibLinear, ''),
            # "subgradient" : (shogun.Classifier.SubGradientSVM, ''),
            ## good 2-class linear SVMs
            # "ocas" : (shogun.Classifier.SVMOcas, ''),
            # "sgd" : ( shogun.Classifier.SVMSGD, ''),

            # regressions
            "libsvr": (shogun.Regression.LibSVR, ('C', 'tube_epsilon',),
                      ('regression',),
                       "LIBSVM's epsilon-SVR"),
            }


    def __init__(self, **kwargs):
        """Interface class to Shogun's classifiers and regressions.

        Default implementation is 'libsvm'.
        """


        svm_impl = kwargs.get('svm_impl', 'libsvm').lower()
        kwargs['svm_impl'] = svm_impl

        # init base class
        _SVM.__init__(self, **kwargs)

        self.__svm = None
        """Holds the trained svm."""
        self.__svm_apply = None
        """Compatibility convenience to bind to the classify/apply method
           of __svm"""
        # Need to store original data...
        # TODO: keep 1 of them -- just __traindata or __traindataset
        # For now it is needed for computing sensitivities
        self.__traindataset = None

        # internal SG swig proxies
        self.__traindata = None
        self.__kernel = None
        self.__kernel_test = None
        self.__testdata = None

        # remove kernel-based for some
        # TODO RF: provide separate handling for non-kernel machines
        if svm_impl in ['svmocas']:
            if not (self.__kernel is None
                    or self.__kernel.__kernel_name__ == 'linear'):
                raise ValueError(
                    "%s is inherently linear, thus provided kernel %s "
                    "is of no effect" % (svm_impl, self.__kernel))
            self.__tags__.pop(self.__tags__.index('kernel-based'))
            self.__tags__.pop(self.__tags__.index('retrainable'))

    # TODO: integrate with kernel framework
    #def __condition_kernel(self, kernel):
        ## XXX I thought that it is needed only for retrainable classifier,
        ##     but then krr gets confused, and svrlight needs it to provide
        ##     meaningful results even without 'retraining'
        #if self._svm_impl in ['svrlight', 'lightsvm']:
            #try:
                #kernel.set_precompute_matrix(True, True)
            #except Exception, e:
                ## N/A in shogun 0.9.1... TODO: RF
                #if __debug__:
                    #debug('SG_', "Failed call to set_precompute_matrix for %s: %s"
                          #% (self, e))


    @due.dcite(
        BibTeX("""
@article{Sonnenburg+2010:Shogun,
 author = {Sonnenburg, Sören and Rätsch, Gunnar and Henschel, Sebastian
           and Widmer, Christian and Behr, Jonas and Zien, Alexander
           and Bona, Fabio de and Binder, Alexander and Gehl, Christian
           and Franc, Vojtěch},
 title = {The SHOGUN Machine Learning Toolbox},
 journal = {J. Mach. Learn. Res.},
 issue_date = {3/1/2010},
 volume = {11},
 month = aug,
 year = {2010},
 issn = {1532-4435},
 pages = {1799--1802},
 numpages = {4},
 url = {http://dl.acm.org/citation.cfm?id=1756006.1859911},
 acmid = {1859911},
 publisher = {JMLR.org},
}"""),
        description="Shogun: Machine learning toolbox. SVM implementations",
        path="shogun",
        version=externals.versions['shogun'],
        tags=["implementation"])
    def _train(self, dataset):
        """Train SVM
        """
        super(SVM, self)._train(dataset)
        # XXX watchout
        # self.untrain()
        newkernel, newsvm = False, False
        # local bindings for faster lookup
        params = self.params
        retrainable = self.params.retrainable

        targets_sa_name = self.get_space()    # name of targets sa
        targets_sa = dataset.sa[targets_sa_name] # actual targets sa

        if retrainable:
            _changedData = self._changedData

        # LABELS
        ul = None
        self.__traindataset = dataset


        # OK -- we have to map labels since
        #  binary ones expect -1/+1
        #  Multiclass expect labels starting with 0, otherwise they puke
        #   when ran from ipython... yikes
        if __debug__:
            debug("SG_", "Creating labels instance")

        if self.__is_regression__:
            labels_ = np.asarray(targets_sa.value, dtype='double')
        else:
            ul = targets_sa.unique
            # ul.sort()

            if len(ul) == 2:
                # assure that we have -1/+1
                _labels_dict = {ul[0]:-1.0, ul[1]:+1.0}
            elif len(ul) < 2:
                raise FailedToTrainError, \
                      "We do not have 1-class SVM brought into SG yet"
            else:
                # can't use plain enumerate since we need them swapped
                _labels_dict = dict([ (u, i) for i, u in enumerate(ul)])

            # Create SG-customized attrmap to assure -1 / +1 if necessary
            self._attrmap = AttributeMap(_labels_dict, mapnumeric=True)

            if __debug__:
                debug("SG__", "Mapping labels using dict %s" % _labels_dict)
            labels_ = self._attrmap.to_numeric(targets_sa.value).astype(float)

        labels = shogun.Features.Labels(labels_)
        _setdebug(labels, 'Labels')


        # KERNEL

        # XXX cruel fix for now... whole retraining business needs to
        # be rethought
        if retrainable:
            _changedData['kernel_params'] = _changedData.get('kernel_params', False)

        # TODO: big RF to move non-kernel classifiers away
        if 'kernel-based' in self.__tags__ and (not retrainable
               or _changedData['traindata'] or _changedData['kernel_params']):
            # If needed compute or just collect arguments for SVM and for
            # the kernel

            if retrainable and __debug__:
                if _changedData['traindata']:
                    debug("SG",
                          "Re-Creating kernel since training data has changed")

                if _changedData['kernel_params']:
                    debug("SG",
                          "Re-Creating kernel since params %s has changed" %
                          _changedData['kernel_params'])


            k = self.params.kernel
            k.compute(dataset)
            self.__kernel = kernel = k.as_raw_sg()

            newkernel = True
            self.kernel_params.reset()  # mark them as not-changed
            #_setdebug(kernel, 'Kernels')

            #self.__condition_kernel(kernel)
            if retrainable:
                if __debug__:
                    debug("SG_", "Resetting test kernel for retrainable SVM")
                self.__kernel_test = None

        # TODO -- handle _changedData['params'] correctly, ie without recreating
        # whole SVM
        Cs = None
        if not retrainable or self.__svm is None or _changedData['params']:
            # SVM
            if 'C' in self.params:
                Cs = self._get_cvec(dataset)

                # XXX do not jump over the head and leave it up to the user
                #     ie do not rescale automagically by the number of samples
                #if len(Cs) == 2 and not ('regression' in self.__tags__) and len(ul) == 2:
                #    # we were given two Cs
                #    if np.max(C) < 0 and np.min(C) < 0:
                #        # and both are requested to be 'scaled' TODO :
                #        # provide proper 'features' to the parameters,
                #        # so we could specify explicitely if to scale
                #        # them by the number of samples here
                #        nl = [np.sum(labels_ == _labels_dict[l]) for l in ul]
                #        ratio = np.sqrt(float(nl[1]) / nl[0])
                #        #ratio = (float(nl[1]) / nl[0])
                #        Cs[0] *= ratio
                #        Cs[1] /= ratio
                #        if __debug__:
                #            debug("SG_", "Rescaled Cs to %s to accomodate the "
                #                  "difference in number of training samples" %
                #                  Cs)

            # Choose appropriate implementation
            svm_impl_class = self.__get_implementation(ul)

            if __debug__:
                debug("SG", "Creating SVM instance of %s" % `svm_impl_class`)

            if self._svm_impl in ['libsvr', 'svrlight']:
                # for regressions constructor a bit different
                self.__svm = svm_impl_class(Cs[0], self.params.tube_epsilon, self.__kernel, labels)
                # we need to set epsilon explicitly
                self.__svm.set_epsilon(self.params.epsilon)
            elif self._svm_impl in ['krr']:
                self.__svm = svm_impl_class(self.params.tau, self.__kernel, labels)
            elif 'kernel-based' in self.__tags__:
                self.__svm = svm_impl_class(Cs[0], self.__kernel, labels)
                self.__svm.set_epsilon(self.params.epsilon)
            else:
                traindata_sg = _tosg(dataset.samples)
                self.__svm = svm_impl_class(Cs[0], traindata_sg, labels)
                self.__svm.set_epsilon(self.params.epsilon)

            # To stay compatible with versions across API changes in sg 1.0.0
            self.__svm_apply = externals.versions['shogun'] >= '1' \
                               and self.__svm.apply \
                               or  self.__svm.classify # the last one for old API

            # Set shrinking
            if 'shrinking' in params:
                shrinking = params.shrinking
                if __debug__:
                    debug("SG_", "Setting shrinking to %s" % shrinking)
                self.__svm.set_shrinking_enabled(shrinking)

            if Cs is not None and len(Cs) == 2:
                if __debug__:
                    debug("SG_", "Since multiple Cs are provided: %s, assign them" % Cs)
                self.__svm.set_C(Cs[0], Cs[1])

            self.params.reset()  # mark them as not-changed
            newsvm = True
            _setdebug(self.__svm, 'SVM')
            # Set optimization parameters
            if 'tube_epsilon' in self.params and \
                   hasattr(self.__svm, 'set_tube_epsilon'):
                self.__svm.set_tube_epsilon(self.params.tube_epsilon)
            self.__svm.parallel.set_num_threads(self.params.num_threads)
        else:
            if __debug__:
                debug("SG_", "SVM instance is not re-created")
            if _changedData['targets']:          # labels were changed
                if __debug__: debug("SG__", "Assigning new labels")
                self.__svm.set_labels(labels)
            if newkernel:               # kernel was replaced
                if __debug__: debug("SG__", "Assigning new kernel")
                self.__svm.set_kernel(self.__kernel)
            assert(_changedData['params'] is False)  # we should never get here

        if retrainable:
            # we must assign it only if it is retrainable
            self.ca.retrained = not newsvm or not newkernel

        # Train
        if __debug__ and 'SG' in debug.active:
            if not self.__is_regression__:
                lstr = " with labels %s" % targets_sa.unique
            else:
                lstr = ""
            debug("SG", "%sTraining %s on data%s" %
                  (("","Re-")[retrainable and self.ca.retrained],
                   self, lstr))

        self.__svm.train()

        if __debug__:
            debug("SG_", "Done training SG_SVM %s" % self)

        # Report on training
        if (__debug__ and 'SG__' in debug.active) or \
           self.ca.is_enabled('training_stats'):
            if __debug__:
                debug("SG_", "Assessing predictions on training data")
            trained_targets = self.__svm_apply().get_labels()

        else:
            trained_targets = None

        if __debug__ and "SG__" in debug.active:
            debug("SG__", "Original labels: %s, Trained labels: %s" %
                  (targets_sa.value, trained_targets))

        # Assign training confusion right away here since we are ready
        # to do so.
        # XXX TODO use some other conditional attribute like 'trained_targets' and
        #     use it within base Classifier._posttrain to assign predictions
        #     instead of duplicating code here
        # XXX For now it can be done only for regressions since labels need to
        #     be remapped and that becomes even worse if we use regression
        #     as a classifier so mapping happens upstairs
        if self.__is_regression__ and self.ca.is_enabled('training_stats'):
            self.ca.training_stats = self.__summary_class__(
                targets=targets_sa.value,
                predictions=trained_targets)


    # XXX actually this is the beast which started this evil conversion
    #     so -- make use of dataset here! ;)
    @accepts_samples_as_dataset
    def _predict(self, dataset):
        """Predict values for the data
        """

        retrainable = self.params.retrainable

        if retrainable:
            changed_testdata = self._changedData['testdata'] or \
                               self.__kernel_test is None

        if not retrainable:
            if __debug__:
                debug("SG__",
                      "Initializing SVMs kernel of %s with training/testing samples"
                      % self)
            self.params.kernel.compute(self.__traindataset, dataset)
            self.__kernel_test = self.params.kernel.as_sg()._k
            # We can just reuse kernel used for training
            #self.__condition_kernel(self.__kernel)

        else:
            if changed_testdata:
                #if __debug__:
                    #debug("SG__",
                          #"Re-creating testing kernel of %s giving "
                          #"arguments %s" %
                          #(`self._kernel_type`, self.__kernel_args))
                self.params.kernel.compute(self.__traindataset, dataset)

                #_setdebug(kernel_test, 'Kernels')

                #_setdebug(kernel_test_custom, 'Kernels')
                self.__kernel_test = self.params.kernel.as_raw_sg()

            elif __debug__:
                debug("SG__", "Re-using testing kernel")

        assert(self.__kernel_test is not None)

        if 'kernel-based' in self.__tags__:
            self.__svm.set_kernel(self.__kernel_test)
            # doesn't do any good imho although on unittests helps tiny bit... hm
            #self.__svm.init_kernel_optimization()
            values_ = self.__svm_apply()
        else:
            testdata_sg = _tosg(dataset.samples)
            self.__svm.set_features(testdata_sg)
            values_ = self.__svm_apply()

        if __debug__:
            debug("SG_", "Classifying testing data")

        if values_ is None:
            raise RuntimeError, "We got empty list of values from %s" % self

        values = values_.get_labels()

        if retrainable:
            # we must assign it only if it is retrainable
            self.ca.repredicted = repredicted = not changed_testdata
            if __debug__:
                debug("SG__", "Re-assigning learing kernel. Repredicted is %s"
                      % repredicted)
            # return back original kernel
            if 'kernel-based' in self.__tags__:
                self.__svm.set_kernel(self.__kernel)

        if __debug__:
            debug("SG__", "Got values %s" % values)

        if (self.__is_regression__):
            predictions = values
        else:
            if len(self._attrmap.keys()) == 2:
                predictions = np.sign(values)
                # since np.sign(0) == 0
                predictions[predictions==0] = 1
            else:
                predictions = values

            # remap labels back adjusting their type
            # XXX YOH: This is done by topclass now (needs RF)
            #predictions = self._attrmap.to_literal(predictions)

            if __debug__:
                debug("SG__", "Tuned predictions %s" % predictions)

        # store conditional attribute
        # TODO: extract values properly for multiclass SVMs --
        #       ie 1 value per label or pairs for all 1-vs-1 classifications
        self.ca.estimates = values

        ## to avoid leaks with not yet properly fixed shogun
        if not retrainable:
            try:
                testdata.free_features()
            except:
                pass

        return predictions


    def _untrain(self):
        super(SVM, self)._untrain()
        # untrain/clean the kernel -- we might not allow to drag SWIG
        # instance around BUT XXX -- make it work fine with
        # CachedKernel -- we might not want to fully "untrain" in such
        # case
        self.params.kernel.cleanup()    # XXX unify naming
        if not self.params.retrainable:
            if __debug__:
                debug("SG__", "Untraining %(clf)s and destroying sg's SVM",
                      msgargs={'clf':self})

            # to avoid leaks with not yet properly fixed shogun
            # XXX make it nice... now it is just stable ;-)
            if True: # not self.__traindata is None:
                if True:
                # try:
                    if self.__kernel is not None:
                        del self.__kernel
                        self.__kernel = None

                    if self.__kernel_test is not None:
                        del self.__kernel_test
                        self.__kernel_test = None

                    if self.__svm is not None:
                        del self.__svm
                        self.__svm = None
                        self.__svm_apply = None

                    if self.__traindata is not None:
                        # Let in for easy demonstration of the memory leak in shogun
                        #for i in xrange(10):
                        #    debug("SG__", "cachesize pre free features %s" %
                        #          (self.__svm.get_kernel().get_cache_size()))
                        self.__traindata.free_features()
                        del self.__traindata
                        self.__traindata = None

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
        if self.__is_regression__ or len(ul) == 2:
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
                      (self._svm_impl,
                       self.__traindataset.sa[self.get_space()].unique)
            if __debug__:
                debug("SG_", "Using %s for multiclass data of %s" %
                      (svm_impl_class, self._svm_impl))

        return svm_impl_class


    svm = property(fget=lambda self: self.__svm)
    """Access to the SVM model."""

    traindataset = property(fget=lambda self: self.__traindataset)
    """Dataset which was used for training

    TODO -- might better become conditional attribute I guess"""



# Conditionally make some of the implementations available if they are
# present in the present shogun
if externals.exists('shogun'):
    for name, item, params, descr in \
            [('mpd', "shogun.Classifier.MPDSVM", "('C',), ('binary',)",
              "MPD classifier from shogun"),
             ('lightsvm', "shogun.Classifier.SVMLight", "('C',), ('binary',)",
              "SVMLight classification http://svmlight.joachims.org/"),
             ('svrlight', "shogun.Regression.SVRLight", "('C','tube_epsilon',), ('regression',)",
              "SVMLight regression http://svmlight.joachims.org/"),
             ('krr', "shogun.Regression.KRR", "('tau',), ('regression',)",
              "Kernel Ridge Regression"),
             ('svmocas', "shogun.Classifier.SVMOcas", "('C',), ('binary', 'linear')",
              "SVM with OCAS (Optimized Cutting Plane Algorithm) solver"),
             ]:
        if externals.exists('shogun.%s' % name):
            exec "SVM._KNOWN_IMPLEMENTATIONS[\"%s\"] = (%s, %s, \"%s\")" \
                 % (name, item, params, descr)

# Assign SVM class to limited set of LinearSVMWeights
LinearSVMWeights._LEGAL_CLFS = [SVM]
