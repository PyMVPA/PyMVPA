#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.algorithms.datameasure import ClassifierBasedSensitivityAnalyzer
from mvpa.misc import warning
from mvpa.misc.state import StateVariable

# Import libsvm SVM implementation
import mvpa.clfs.libsvm.svm as svm_libsvm

try:
    import mvpa.clfs.sg.svm as svm_sg
    import shogun.Classifier

    __sg_present = True
except ImportError:
    # no shogun library is available, thus no sensitivity could be even checked
    # for
    __sg_present = False

if __debug__:
    from mvpa.misc import debug


class LinearSVMWeights(ClassifierBasedSensitivityAnalyzer):
    """`SensitivityAnalyzer` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """

    offsets = StateVariable(enabled=True,
                            doc="Offsets of separating hyperplane")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf : LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        # init base classes first
        ClassifierBasedSensitivityAnalyzer.__init__(self, clf, **kwargs)

        # poor man dispatch table
        if isinstance(clf, svm_libsvm.LinearSVM):
            self.__sens = self.__libsvm
        elif isinstance(clf, svm_sg.SVM_SG_Modular):
            self.__sens = self.__sg
        else:
            raise ValueError, "Don't know how to compute Linear SVM " + \
                  "sensitivity for clf %s of type %s." % \
                  (`clf`, `type(clf)`)


    def __libsvm(self, dataset, callables=[]):
        if self.clf.model.nr_class != 2:
            warning("You are estimating sensitivity for SVM %s trained on %d" %
                    (`self.clf`, self.clf.model.nr_class) +
                    " classes. Make sure that it is what you intended to do" )

        svcoef = N.matrix(self.clf.model.getSVCoef())
        svs = N.matrix(self.clf.model.getSV())
        rhos = N.array(self.clf.model.getRho())
        if __debug__:
            debug('SVM',
                  "Extracting weigts for %d-class SVM: #SVs=%s, " %
                  (self.clf.model.nr_class, `self.clf.model.getNSV()`) +
                  " SVcoefshape=%s SVs.shape=%s Rhos=%s" %\
                  (svcoef.shape, svs.shape, rhos))

        self.offsets = rhos
        # XXX yoh: .mean() is effectively
        # averages across "sensitivities" of all paired classifiers (I
        # think). See more info on this topic in svm.py on how sv_coefs
        # are stored
        #
        # First multiply SV coefficients with the actuall SVs to get
        # weighted impact of SVs on decision, then for each feature
        # take absolute mean across SVs to get a single weight value
        # per feature

        return N.abs((svcoef * svs).mean(axis=0).A1)

    def __sg(self, dataset, callables=[]):
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
            anal = selectAnalyzer(self.__mclf, basic_analyzer=self)
            if __debug__:
                debug('SVM',
                      '! Delegating computing sensitivity to %s' % `anal`)
            return anal(dataset, callables)

        if isinstance(self.clf.svm, shogun.Classifier.MultiClassSVM):
            raise NotImplementedError
        else:
            svm = self.clf.svm
            self.offsets = svm.get_bias()
            svcoef = N.matrix(svm.get_alphas())
            svnums = svm.get_support_vectors()
            svs = self.clf.traindataset.samples[svnums,:]
            res = (svcoef * svs).mean(axis=0).A1
            return N.abs(res)


    def _call(self, dataset, callables=[]):
        """Extract weights from Linear SVM classifier.
        """
        return self.__sens(dataset, callables)
