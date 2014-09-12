# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provide sensitivity measures for libsvm's SVM."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import warning
from mvpa2.base.state import ConditionalAttribute
from mvpa2.base.param import Parameter
from mvpa2.base.types import asobjarray
from mvpa2.measures.base import Sensitivity
from mvpa2.datasets.base import Dataset

if __debug__:
    from mvpa2.base import debug

class LinearSVMWeights(Sensitivity):
    """`SensitivityAnalyzer` for the LIBSVM implementation of a linear SVM.
    """

    _ATTRIBUTE_COLLECTIONS = ['params']

    split_weights = Parameter(False, constraints='bool',
                  doc="If binary classification either to sum SVs per each "
                      "class separately.  Note: be careful with interpretation"
                      " of the values")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        Parameters
        ----------
        clf : LinearSVM
          classifier to use. Only classifiers sub-classed from
          `LinearSVM` may be used.
        """
        # init base classes first
        Sensitivity.__init__(self, clf, **kwargs)


    def _call(self, dataset, callables=[]):
        # local bindings
        clf = self.clf
        model = clf.model

        # Labels for sensitivities to be returned
        sens_labels = None

        if clf.__is_regression__:
            nr_class = None
            svm_labels = None           # shouldn't bother to provide "targets" for regressions
        else:
            nr_class = model.nr_class
            svm_labels = model.labels

        # No need to warn since now we by default we do not do
        # anything evil and provide labels -- so it is up for a user
        # to decide either he wants to do something silly
        #if nr_class != 2:
        #    warning("You are estimating sensitivity for SVM %s trained on %d" %
        #            (str(clf), nr_class) +
        #            " classes. Make sure that it is what you intended to do" )

        svcoef = np.matrix(model.get_sv_coef())
        svs = np.matrix(model.get_sv())
        rhos = np.asarray(model.get_rho())

        if self.params.split_weights:
            if nr_class != 2:
                raise NotImplementedError, \
                      "Cannot compute per-class weights for" \
                      " non-binary classification task"
            # libsvm might have different idea on the ordering
            # of labels, so we would need to map them back explicitely
            ds_labels = list(dataset.sa[clf.get_space()].unique) # labels in the dataset
            senses = [None for i in ds_labels]
            # first label is given positive value
            for i, (c, l) in enumerate( [(svcoef > 0, lambda x: x),
                                         (svcoef < 0, lambda x: x*-1)] ):
                # convert to array, and just take the meaningful dimension
                c_ = c.A[0]
                # NOTE svm_labels are numerical; ds_labels are literal
                senses[ds_labels.index(
                            clf._attrmap.to_literal(svm_labels[i]))] = \
                                (l(svcoef[:, c_] * svs[c_, :])).A[0]
            weights = np.array(senses)
            sens_labels = svm_labels
        else:
            # XXX yoh: .mean() is effectively
            # averages across "sensitivities" of all paired classifiers (I
            # think). See more info on this topic in svm.py on how sv_coefs
            # are stored
            #
            # First multiply SV coefficients with the actual SVs to get
            # weighted impact of SVs on decision, then for each feature
            # take mean across SVs to get a single weight value
            # per feature
            if nr_class is None or nr_class <= 2:
                # as simple as this
                weights = (svcoef * svs).A
                # and only in case of classification
                if nr_class:
                    # ??? First label seems corresponds to positive
                    sens_labels = [tuple(svm_labels[::-1])]
            else:
                # we need to compose correctly per each pair of classifiers.
                # See docstring for get_sv_coef for more details on internal
                # structure of bloody storage

                # total # of pairs
                npairs = nr_class * (nr_class-1)/2
                # # of SVs in each class
                NSVs_perclass = model.get_n_sv()
                # indices where each class starts in each row of SVs
                # name is after similar variable in libsvm internals
                nz_start = np.cumsum([0] + NSVs_perclass[:-1])
                nz_end = nz_start + NSVs_perclass
                # reserve storage
                weights = np.zeros((npairs, svs.shape[1]))
                ipair = 0               # index of the pair
                """
                // classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]
                """
                sens_labels = []
                for i in xrange(nr_class):
                    for j in xrange(i+1, nr_class):
                        weights[ipair, :] = np.asarray(
                            svcoef[j-1, nz_start[i]:nz_end[i]]
                            * svs[nz_start[i]:nz_end[i]]
                            +
                            svcoef[i, nz_start[j]:nz_end[j]]
                            * svs[nz_start[j]:nz_end[j]]
                            )
                        # ??? First label corresponds to positive
                        # that is why [j], [i]
                        sens_labels += [(svm_labels[j], svm_labels[i])]
                        ipair += 1      # go to the next pair
                assert(ipair == npairs)

        if __debug__ and 'SVM' in debug.active:
            if nr_class:
                nsvs = model.get_n_sv()
            else:
                nsvs = model.get_total_n_sv()
            if clf.__is_regression__:
                svm_type = clf._svm_impl # type of regression
            else:
                svm_type = '%d-class SVM(%s)' % (nr_class, clf._svm_impl)
            debug('SVM',
                  "Extracting weights for %s: #SVs=%s, " % \
                  (svm_type, nsvs) + \
                  " SVcoefshape=%s SVs.shape=%s Rhos=%s." % \
                  (svcoef.shape, svs.shape, rhos) + \
                  " Result: min=%f max=%f" % (np.min(weights), np.max(weights)))

        ds_kwargs = {}
        if nr_class:          # for classification only
            # and we should have prepared the labels
            assert(sens_labels is not None)

            if len(clf._attrmap):
                if isinstance(sens_labels[0], tuple):
                    sens_labels = asobjarray(sens_labels)
                sens_labels = clf._attrmap.to_literal(sens_labels, recurse=True)

            # NOTE: `weights` is already and always 2D
            ds_kwargs = dict(sa={clf.get_space(): sens_labels})

        weights_ds = Dataset(weights, **ds_kwargs)
        weights_ds.sa['biases'] = rhos
        return weights_ds

    _customizeDocInherit = True
