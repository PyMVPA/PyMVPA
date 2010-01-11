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

import numpy as N

from mvpa.base import warning
from mvpa.misc.state import StateVariable
from mvpa.misc.param import Parameter
from mvpa.measures.base import Sensitivity
from mvpa.datasets.base import Dataset

if __debug__:
    from mvpa.base import debug

class LinearSVMWeights(Sensitivity):
    """`SensitivityAnalyzer` for the LIBSVM implementation of a linear SVM.
    """

    _ATTRIBUTE_COLLECTIONS = ['params']

    # XXX TODO: should become just as sa may be?
    biases = StateVariable(enabled=True,
                           doc="Offsets of separating hyper-planes")

    split_weights = Parameter(False, allowedtype='bool',
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
        nr_class = model.nr_class
        svm_labels = model.labels

        # Labels for sensitivities to be returned
        sens_labels = None

        # No need to warn since now we by default we do not do
        # anything evil and provide labels -- so it is up for a user
        # to decide either he wants to do something silly
        #if nr_class != 2:
        #    warning("You are estimating sensitivity for SVM %s trained on %d" %
        #            (str(clf), nr_class) +
        #            " classes. Make sure that it is what you intended to do" )

        svcoef = N.matrix(model.getSVCoef())
        svs = N.matrix(model.getSV())
        rhos = N.asarray(model.getRho())

        self.states.biases = rhos
        if self.params.split_weights:
            if nr_class != 2:
                raise NotImplementedError, \
                      "Cannot compute per-class weights for" \
                      " non-binary classification task"
            # libsvm might have different idea on the ordering
            # of labels, so we would need to map them back explicitely
            ds_labels = list(dataset.sa['labels'].unique) # labels in the dataset
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
            weights = N.array(senses)
            sens_labels = svm_labels
        else:
            # XXX yoh: .mean() is effectively
            # averages across "sensitivities" of all paired classifiers (I
            # think). See more info on this topic in svm.py on how sv_coefs
            # are stored
            #
            # First multiply SV coefficients with the actuall SVs to get
            # weighted impact of SVs on decision, then for each feature
            # take mean across SVs to get a single weight value
            # per feature
            if nr_class <= 2:
                # as simple as this
                weights = (svcoef * svs).A
                sens_labels = [tuple(svm_labels)]
            else:
                # we need to compose correctly per each pair of classifiers.
                # See docstring for getSVCoef for more details on internal
                # structure of bloody storage

                # total # of pairs
                npairs = nr_class * (nr_class-1)/2
                # # of SVs in each class
                NSVs_perclass = model.getNSV()
                # indices where each class starts in each row of SVs
                # name is after similar variable in libsvm internals
                nz_start = N.cumsum([0] + NSVs_perclass[:-1])
                nz_end = nz_start + NSVs_perclass
                # reserve storage
                weights = N.zeros((npairs, svs.shape[1]))
                ipair = 0               # index of the pair
                """
                // classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]
                """
                sens_labels = []
                for i in xrange(nr_class):
                    for j in xrange(i+1, nr_class):
                        weights[ipair, :] = N.asarray(
                            svcoef[j-1, nz_start[i]:nz_end[i]]
                            * svs[nz_start[i]:nz_end[i]])
                        sens_labels += [(svm_labels[i], svm_labels[j])]
                        ipair += 1      # go to the next pair
                assert(ipair == npairs)

        if __debug__:
            debug('SVM',
                  "Extracting weights for %d-class SVM: #SVs=%s, " % \
                  (nr_class, str(model.getNSV())) + \
                  " SVcoefshape=%s SVs.shape=%s Rhos=%s." % \
                  (svcoef.shape, svs.shape, rhos) + \
                  " Result: min=%f max=%f" % (N.min(weights), N.max(weights)))

        # and we should have prepared the labels
        assert(sens_labels is not None)

        # NOTE: `weights` is already and always 2D
        weights_ds = Dataset(weights, sa={'labels': sens_labels})

        return weights_ds

    _customizeDocInherit = True
