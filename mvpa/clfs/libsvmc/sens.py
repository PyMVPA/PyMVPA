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

if __debug__:
    from mvpa.base import debug

class LinearSVMWeights(Sensitivity):
    """`SensitivityAnalyzer` for the LIBSVM implementation of a linear SVM.
    """

    _ATTRIBUTE_COLLECTIONS = ['params']

    biases = StateVariable(enabled=True,
                           doc="Offsets of separating hyperplanes")

    split_weights = Parameter(False, allowedtype='bool',
                  doc="If binary classification either to sum SVs per each "
                      "class separately")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf: LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        # init base classes first
        Sensitivity.__init__(self, clf, **kwargs)


    def _call(self, dataset, callables=[]):
        # local bindings
        model = self.clf.model
        nr_class = model.nr_class

        if nr_class != 2:
            warning("You are estimating sensitivity for SVM %s trained on %d" %
                    (str(self.clf), self.clf.model.nr_class) +
                    " classes. Make sure that it is what you intended to do" )

        svcoef = N.matrix(model.getSVCoef())
        svs = N.matrix(model.getSV())
        rhos = N.asarray(model.getRho())

        self.biases = rhos
        if self.split_weights:
            if nr_class != 2:
                raise NotImplementedError, \
                      "Cannot compute per-class weights for" \
                      " non-binary classification task"
            # libsvm might have different idea on the ordering
            # of labels, so we would need to map them back explicitely
            svm_labels = model.getLabels() # labels as assigned by libsvm
            ds_labels = list(dataset.uniquelabels) # labels in the dataset
            senses = [None for i in ds_labels]
            # first label is given positive value
            for i, (c, l) in enumerate( [(svcoef > 0, lambda x: x),
                                         (svcoef < 0, lambda x: x*-1)] ):
                # convert to array, and just take the meaningful dimension
                c_ = c.A[0]
                senses[ds_labels.index(svm_labels[i])] = \
                                (l(svcoef[:, c_] * svs[c_, :])).A[0]
            weights = N.array(senses)
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
            weights = svcoef * svs

        if __debug__:
            debug('SVM',
                  "Extracting weights for %d-class SVM: #SVs=%s, " % \
                  (nr_class, str(model.getNSV())) + \
                  " SVcoefshape=%s SVs.shape=%s Rhos=%s." % \
                  (svcoef.shape, svs.shape, rhos) + \
                  " Result: min=%f max=%f" % (N.min(weights), N.max(weights)))

        return N.asarray(weights.T)

    _customizeDocInherit = True
