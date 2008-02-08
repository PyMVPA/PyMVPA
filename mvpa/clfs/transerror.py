#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utility class to compute the transfer error of classifiers."""

__docformat__ = 'restructuredtext'

import copy

import numpy as N

from sets import Set
from StringIO import StringIO
from math import log10, ceil

from mvpa.misc.errorfx import MeanMismatchErrorFx
from mvpa.misc import warning
from mvpa.misc.state import StateVariable, Stateful

if __debug__:
    from mvpa.misc import debug


class ConfusionMatrix(object):
    """Simple class for confusion matrix computation / display.

    Implementation is aimed to be simple, thus it delays actual
    computation of confusion matrix untill all data is acquired (to
    figure out complete set of labels. If testing data doesn't have a
    complete set of labels, but you like to include all labels,
    provide them as a parameter to constructor.
    """
    # XXX Michael: - How do multiple sets work and what are they there for?
    #              - This class does not work with regular Python sequences
    #                when passed to the constructor as targets and predictions.
    def __init__(self, labels=[], targets=None, predictions=None):
        """Initialize ConfusionMatrix with optional list of `labels`

        :Parameters:
         labels : list
           Optional set of labels to include in the matrix
         targets
           Optional set of targets
         predictions
           Optional set of predictions
        """

        self.__labels = labels
        """List of known labels"""
        self.__computed = False
        """Flag either it was computed for a given set of data"""
        self.__sets = []
        """Datasets (target, prediction) to compute confusion matrix on"""
        self.__matrix = None
        """Resultant confusion matrix"""

        if not targets is None or not predictions is None:
            if not targets is None and not predictions is None:
                self.add(targets=targets, predictions=predictions)
            else:
                raise ValueError, \
                      "Please provide none or both targets and predictions"


    def add(self, targets, predictions):
        """Add new results to the set of known results"""
        if len(targets)!=len(predictions):
            raise ValueError, \
                  "Targets[%d] and predictions[%d]" % (len(targets),
                                                       len(predictions)) + \
                  " have different number of samples"

        # enforce labels in predictions to be of the same datatype as in
        # targets, since otherwise we are getting doubles for unknown at a
        # given moment labels
        for i in xrange(len(targets)):
            t1, t2 = type(targets[i]), type(predictions[i])
            if t1 != t2:
                #warning("Obtained target %s and prediction %s are of " %
                #       (t1, t2) + "different datatypes.")
                predictions[i] = t1(predictions[i])

        self.__sets.append( (targets, predictions) )
        self.__computed = False


    def _compute(self):
        """Actually compute the confusion matrix based on all the sets"""
        if self.__computed:
            return

        if __debug__:
            if not self.__matrix is None:
                debug("LAZY", "Have to recompute ConfusionMatrix %s" % `self`)

        # TODO: BinaryClassifier might spit out a list of predictions for each value
        # need to handle it... for now just keep original labels
        try:
            # figure out what labels we have
            labels = list(reduce(lambda x, y: x.union(Set(y[0]).union(Set(y[1]))),
                                 self.__sets,
                                 Set(self.__labels)))
        except:
            labels = self.__labels

        labels.sort()
        self.__labels = labels          # store the recomputed labels

        Nlabels, Nsets = len(labels), len(self.__sets)

        if __debug__:
            debug("CM", "Got labels %s" % labels)

        # Create a matrix for all votes
        mat_all = N.zeros( (Nsets, Nlabels, Nlabels) )

        # create total number of samples of each label counts
        # just for convinience I guess since it can always be
        # computed from mat_all
        counts_all = N.zeros( (Nsets, Nlabels) )

        iset = 0
        for targets, predictions in self.__sets:
            # convert predictions into numpy array
            pred = N.array(predictions)

            # create the contingency table template
            mat = N.zeros( (len(labels), len(labels)), dtype = 'uint' )

            for t, tl in enumerate( labels ):
                for p, pl in enumerate( labels ):
                    mat_all[iset, t, p] = N.sum( pred[targets==tl] == pl )

            iset += 1                   # go to next set


        # for now simply compute a sum of votes across different sets
        # we might do something more sophisticated later on, and this setup
        # should easily allow it
        self.__matrix = N.sum(mat_all, axis=0)
        self.__Nsamples = N.sum(self.__matrix, axis=1)
        self.__Ncorrect = sum(N.diag(self.__matrix))
        self.__computed = True


    def __str__(self, header=True, percents=True, summary=True,
                print_empty=False):
        """'Pretty print' the matrix"""
        self._compute()

        # some shortcuts
        labels = self.__labels
        matrix = self.__matrix

        out = StringIO()
        # numbers of different entries
        Nlabels = len(labels)
        Nsamples = self.__Nsamples

        if len(self.__sets) == 0:
            return "Empty confusion matrix"

        Ndigitsmax = int(ceil(log10(max(Nsamples))))
        Nlabelsmax = max( [len(str(x)) for x in labels] )

        L = max(Ndigitsmax, Nlabelsmax)     # length of a single label/value
        res = ""

        prefixlen = Nlabelsmax+2+Ndigitsmax+1
        pref = ' '*(prefixlen) # empty prefix
        if header:
            # print out the header
            out.write(pref)
            for label in labels:
                label = str(label)      # make it a string
                # center damn label
                Nspaces = int(ceil((L-len(label))/2.0))
                out.write(" %%%ds%%s%%%ds"
                          % (Nspaces, L-Nspaces-len(label))
                          % ('', label, ''))
            out.write("\n")

            # underscores
            out.write("%s%s\n" % (pref, (" %s" % ("-" * L)) * Nlabels))

        if matrix.shape != (Nlabels, Nlabels):
            raise ValueError, \
                  "Number of labels %d doesn't correspond the size" + \
                  " of a confusion matrix %s" % (Nlabels, matrix.shape)

        for i in xrange(Nlabels):
            # print the label
            if Nsamples[i] == 0:
                continue
            out.write("%%%ds {%%%dd}" \
                % (Nlabelsmax, Ndigitsmax) % (labels[i], Nsamples[i])),
            for j in xrange(Nlabels):
                out.write(" %%%dd" % L % matrix[i, j])
            if percents:
                out.write(' [%6.2f%%]' % (matrix[i, i] * 100.0 / Nsamples[i]))
            out.write("\n")

        if summary:
            out.write("%%-%ds%%s\n"
                      % prefixlen
                      % ("", "-"*((L+1)*Nlabels)))

            out.write("%%-%ds[%%6.2f%%%%]\n"
                      % (prefixlen + (L+1)*Nlabels)
                      % ("Total Correct {%d out of %d}" \
                        % (self.__Ncorrect, sum(Nsamples)),
                         self.percentCorrect ))


        result = out.getvalue()
        out.close()
        return result


    def __iadd__(self, other):
        """Add the sets from `other` s `ConfusionMatrix` to current one
        """
        #print "adding ", other, " to ", self
        # need to do shallow copy, or otherwise smth like "cm += cm"
        # would loop forever and exhaust memory eventually
        othersets = copy.copy(other.__sets)
        for set in othersets:
            self.add(set[0], set[1])
        return self


    def __add__(self, other):
        """Add two `ConfusionMatrix`
        """
        result = copy.copy(self)
        result += other
        return result


    @property
    def matrices(self):
        """Return a list of separate confusion matrix per each stored set"""
        return [ self.__class__(labels=self.labels,
                                targets=x[0],
                                predictions=x[1]) for x in self.__sets]


    @property
    def labels(self):
        self._compute()
        return self.__labels


    @property
    def matrix(self):
        self._compute()
        return self.__matrix

    @property
    def percentCorrect(self):
        self._compute()
        return 100.0*self.__Ncorrect/sum(self.__Nsamples)

    @property
    def error(self):
        self._compute()
        return 1.0-self.__Ncorrect*1.0/sum(self.__Nsamples)

    sets = property(lambda self:self.__sets)



class ClassifierError(Stateful):
    """Compute the some error of a (trained) classifier on a dataset.
    """

    confusion = StateVariable(enabled=False)
    """TODO Think that labels might be also symbolic thus can't directly
       be indicies of the array
    """

    def __init__(self, clf, labels=None, train=True, **kwargs):
        """Initialization.

        :Parameters:
          clf : Classifier
            Either trained or untrained classifier
          labels : list
            if provided, should be a set of labels to add on top of the
            ones present in testdata
          train : bool
            unless train=False, classifier gets trained if
            trainingdata provided to __call__
        """
        Stateful.__init__(self, **kwargs)
        self.__clf = clf

        self.__labels = labels
        """Labels to add on top to existing in testing data"""

        self.__train = train
        """Either to train classifier if trainingdata is provided"""

    def __copy__(self):
        """TODO: think... may be we need to copy self.clf"""
        out = ClassifierError.__new__(TransferError)
        ClassifierError.__init__(out, self.clf)
        out._copy_states_(self)
        return out


    def _precall(self, testdataset, trainingdataset=None):
        """Generic part which trains the classifier if necessary
        """
        if not trainingdataset is None:
            if self.__train:
                if self.__clf.isTrained(trainingdataset):
                    warning('It seems that classifier %s was already trained' %
                            self.__clf + ' on dataset %s. Please inspect' \
                                % trainingdataset)
                self.__clf.train(trainingdataset)

        if self.__clf.states.isEnabled('trained_labels') and \
               not testdataset is None:
            newlabels = Set(testdataset.uniquelabels) - self.__clf.trained_labels
            if len(newlabels)>0:
                warning("Classifier %s wasn't trained to classify labels %s" %
                        (`self.__clf`, `newlabels`) +
                        " present in testing dataset. Make sure that you has" %
                        " not mixed order/names of the arguments anywhere")

        ### Here checking for if it was trained... might be a cause of trouble
        # XXX disabled since it is unreliable.. just rely on explicit
        # self.__train
        #    if  not self.__clf.isTrained(trainingdataset):
        #        self.__clf.train(trainingdataset)
        #    elif __debug__:
        #        debug('CERR',
        #              'Not training classifier %s since it was ' % `self.__clf`
        #              + ' already trained on dataset %s' % `trainingdataset`)


    def _call(self, testdataset, trainingdataset=None):
        raise NotImplementedError


    def _postcall(self, testdataset, trainingdataset=None, error=None):
        pass


    def __call__(self, testdataset, trainingdataset=None):
        """Compute the transfer error for a certain test dataset.

        If `trainingdataset` is not `None` the classifier is trained using the
        provided dataset before computing the transfer error. Otherwise the
        classifier is used in it's current state to make the predictions on
        the test dataset.

        Returns a scalar value of the transfer error.
        """
        self._precall(testdataset, trainingdataset)
        error = self._call(testdataset, trainingdataset)
        self._postcall(testdataset, trainingdataset, error)
        return error

    @property
    def clf(self):
        return self.__clf

    @property
    def labels(self):
        return self.__labels



class TransferError(ClassifierError):
    """Compute the transfer error of a (trained) classifier on a dataset.

    The actual error value is computed using a customizable error function.
    Optionally the classifier can be trained by passing an additional
    training dataset to the __call__() method.
    """
    def __init__(self, clf, errorfx=MeanMismatchErrorFx(), labels=None,
                 **kwargs):
        """Initialization.

        :Parameters:
          clf : Classifier
            Either trained or untrained classifier
          errorfx
            Functor that computes a scalar error value from the vectors of
            desired and predicted values (e.g. subclass of `ErrorFunction`)
          labels : list
            if provided, should be a set of labels to add on top of the
            ones present in testdata
        """
        ClassifierError.__init__(self, clf, labels, **kwargs)
        self.__errorfx = errorfx


    def __copy__(self):
        """TODO: think... may be we need to copy self.clf"""
        # TODO TODO -- use ClassifierError.__copy__
        out = TransferError.__new__(TransferError)
        TransferError.__init__(out, self.clf, self.errorfx, self.__labels)
        out._copy_states_(self)
        return out


    def _call(self, testdataset, trainingdataset=None):
        """Compute the transfer error for a certain test dataset.

        If `trainingdataset` is not `None` the classifier is trained using the
        provided dataset before computing the transfer error. Otherwise the
        classifier is used in it's current state to make the predictions on
        the test dataset.

        Returns a scalar value of the transfer error.
        """

        predictions = self.clf.predict(testdataset.samples)

        # compute confusion matrix
        # TODO should migrate into ClassifierError.__postcall?
        if self.states.isEnabled('confusion'):
            self.confusion = ConfusionMatrix(
                labels=self.labels, targets=testdataset.labels,
                predictions=predictions)

        # TODO

        # compute error from desired and predicted values
        error = self.__errorfx(predictions,
                               testdataset.labels)

        return error

    @property
    def errorfx(self): return self.__errorfx



class ConfusionBasedError(ClassifierError):
    """For a given classifier report an error based on internally
    computed error measure (given by some `ConfusionMatrix` stored in
    some state variable of `Classifier`).

    This way we can perform feature selection taking as the error
    criterion either learning error, or transfer to splits error in
    the case of SplitClassifier

    TODO: Derive it from some common class with `TransferError`
    """

    def __init__(self, clf, labels=None, confusion_state="training_confusion",
                 **kwargs):
        """Initialization.

        :Parameters:
          clf : Classifier
            Either trained or untrained classifier
          confusion_state
            Id of the state variable which stores `ConfusionMatrix`
          labels : list
            if provided, should be a set of labels to add on top of the
            ones present in testdata
        """
        ClassifierError.__init__(self, clf, labels, **kwargs)

        self.__confusion_state = confusion_state
        """What state to extract from"""

        if not clf.states.isKnown(confusion_state):
            raise ValueError, \
                  "State variable %s is not defined for classifier %s" % \
                  (confusion_state, `clf`)
        if not clf.states.isEnabled(confusion_state):
            if __debug__:
                debug('CERR', "Forcing state %s to be enabled for %s" %
                      (confusion_state, `clf`))
            clf.states.enable(confusion_state)


    def _call(self, testdata, trainingdata=None):
        """Extract transfer error. Nor testdata, neither trainingdata is used

        TODO: may be we should train here the same way as TransferError does?
        """
        confusion = self.clf.states.get(self.__confusion_state)
        self.confusion = confusion
        return confusion.error
