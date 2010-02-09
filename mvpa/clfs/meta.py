# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classes for meta classifiers -- classifiers which use other classifiers

Meta Classifiers can be grouped according to their function as

:group BoostedClassifiers: CombinedClassifier MulticlassClassifier
  SplitClassifier
:group ProxyClassifiers: ProxyClassifier BinaryClassifier MappedClassifier
  FeatureSelectionClassifier
:group PredictionsCombiners for CombinedClassifier: PredictionsCombiner
  MaximalVote MeanPrediction

"""

__docformat__ = 'restructuredtext'

import operator
import numpy as N

from sets import Set

from mvpa.misc.args import group_kwargs
from mvpa.misc.param import Parameter

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.datasets.miscfx import get_samples_by_attr
from mvpa.misc.attrmap import AttributeMap
from mvpa.misc.state import StateVariable, ClassWithCollections, Harvestable
from mvpa.mappers.base import FeatureSliceMapper

from mvpa.clfs.base import Classifier
from mvpa.clfs.distance import cartesian_distance
from mvpa.misc.transformers import first_axis_mean

from mvpa.measures.base import \
    BoostedClassifierSensitivityAnalyzer, ProxyClassifierSensitivityAnalyzer, \
    MappedClassifierSensitivityAnalyzer, \
    FeatureSelectionClassifierSensitivityAnalyzer, \
    RegressionAsClassifierSensitivityAnalyzer

from mvpa.base import warning

if __debug__:
    from mvpa.base import debug


class BoostedClassifier(Classifier, Harvestable):
    """Classifier containing the farm of other classifiers.

    Should rarely be used directly. Use one of its childs instead
    """

    # should not be needed if we have prediction_estimates upstairs
    # raw_predictions should be handled as Harvestable???
    raw_predictions = StateVariable(enabled=False,
        doc="Predictions obtained from each classifier")

    raw_estimates = StateVariable(enabled=False,
        doc="Estimates obtained from each classifier")


    def __init__(self, clfs=None, propagate_ca=True,
                 harvest_attribs=None, copy_attribs='copy',
                 **kwargs):
        """Initialize the instance.

        Parameters
        ----------
        clfs : list
          list of classifier instances to use (slave classifiers)
        propagate_ca : bool
          either to propagate enabled ca into slave classifiers.
          It is in effect only when slaves get assigned - so if state
          is enabled not during construction, it would not necessarily
          propagate into slaves
        kwargs : dict
          dict of keyworded arguments which might get used
          by State or Classifier
        """
        if clfs == None:
            clfs = []

        Classifier.__init__(self, **kwargs)
        Harvestable.__init__(self, harvest_attribs, copy_attribs)

        self.__clfs = None
        """Pylint friendly definition of __clfs"""

        self.__propagate_ca = propagate_ca
        """Enable current enabled ca in slave classifiers"""

        self._set_classifiers(clfs)
        """Store the list of classifiers"""


    def __repr__(self, prefixes=[]):
        if self.__clfs is None or len(self.__clfs)==0:
            #prefix_ = "clfs=%s" % repr(self.__clfs)
            prefix_ = []
        else:
            prefix_ = ["clfs=[%s,...]" % repr(self.__clfs[0])]
        return super(BoostedClassifier, self).__repr__(prefix_ + prefixes)


    def _train(self, dataset):
        """Train `BoostedClassifier`
        """
        for clf in self.__clfs:
            clf.train(dataset)


    def _posttrain(self, dataset):
        """Custom posttrain of `BoostedClassifier`

        Harvest over the trained classifiers if it was asked to so
        """
        Classifier._posttrain(self, dataset)
        if self.ca.is_enabled('harvested'):
            for clf in self.__clfs:
                self._harvest(locals())
        if self.params.retrainable:
            self.__changedData_isset = False


    ##REF: Name was automagically refactored
    def _get_feature_ids(self):
        """Custom _get_feature_ids for `BoostedClassifier`
        """
        # return union of all used features by slave classifiers
        feature_ids = Set([])
        for clf in self.__clfs:
            feature_ids = feature_ids.union(Set(clf.ca.feature_ids))
        return list(feature_ids)


    def _predict(self, dataset):
        """Predict using `BoostedClassifier`
        """
        raw_predictions = [ clf.predict(dataset) for clf in self.__clfs ]
        self.ca.raw_predictions = raw_predictions
        assert(len(self.__clfs)>0)
        if self.ca.is_enabled("estimates"):
            if N.array([x.ca.is_enabled("estimates")
                        for x in self.__clfs]).all():
                estimates = [ clf.ca.estimates for clf in self.__clfs ]
                self.ca.raw_estimates = estimates
            else:
                warning("One or more classifiers in %s has no 'estimates' state" %
                        self + "enabled, thus BoostedClassifier can't have" +
                        " 'raw_estimates' state variable defined")

        return raw_predictions


    ##REF: Name was automagically refactored
    def _set_classifiers(self, clfs):
        """Set the classifiers used by the boosted classifier

        We have to allow to set list of classifiers after the object
        was actually created. It will be used by
        MulticlassClassifier
        """
        self.__clfs = clfs
        """Classifiers to use"""

        if len(clfs):
            # enable corresponding ca in the slave-classifiers
            if self.__propagate_ca:
                for clf in self.__clfs:
                    clf.ca.enable(self.ca.enabled, missingok=True)

        # adhere to their capabilities + 'multiclass'
        # XXX do intersection across all classifiers!
        # TODO: this seems to be wrong since it can be regression etc
        self.__tags__ = [ 'binary', 'multiclass', 'meta' ]
        if len(clfs)>0:
            self.__tags__ += self.__clfs[0].__tags__

    def untrain(self):
        """Untrain `BoostedClassifier`

        Has to untrain any known classifier
        """
        if not self.trained:
            return
        for clf in self.clfs:
            clf.untrain()
        super(BoostedClassifier, self).untrain()

    ##REF: Name was automagically refactored
    def get_sensitivity_analyzer(self, **kwargs):
        """Return an appropriate SensitivityAnalyzer"""
        return BoostedClassifierSensitivityAnalyzer(
                self,
                **kwargs)


    clfs = property(fget=lambda x:x.__clfs,
                    fset=_set_classifiers,
                    doc="Used classifiers")



class ProxyClassifier(Classifier):
    """Classifier which decorates another classifier

    Possible uses:

     - modify data somehow prior training/testing:
       * normalization
       * feature selection
       * modification

     - optimized classifier?

    """

    __sa_class__ = ProxyClassifierSensitivityAnalyzer
    """Sensitivity analyzer to use for a generic ProxyClassifier"""

    def __init__(self, clf, **kwargs):
        """Initialize the instance of ProxyClassifier

        Parameters
        ----------
        clf : Classifier
          Classifier to proxy, i.e. to use after decoration
        """

        # Is done before parents __init__ since we need
        # it for _set_retrainable called during __init__
        self.__clf = clf
        """Store the classifier to use."""

        Classifier.__init__(self, **kwargs)

        # adhere to slave classifier capabilities
        # TODO: unittest
        self.__tags__ = self.__tags__[:] + ['meta']
        if clf is not None:
            self.__tags__ += clf.__tags__


    def __repr__(self, prefixes=[]):
        return super(ProxyClassifier, self).__repr__(
            ["clf=%s" % repr(self.__clf)] + prefixes)

    def summary(self):
        s = super(ProxyClassifier, self).summary()
        if self.trained:
            s += "\n Slave classifier summary:" + \
                 '\n + %s' % \
                 (self.__clf.summary().replace('\n', '\n |'))
        return s

    ##REF: Name was automagically refactored
    def _set_retrainable(self, value, force=False):
        # XXX Lazy implementation
        self.clf._set_retrainable(value, force=force)
        super(ProxyClassifier, self)._set_retrainable(value, force)
        if value and not (self.ca['retrained']
                          is self.clf.ca['retrained']):
            if __debug__:
                debug("CLFPRX",
                      "Rebinding state variables from slave clf %s" % self.clf)
            self.ca['retrained'] = self.clf.ca['retrained']
            self.ca['repredicted'] = self.clf.ca['repredicted']


    def _train(self, dataset):
        """Train `ProxyClassifier`
        """
        # base class does nothing much -- just proxies requests to underlying
        # classifier
        self.__clf.train(dataset)

        # for the ease of access
        # TODO: if to copy we should exclude some ca which are defined in
        #       base Classifier (such as training_time, predicting_time)
        # YOH: for now _copy_ca_ would copy only set ca variables. If
        #      anything needs to be overriden in the parent's class, it is
        #      welcome to do so
        #self.ca._copy_ca_(self.__clf, deep=False)


    def _predict(self, dataset):
        """Predict using `ProxyClassifier`
        """
        clf = self.__clf
        if self.ca.is_enabled('estimates'):
            clf.ca.enable(['estimates'])

        result = clf.predict(dataset)
        # for the ease of access
        self.ca._copy_ca_(self.__clf, ['estimates'], deep=False)
        return result


    def untrain(self):
        """Untrain ProxyClassifier
        """
        if not self.__clf is None:
            self.__clf.untrain()
        super(ProxyClassifier, self).untrain()


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    ##REF: Name was automagically refactored
    def get_sensitivity_analyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer"""
        return self.__sa_class__(
                self,
                analyzer=self.__clf.get_sensitivity_analyzer(**slave_kwargs),
                **kwargs)


    clf = property(lambda x:x.__clf, doc="Used `Classifier`")



#
# Various combiners for CombinedClassifier
#

class PredictionsCombiner(ClassWithCollections):
    """Base class for combining decisions of multiple classifiers"""

    def train(self, clfs, dataset):
        """PredictionsCombiner might need to be trained

        Parameters
        ----------
        clfs : list of Classifier
          List of classifiers to combine. Has to be classifiers (not
          pure predictions), since combiner might use some other
          state variables (value's) instead of pure prediction's
        dataset : Dataset
          training data in this case
        """
        pass


    def __call__(self, clfs, dataset):
        """Call function

        Parameters
        ----------
        clfs : list of Classifier
          List of classifiers to combine. Has to be classifiers (not
          pure predictions), since combiner might use some other
          state variables (value's) instead of pure prediction's
        """
        raise NotImplementedError



class MaximalVote(PredictionsCombiner):
    """Provides a decision using maximal vote rule"""

    predictions = StateVariable(enabled=True,
        doc="Voted predictions")
    estimates = StateVariable(enabled=False,
        doc="Estimates keep counts across classifiers for each label/sample")

    def __init__(self):
        """XXX Might get a parameter to use raw decision estimates if
        voting is not unambigous (ie two classes have equal number of
        votes
        """
        PredictionsCombiner.__init__(self)


    def __call__(self, clfs, dataset):
        """Actuall callable - perform voting

        Extended functionality which might not be needed actually:
        Since `BinaryClassifier` might return a list of possible
        predictions (not just a single one), we should consider all of those

        MaximalVote doesn't care about dataset itself
        """
        if len(clfs)==0:
            return []                   # to don't even bother

        all_label_counts = None
        for clf in clfs:
            # Lets check first if necessary state variable is enabled
            if not clf.ca.is_enabled("predictions"):
                raise ValueError, "MaximalVote needs classifiers (such as " + \
                      "%s) with state 'predictions' enabled" % clf
            predictions = clf.ca.predictions
            if all_label_counts is None:
                all_label_counts = [ {} for i in xrange(len(predictions)) ]

            # for every sample
            for i in xrange(len(predictions)):
                prediction = predictions[i]
                # XXX fishy location due to literal labels,
                # TODO simplify assumptions and logic
                if isinstance(prediction, basestring) or \
                       not operator.isSequenceType(prediction):
                    prediction = (prediction,)
                for label in prediction: # for every label
                    # XXX we might have multiple labels assigned
                    # but might not -- don't remember now
                    if not all_label_counts[i].has_key(label):
                        all_label_counts[i][label] = 0
                    all_label_counts[i][label] += 1

        predictions = []
        # select maximal vote now for each sample
        for i in xrange(len(all_label_counts)):
            label_counts = all_label_counts[i]
            # lets do explicit search for max so we know
            # if it is unique
            maxk = []                   # labels of elements with max vote
            maxv = -1
            for k, v in label_counts.iteritems():
                if v > maxv:
                    maxk = [k]
                    maxv = v
                elif v == maxv:
                    maxk.append(k)

            assert len(maxk) >= 1, \
                   "We should have obtained at least a single key of max label"

            if len(maxk) > 1:
                warning("We got multiple labels %s which have the " % maxk +
                        "same maximal vote %d. XXX disambiguate" % maxv)
            predictions.append(maxk[0])

        ca = self.ca
        ca.estimates = all_label_counts
        ca.predictions = predictions
        return predictions



class MeanPrediction(PredictionsCombiner):
    """Provides a decision by taking mean of the results
    """

    predictions = StateVariable(enabled=True,
        doc="Mean predictions")

    estimates = StateVariable(enabled=True,
        doc="Predictions from all classifiers are stored")

    def __call__(self, clfs, dataset):
        """Actuall callable - perform meaning

        """
        if len(clfs)==0:
            return []                   # to don't even bother

        all_predictions = []
        for clf in clfs:
            # Lets check first if necessary state variable is enabled
            if not clf.ca.is_enabled("predictions"):
                raise ValueError, "MeanPrediction needs learners (such " \
                      " as %s) with state 'predictions' enabled" % clf
            all_predictions.append(clf.ca.predictions)

        # compute mean
        all_predictions = N.asarray(all_predictions)
        predictions = N.mean(all_predictions, axis=0)

        ca = self.ca
        ca.estimates = all_predictions
        ca.predictions = predictions
        return predictions


class ClassifierCombiner(PredictionsCombiner):
    """Provides a decision using training a classifier on predictions/estimates

    TODO: implement
    """

    predictions = StateVariable(enabled=True,
        doc="Trained predictions")


    def __init__(self, clf, variables=None):
        """Initialize `ClassifierCombiner`

        Parameters
        ----------
        clf : Classifier
          Classifier to train on the predictions
        variables : list of str
          List of state variables stored in 'combined' classifiers, which
          to use as features for training this classifier
        """
        PredictionsCombiner.__init__(self)

        self.__clf = clf
        """Classifier to train on `variables` ca of provided classifiers"""

        if variables == None:
            variables = ['predictions']
        self.__variables = variables
        """What state variables of the classifiers to use"""


    def untrain(self):
        """It might be needed to untrain used classifier"""
        if self.__clf:
            self.__clf.untrain()

    def __call__(self, clfs, dataset):
        """
        """
        if len(clfs)==0:
            return []                   # to don't even bother

        raise NotImplementedError



class CombinedClassifier(BoostedClassifier):
    """`BoostedClassifier` which combines predictions using some
    `PredictionsCombiner` functor.
    """

    def __init__(self, clfs=None, combiner=None, **kwargs):
        """Initialize the instance.

        Parameters
        ----------
        clfs : list of Classifier
          list of classifier instances to use
        combiner : PredictionsCombiner
          callable which takes care about combining multiple
          results into a single one (e.g. maximal vote for
          classification, MeanPrediction for regression))
        kwargs : dict
          dict of keyworded arguments which might get used
          by State or Classifier

        NB: `combiner` might need to operate not on 'predictions' descrete
            labels but rather on raw 'class' estimates classifiers
            estimate (which is pretty much what is stored under
            `estimates`
        """
        if clfs == None:
            clfs = []

        BoostedClassifier.__init__(self, clfs, **kwargs)

        self.__combiner = combiner
        """Functor destined to combine results of multiple classifiers"""


    def __repr__(self, prefixes=[]):
        """Literal representation of `CombinedClassifier`.
        """
        return super(CombinedClassifier, self).__repr__(
            ["combiner=%s" % repr(self.__combiner)] + prefixes)

    @property
    def combiner(self):
        # Decide either we are dealing with regressions
        # by looking at 1st learner
        if self.__combiner is None:
            self.__combiner = (
                MaximalVote,
                MeanPrediction)[int(self.clfs[0].__is_regression__)]()
        return self.__combiner


    def summary(self):
        """Provide summary for the `CombinedClassifier`.
        """
        s = super(CombinedClassifier, self).summary()
        if self.trained:
            s += "\n Slave classifiers summaries:"
            for i, clf in enumerate(self.clfs):
                s += '\n + %d clf: %s' % \
                     (i, clf.summary().replace('\n', '\n |'))
        return s


    def untrain(self):
        """Untrain `CombinedClassifier`
        """
        try:
            self.__combiner.untrain()
        except:
            pass
        super(CombinedClassifier, self).untrain()


    def _train(self, dataset):
        """Train `CombinedClassifier`
        """
        BoostedClassifier._train(self, dataset)

        # combiner might need to train as well
        self.combiner.train(self.clfs, dataset)


    def _predict(self, dataset):
        """Predict using `CombinedClassifier`
        """
        ca = self.ca
        cca = self.combiner.ca
        BoostedClassifier._predict(self, dataset)
        if ca.is_enabled("estimates"):
            cca.enable('estimates')
        # combiner will make use of state variables instead of only predictions
        # returned from _predict
        predictions = self.combiner(self.clfs, dataset)
        ca.predictions = predictions

        if ca.is_enabled("estimates"):
            if cca.is_active("estimates"):
                # XXX or may be we could leave simply up to accessing .combiner?
                ca.estimates = cca.estimates
            else:
                if __debug__:
                    warning("Boosted classifier %s has 'estimates' state enabled,"
                            " but combiner doesn't have 'estimates' active, thus "
                            " .estimates cannot be provided directly, access .clfs"
                            % self)
        return predictions



class TreeClassifier(ProxyClassifier):
    """`TreeClassifier` which allows to create hierarchy of classifiers

    Functions by grouping some labels into a single "meta-label" and training
    classifier first to separate between meta-labels.  Then
    each group further proceeds with classification within each group.

    Possible scenarios::

      TreeClassifier(SVM(),
       {'animate':  ((1,2,3,4),
                     TreeClassifier(SVM(),
                         {'human': (('male', 'female'), SVM()),
                          'animals': (('monkey', 'dog'), SMLR())})),
        'inanimate': ((5,6,7,8), SMLR())})

    would create classifier which would first do binary classification
    to separate animate from inanimate, then for animate result it
    would separate to classify human vs animal and so on::

                                   SVM
                                 /      \
                            animate   inanimate
                             /             \
                           SVM             SMLR
                         /     \          / | \ \
                    human    animal      5  6 7  8
                     |          |
                    SVM        SVM
                   /   \       /  \
                 male female monkey dog
                  1      2    3      4

    """

    _DEV__doc = """
    Questions:
     * how to collect confusion matrices at a particular layer if such
       classifier is given to SplitClassifier or CVTE

     * What additional ca to add, something like
        clf_labels  -- store remapped labels for the dataset
        clf_estimates  ...

     * What do we store into estimates ? just estimates from the clfs[]
       for corresponding samples, or top level clf estimates as well?

     * what should be SensitivityAnalyzer?  by default it would just
       use top slave classifier (i.e. animate/inanimate)

    Problems?
     *  .clf is not actually "proxied" per se, so not sure what things
        should be taken care of yet...

    TODO:
     * Allow a group to be just a single category, so no further
        classifier is needed, it just should stay separate from the
        other groups

    Possible TODO:
     *  Add ability to provide results of clf.estimates as features into
        input of clfs[]. This way we could provide additional 'similarity'
        information to the "other" branch

    """

    def __init__(self, clf, groups, **kwargs):
        """Initialize TreeClassifier

        Parameters
        ----------
        clf : Classifier
          Classifier to separate between the groups
        groups : dict of meta-label: tuple of (tuple of labels, classifier)
          Defines the groups of labels and their classifiers.
          See :class:`~mvpa.clfs.meta.TreeClassifier` for example
        """

        # Basic initialization
        ProxyClassifier.__init__(self, clf, **kwargs)

        # XXX RF: probably create internal structure with dictionary,
        # not just a tuple, and store all information in there
        # accordingly

        self._groups = groups
        self._index2group = groups.keys()

        # All processing of groups needs to be handled within _train
        # since labels_map is not available here and definition
        # is allowed to carry both symbolic and numeric values for
        # labels
        # XXX TODO due to abandoning of labels_map -- may be this is
        #     no longer the case?

        # We can only assign respective classifiers
        self.clfs = dict([(gk, c) for gk, (ls, c) in groups.iteritems()])
        """Dictionary of classifiers used by the groups"""


    def __repr__(self, prefixes=[]):
        """String representation of TreeClassifier
        """
        prefix = "groups=%s" % repr(self._groups)
        return super(TreeClassifier, self).__repr__([prefix] + prefixes)


    def summary(self):
        """Provide summary for the `TreeClassifier`.
        """
        s = super(TreeClassifier, self).summary()
        if self.trained:
            s += "\n Node classifiers summaries:"
            for i, (clfname, clf) in enumerate(self.clfs.iteritems()):
                s += '\n + %d %s clf: %s' % \
                     (i, clfname, clf.summary().replace('\n', '\n |'))
        return s


    def _train(self, dataset):
        """Train TreeClassifier

        First train .clf on groupped samples, then train each of .clfs
        on a corresponding subset of samples.
        """
        # Local bindings
        targets_sa_name = self.params.targets    # name of targets sa
        targets_sa = dataset.sa[targets_sa_name] # actual targets sa
        clf, clfs, index2group = self.clf, self.clfs, self._index2group

        # Handle groups of labels
        groups = self._groups
        groups_labels = {}              # just groups with numeric indexes
        label2index = {}                # how to map old labels to new
        known = set()
        for gi, gk in enumerate(index2group):
            ls = groups[gk][0]
            known_already = known.intersection(ls)
            if len(known_already):
                raise ValueError, "Grouping of labels is not appropriate. " \
                      "Got labels %s already among known in %s. " % \
                       (known_already, known  )
            groups_labels[gk] = ls      # needed? XXX
            for l in ls :
                label2index[l] = gi
            known = known.union(ls )
        # TODO: check if different literal labels weren't mapped into
        #       same numerical but here asked to belong to different groups
        #  yoh: actually above should catch it

        # Check if none of the labels is missing from known groups
        dsul = set(targets_sa.unique)
        if known.intersection(dsul) != dsul:
            raise ValueError, \
                  "Dataset %s had some labels not defined in groups: %s. " \
                  "Known are %s" % \
                  (dataset, dsul.difference(known), known)

        # We can operate on the same dataset here 
        # Nope: doesn't work nicely with the classifier like kNN
        #      which links to the dataset used in the training,
        #      so whenever if we simply restore labels back, we
        #      would get kNN confused in _predict()
        #      Therefore we need to create a shallow copy of
        #      dataset and provide it with new labels
        ds_group = dataset.copy(deep=False)
        # assign new labels group samples into groups of labels
        ds_group.sa[targets_sa_name].value = [label2index[l]
                                              for l in targets_sa.value]

        # train primary classifier
        if __debug__:
            debug('CLFTREE', "Training primary %(clf)s on %(ds)s",
                  msgargs=dict(clf=clf, ds=ds_group))
        clf.train(ds_group)

        # ??? should we obtain values for anything?
        #     may be we could training values of .clfs to be added
        #     as features to the next level -- i.e. .clfs

        # Proceed with next 'layer' and train all .clfs on corresponding
        # selection of samples
        # ??? should we may be allow additional 'the other' category, to
        #     signal contain all the other categories data? probably not
        #     since then it would lead to undetermined prediction (which
        #     might be not a bad thing altogether...)
        for gk in groups.iterkeys():
            # select samples per each group
            ids = get_samples_by_attr(dataset, targets_sa_name, groups_labels[gk])
            ds_group = dataset[ids]
            if __debug__:
                debug('CLFTREE', "Training %(clf)s for group %(gk)s on %(ds)s",
                      msgargs=dict(clf=clfs[gk], gk=gk, ds=ds_group))
            # and train corresponding slave clf
            clfs[gk].train(ds_group)


    def untrain(self):
        """Untrain TreeClassifier
        """
        super(TreeClassifier, self).untrain()
        for clf in self.clfs.values():
            clf.untrain()


    def _predict(self, dataset):
        """
        """
        # Local bindings
        clfs, index2group = self.clfs, self._index2group
        clf_predictions = N.asanyarray(ProxyClassifier._predict(self, dataset))
        # assure that predictions are indexes, ie int
        clf_predictions = clf_predictions.astype(int)

        # now for predictions pointing to specific groups go into
        # corresponding one
        # defer initialization since dtype would depend on predictions
        predictions = None
        for pred_group in set(clf_predictions):
            gk = index2group[pred_group]
            clf_ = clfs[gk]
            group_indexes = (clf_predictions == pred_group)
            if __debug__:
                debug('CLFTREE',
                      'Predicting for group %s using %s on %d samples' %
                      (gk, clf_, N.sum(group_indexes)))
            p = clf_.predict(dataset[group_indexes])
            if predictions is None:
                predictions = N.zeros((len(dataset),),
                                      dtype=N.asanyarray(p).dtype)
            predictions[group_indexes] = p
        return predictions


class BinaryClassifier(ProxyClassifier):
    """`ProxyClassifier` which maps set of two labels into +1 and -1
    """

    def __init__(self, clf, poslabels, neglabels, **kwargs):
        """
        Parameters
        ----------
        clf : Classifier
          classifier to use
        poslabels : list
          list of labels which are treated as +1 category
        neglabels : list
          list of labels which are treated as -1 category
        """

        ProxyClassifier.__init__(self, clf, **kwargs)

        # Handle labels
        sposlabels = Set(poslabels) # so to remove duplicates
        sneglabels = Set(neglabels) # so to remove duplicates

        # check if there is no overlap
        overlap = sposlabels.intersection(sneglabels)
        if len(overlap)>0:
            raise ValueError("Sets of positive and negative labels for " +
                "BinaryClassifier must not overlap. Got overlap " %
                overlap)

        self.__poslabels = list(sposlabels)
        self.__neglabels = list(sneglabels)

        # define what values will be returned by predict: if there is
        # a single label - return just it alone, otherwise - whole
        # list
        # Such approach might come useful if we use some classifiers
        # over different subsets of data with some voting later on
        # (1-vs-therest?)

        if len(self.__poslabels) > 1:
            self.__predictpos = self.__poslabels
        else:
            self.__predictpos = self.__poslabels[0]

        if len(self.__neglabels) > 1:
            self.__predictneg = self.__neglabels
        else:
            self.__predictneg = self.__neglabels[0]


    def __repr__(self, prefixes=[]):
        prefix = "poslabels=%s, neglabels=%s" % (
            repr(self.__poslabels), repr(self.__neglabels))
        return super(BinaryClassifier, self).__repr__([prefix] + prefixes)


    def _train(self, dataset):
        """Train `BinaryClassifier`
        """
        targets_sa_name = self.params.targets
        idlabels = [(x, +1) for x in get_samples_by_attr(dataset, targets_sa_name,
                                                         self.__poslabels)] + \
                    [(x, -1) for x in get_samples_by_attr(dataset, targets_sa_name,
                                                          self.__neglabels)]
        # XXX we have to sort ids since at the moment Dataset.select_samples
        #     doesn't take care about order
        idlabels.sort()

        # If we need all samples, why simply not perform on original
        # data, an just store/restore labels. But it really should be done
        # within Dataset.select_samples
        if len(idlabels) == dataset.nsamples \
            and [x[0] for x in idlabels] == range(dataset.nsamples):
            # the last condition is not even necessary... just overly
            # cautious
            datasetselected = dataset.copy(deep=False)   # no selection is needed
            if __debug__:
                debug('CLFBIN',
                      "Created shallow copy with %d samples for binary " %
                      (dataset.nsamples) +
                      " classification among labels %s/+1 and %s/-1" %
                      (self.__poslabels, self.__neglabels))
        else:
            datasetselected = dataset[[ x[0] for x in idlabels ]]
            if __debug__:
                debug('CLFBIN',
                      "Selected %d samples out of %d samples for binary " %
                      (len(idlabels), dataset.nsamples) +
                      " classification among labels %s/+1 and %s/-1" %
                      (self.__poslabels, self.__neglabels) +
                      ". Selected %s" % datasetselected)

        # adjust the labels
        datasetselected.sa[targets_sa_name].value = [ x[1] for x in idlabels ]

        # now we got a dataset with only 2 labels
        if __debug__:
            assert((datasetselected.sa[targets_sa_name].unique == [-1, 1]).all())

        self.clf.train(datasetselected)


    def _predict(self, dataset):
        """Predict the labels for a given `dataset`

        Predicts using binary classifier and spits out list (for each sample)
        where with either poslabels or neglabels as the "label" for the sample.
        If there was just a single label within pos or neg labels then it would
        return not a list but just that single label.
        """
        binary_predictions = ProxyClassifier._predict(self, dataset)
        self.ca.estimates = binary_predictions
        predictions = [ {-1: self.__predictneg,
                         +1: self.__predictpos}[x] for x in binary_predictions]
        self.ca.predictions = predictions
        return predictions



class MulticlassClassifier(CombinedClassifier):
    """`CombinedClassifier` to perform multiclass using a list of
    `BinaryClassifier`.

    such as 1-vs-1 (ie in pairs like libsvm doesn) or 1-vs-all (which
    is yet to think about)
    """

    def __init__(self, clf, bclf_type="1-vs-1", **kwargs):
        """Initialize the instance

        Parameters
        ----------
        clf : Classifier
          classifier based on which multiple classifiers are created
          for multiclass
        bclf_type
          "1-vs-1" or "1-vs-all", determines the way to generate binary
          classifiers
        """
        CombinedClassifier.__init__(self, **kwargs)

        self.__clf = clf
        """Store sample instance of basic classifier"""

        # Some checks on known ways to do multiclass
        if bclf_type == "1-vs-1":
            pass
        elif bclf_type == "1-vs-all": # TODO
            raise NotImplementedError
        else:
            raise ValueError, \
                  "Unknown type of classifier %s for " % bclf_type + \
                  "BoostedMulticlassClassifier"
        self.__bclf_type = bclf_type

    # XXX fix it up a bit... it seems that MulticlassClassifier should
    # be actually ProxyClassifier and use BoostedClassifier internally
    def __repr__(self, prefixes=[]):
        prefix = "bclf_type=%s, clf=%s" % (repr(self.__bclf_type),
                                            repr(self.__clf))
        return super(MulticlassClassifier, self).__repr__([prefix] + prefixes)


    def _train(self, dataset):
        """Train classifier
        """
        targets_sa_name = self.params.targets

        # construct binary classifiers
        ulabels = dataset.sa[targets_sa_name].unique
        if self.__bclf_type == "1-vs-1":
            # generate pairs and corresponding classifiers
            biclfs = []
            for i in xrange(len(ulabels)):
                for j in xrange(i+1, len(ulabels)):
                    clf = self.__clf.clone()
                    biclfs.append(
                        BinaryClassifier(
                            clf,
                            poslabels=[ulabels[i]], neglabels=[ulabels[j]]))
            if __debug__:
                debug("CLFMC", "Created %d binary classifiers for %d labels" %
                      (len(biclfs), len(ulabels)))

            self.clfs = biclfs

        elif self.__bclf_type == "1-vs-all":
            raise NotImplementedError

        # perform actual training
        CombinedClassifier._train(self, dataset)



class SplitClassifier(CombinedClassifier):
    """`BoostedClassifier` to work on splits of the data

    """

    """
    TODO: SplitClassifier and MulticlassClassifier have too much in
          common -- need to refactor: just need a splitter which would
          split dataset in pairs of class labels. MulticlassClassifier
          does just a tiny bit more which might be not necessary at
          all: map sets of labels into 2 categories...
    """

    # TODO: unify with CrossValidatedTransferError which now uses
    # harvest_attribs to expose gathered attributes
    confusion = StateVariable(enabled=False,
        doc="Resultant confusion whenever classifier trained " +
            "on 1 part and tested on 2nd part of each split")

    splits = StateVariable(enabled=False, doc=
       """Store the actual splits of the data. Can be memory expensive""")

    # ??? couldn't be training_confusion since it has other meaning
    #     here, BUT it is named so within CrossValidatedTransferError
    #     -- unify
    #  decided to go with overriding semantics tiny bit. For split
    #     classifier training_confusion would correspond to summary
    #     over training errors across all splits. Later on if need comes
    #     we might want to implement global_training_confusion which would
    #     correspond to overall confusion on full training dataset as it is
    #     done in base Classifier
    #global_training_confusion = StateVariable(enabled=False,
    #    doc="Summary over training confusions acquired at each split")

    def __init__(self, clf, splitter=NFoldSplitter(cvtype=1), **kwargs):
        """Initialize the instance

        Parameters
        ----------
        clf : Classifier
          classifier based on which multiple classifiers are created
          for multiclass
        splitter : Splitter
          `Splitter` to use to split the dataset prior training
        """

        CombinedClassifier.__init__(self, **kwargs)
        self.__clf = clf
        """Store sample instance of basic classifier"""

        if isinstance(splitter, type):
            raise ValueError, \
                  "Please provide an instance of a splitter, not a type." \
                  " Got %s" % splitter

        self.__splitter = splitter


    def _train(self, dataset):
        """Train `SplitClassifier`
        """
        targets_sa_name = self.params.targets

        # generate pairs and corresponding classifiers
        bclfs = []

        # local binding
        ca = self.ca

        clf_template = self.__clf
        if ca.is_enabled('confusion'):
            ca.confusion = clf_template.__summary_class__()
        if ca.is_enabled('training_confusion'):
            clf_template.ca.enable(['training_confusion'])
            ca.training_confusion = clf_template.__summary_class__()

        clf_hastestdataset = hasattr(clf_template, 'testdataset')

        # for proper and easier debugging - first define classifiers and then
        # train them
        for split in self.__splitter.splitcfg(dataset):
            if __debug__:
                debug("CLFSPL_",
                      "Deepcopying %(clf)s for %(sclf)s",
                      msgargs={'clf':clf_template,
                               'sclf':self})
            clf = clf_template.clone()
            bclfs.append(clf)
        self.clfs = bclfs

        self.ca.splits = []

        for i, split in enumerate(self.__splitter(dataset)):
            if __debug__:
                debug("CLFSPL", "Training classifier for split %d" % (i))

            if ca.is_enabled("splits"):
                self.ca.splits.append(split)

            clf = self.clfs[i]

            # assign testing dataset if given classifier can digest it
            if clf_hastestdataset:
                clf.testdataset = split[1]

            clf.train(split[0])

            # unbind the testdataset from the classifier
            if clf_hastestdataset:
                clf.testdataset = None

            if ca.is_enabled("confusion"):
                predictions = clf.predict(split[1])
                self.ca.confusion.add(split[1].sa[targets_sa_name].value,
                                          predictions,
                                          clf.ca.get('estimates', None))
                if __debug__:
                    dact = debug.active
                    if 'CLFSPL_' in dact:
                        debug('CLFSPL_', 'Split %d:\n%s' % (i, self.confusion))
                    elif 'CLFSPL' in dact:
                        debug('CLFSPL', 'Split %d error %.2f%%'
                              % (i, self.ca.confusion.summaries[-1].error))

            if ca.is_enabled("training_confusion"):
                # XXX this is broken, as it cannot deal with not yet set ca
                ca.training_confusion += clf.ca.training_confusion


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    ##REF: Name was automagically refactored
    def get_sensitivity_analyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer for `SplitClassifier`

        Parameters
        ----------
        combiner
          If not provided, first_axis_mean is assumed
        """
        return BoostedClassifierSensitivityAnalyzer(
                self,
                analyzer=self.__clf.get_sensitivity_analyzer(**slave_kwargs),
                **kwargs)

    splitter = property(fget=lambda x:x.__splitter,
                        doc="Splitter user by SplitClassifier")


class MappedClassifier(ProxyClassifier):
    """`ProxyClassifier` which uses some mapper prior training/testing.

    `MaskMapper` can be used just a subset of features to
    train/classify.
    Having such classifier we can easily create a set of classifiers
    for BoostedClassifier, where each classifier operates on some set
    of features, e.g. set of best spheres from SearchLight, set of
    ROIs selected elsewhere. It would be different from simply
    applying whole mask over the dataset, since here initial decision
    is made by each classifier and then later on they vote for the
    final decision across the set of classifiers.
    """

    __sa_class__ = MappedClassifierSensitivityAnalyzer

    def __init__(self, clf, mapper, **kwargs):
        """Initialize the instance

        Parameters
        ----------
        clf : Classifier
          classifier based on which mask classifiers is created
        mapper
          whatever `Mapper` comes handy
        """
        ProxyClassifier.__init__(self, clf, **kwargs)

        self.__mapper = mapper
        """mapper to help us our with prepping data to
        training/classification"""


    def _train(self, dataset):
        """Train `MappedClassifier`
        """
        # first train the mapper
        # XXX: should training be done using whole dataset or just samples
        # YYY: in some cases labels might be needed, thus better full dataset
        self.__mapper.train(dataset)

        # for train() we have to provide dataset -- not just samples to train!
        wdataset = dataset.get_mapped(self.__mapper)
        ProxyClassifier._train(self, wdataset)


    def _predict(self, dataset):
        """Predict using `MappedClassifier`
        """
        return ProxyClassifier._predict(self, self.__mapper.forward(dataset))


    mapper = property(lambda x:x.__mapper, doc="Used mapper")



class FeatureSelectionClassifier(ProxyClassifier):
    """`ProxyClassifier` which uses some `FeatureSelection` prior training.

    `FeatureSelection` is used first to select features for the classifier to
    use for prediction. Internally it would rely on MappedClassifier which
    would use created MaskMapper.

    TODO: think about removing overhead of retraining the same classifier if
    feature selection was carried out with the same classifier already. It
    has been addressed by adding .trained property to classifier, but now
    we should expclitely use is_trained here if we want... need to think more
    """

    __tags__ = [ 'does_feature_selection', 'meta' ]

    __sa_class__ = FeatureSelectionClassifierSensitivityAnalyzer

    def __init__(self, clf, feature_selection, testdataset=None, **kwargs):
        """Initialize the instance

        Parameters
        ----------
        clf : Classifier
          classifier based on which mask classifiers is created
        feature_selection : FeatureSelection
          whatever `FeatureSelection` comes handy
        testdataset : Dataset
          optional dataset which would be given on call to feature_selection
        """
        ProxyClassifier.__init__(self, clf, **kwargs)

        self.__maskclf = None
        """Should become `MappedClassifier`(mapper=`MaskMapper`) later on."""

        self.__feature_selection = feature_selection
        """`FeatureSelection` to select the features prior training"""

        self.__testdataset = testdataset
        """`FeatureSelection` might like to use testdataset"""


    def untrain(self):
        """Untrain `FeatureSelectionClassifier`

        Has to untrain any known classifier
        """
        if self.__feature_selection is not None:
            self.__feature_selection.untrain()
        if not self.trained:
            return
        if not self.__maskclf is None:
            self.__maskclf.untrain()
        super(FeatureSelectionClassifier, self).untrain()


    def _train(self, dataset):
        """Train `FeatureSelectionClassifier`
        """
        # temporarily enable selected_ids
        self.__feature_selection.ca.change_temporarily(
            enable_ca=["selected_ids"])

        if __debug__:
            debug("CLFFS", "Performing feature selection using %s" %
                  self.__feature_selection + " on %s" % dataset)

        (wdataset, tdataset) = self.__feature_selection(dataset,
                                                        self.__testdataset)
        if __debug__:
            add_ = ""
            if "CLFFS_" in debug.active:
                add_ = " Selected features: %s" % \
                       self.__feature_selection.selected_ids
            debug("CLFFS", "%(fs)s selected %(nfeat)d out of " +
                  "%(dsnfeat)d features.%(app)s",
                  msgargs={'fs':self.__feature_selection,
                           'nfeat':wdataset.nfeatures,
                           'dsnfeat':dataset.nfeatures,
                           'app':add_})

        # create a mask to devise a mapper
        # TODO -- think about making selected_ids a MaskMapper
        mappermask = N.zeros(dataset.nfeatures, dtype='bool')
        mappermask[self.__feature_selection.ca.selected_ids] = True
        mapper = FeatureSliceMapper(mappermask, dshape=mappermask.shape)

        self.__feature_selection.ca.reset_changed_temporarily()

        # create and assign `MappedClassifier`
        self.__maskclf = MappedClassifier(self.clf, mapper)
        # we could have called self.__clf.train(dataset), but it would
        # cause unnecessary masking
        self.__maskclf.clf.train(wdataset)

        # for the ease of access
        # TODO see for ProxyClassifier
        #self.ca._copy_ca_(self.__maskclf, deep=False)

    ##REF: Name was automagically refactored
    def _get_feature_ids(self):
        """Return used feature ids for `FeatureSelectionClassifier`

        """
        return self.__feature_selection.ca.selected_ids

    def _predict(self, dataset):
        """Predict using `FeatureSelectionClassifier`
        """
        clf = self.__maskclf
        if self.ca.is_enabled('estimates'):
            clf.ca.enable(['estimates'])

        result = clf._predict(dataset)
        # for the ease of access
        self.ca._copy_ca_(clf, ['estimates'], deep=False)
        return result

    ##REF: Name was automagically refactored
    def set_test_dataset(self, testdataset):
        """Set testing dataset to be used for feature selection
        """
        self.__testdataset = testdataset

    maskclf = property(lambda x:x.__maskclf, doc="Used `MappedClassifier`")
    feature_selection = property(lambda x:x.__feature_selection,
                                 doc="Used `FeatureSelection`")

    testdataset = property(fget=lambda x:x.__testdataset,
                           fset=set_test_dataset)


class RegressionAsClassifier(ProxyClassifier):
    """Allows to use arbitrary regression for classification.

    Possible usecases:

     Binary Classification
      Any regression could easily be extended for binary
      classification. For instance using labels -1 and +1, regression
      results are quantized into labels depending on their signs
     Multiclass Classification
      Although most of the time classes are not ordered and do not
      have a corresponding distance matrix among them it might often
      be the case that there is a hypothesis that classes could be
      well separated in a projection to single dimension (non-linear
      manifold, or just linear projection).  For such use regression
      might provide necessary means of classification
    """

    distances = StateVariable(enabled=False,
        doc="Distances obtained during prediction")

    __sa_class__ = RegressionAsClassifierSensitivityAnalyzer

    def __init__(self, clf, centroids=None, distance_measure=None, **kwargs):
        """
        Parameters
        ----------
        clf : Classifier XXX Should become learner
          Regression to be used as a classifier.  Although it would
          accept any Learner, only providing regressions would make
          sense.
        centroids : None or dict of (float or iterable)
          Hypothesis or prior information on location/distance of
          centroids for each category, provide them.  If None -- during
          training it will choose equidistant points starting from 0.0.
          If dict -- keys should be a superset of labels of dataset
          obtained during training and each value should be numeric value
          or iterable if centroids are multidimensional and regression
          can do multidimensional regression.
        distance_measure : function or None
          What distance measure to use to find closest class label
          from continuous estimates provided by regression.  If None,
          will use Cartesian distance.
        """
        ProxyClassifier.__init__(self, clf, **kwargs)
        self.centroids = centroids
        self.distance_measure = distance_measure

        # Adjust tags which were copied from slave learner
        if self.__is_regression__:
            self.__tags__.pop(self.__tags__.index('regression'))

        # We can do any number of classes, although in most of the scenarios
        # multiclass performance would suck, unless there is a strong
        # hypothesis
        self.__tags__ += ['binary', 'multiclass', 'regression_based']

        # XXX No support for retrainable in RegressionAsClassifier yet
        if 'retrainable' in self.__tags__:
            self.__tags__.remove('retrainable')

        # Pylint/user friendliness
        #self._trained_ul = None
        self._trained_attrmap = None
        self._trained_centers = None


    def __repr__(self, prefixes=[]):
        if self.centroids is not None:
            prefixes = prefixes + ['centroids=%r'
                                   % self.centroids]
        if self.distance_measure is not None:
            prefixes = prefixes + ['distance_measure=%r'
                                   % self.distance_measure]
        return super(RegressionAsClassifier, self).__repr__(prefixes)


    def _train(self, dataset):
        targets_sa_name = self.params.targets
        targets_sa = dataset.sa[targets_sa_name]

        # May be it is an advanced one needing training.
        if hasattr(self.distance_measure, 'train'):
            self.distance_measure.train(dataset)

        # Centroids
        ul = dataset.sa[targets_sa_name].unique
        if self.centroids is None:
            # setup centroids -- equidistant points
            # XXX we might preferred -1/+1 for binary...
            centers = N.arange(len(ul), dtype=float)
        else:
            # verify centroids and assign
            if not set(self.centroids.keys()).issuperset(ul):
                raise ValueError, \
                      "Provided centroids with keys %s do not cover all " \
                      "labels provided during training: %s" \
                      % (self.centroids.keys(), ul)
            # override with superset
            ul = self.centroids.keys()
            centers = N.array([self.centroids[k] for k in ul])

        #self._trained_ul = ul
        # Map labels into indexes (not centers)
        # since later on we would need to get back (see ??? below)
        self._trained_attrmap = AttributeMap(
            map=dict([(l, i) for i,l in enumerate(ul)]),
            mapnumeric=True)
        self._trained_centers = centers

        # Create a shallow copy of dataset, and override labels
        # TODO: we could just bind .a, .fa, and copy only .sa
        dataset_relabeled = dataset.copy(deep=False)
        # ???:  may be we could just craft a monster attrmap
        #       which does min distance search upon to_literal ?
        dataset_relabeled.sa[targets_sa_name].value = \
            self._trained_attrmap.to_numeric(targets_sa.value)

        ProxyClassifier._train(self, dataset_relabeled)


    def _predict(self, dataset):
        # TODO: Probably we should forwardmap labels for target
        #       dataset so slave has proper statistics attached
        self.ca.estimates = regr_predictions \
                           = ProxyClassifier._predict(self, dataset)

        # Local bindings
        #ul = self._trained_ul
        attrmap = self._trained_attrmap
        centers = self._trained_centers
        distance_measure = self.distance_measure
        if distance_measure is None:
            distance_measure = cartesian_distance

        # Compute distances
        self.ca.distances = distances \
            = N.array([[distance_measure(s, c) for c in centers]
                       for s in regr_predictions])

        predictions = attrmap.to_literal(N.argmin(distances, axis=1))
        if __debug__:
            debug("CLF_", "Converted regression distances %(distances)s "
                  "into labels %(predictions)s for %(self_)s",
                      msgargs={'distances':distances, 'predictions':predictions,
                               'self_': self})

        return predictions


    ##REF: Name was automagically refactored
    def _set_retrainable(self, value, **kwargs):
        if value:
            raise NotImplementedError, \
                  "RegressionAsClassifier wrappers are not yet retrainable"
        ProxyClassifier._set_retrainable(self, value, **kwargs)
