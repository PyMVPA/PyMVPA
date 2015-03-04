# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Meta classifiers -- classifiers which use other classifiers or preprocessing

Meta Classifiers can be grouped according to their function as

:group BoostedClassifiers: CombinedClassifier MulticlassClassifier
  SplitClassifier
:group ProxyClassifiers: ProxyClassifier BinaryClassifier MappedClassifier
  FeatureSelectionClassifier
:group PredictionsCombiners for CombinedClassifier: PredictionsCombiner
  MaximalVote MeanPrediction

"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.misc.args import group_kwargs
from mvpa2.base.types import is_sequence_type, asobjarray
from mvpa2.base.param import Parameter

from mvpa2.datasets import Dataset

from mvpa2.generators.splitters import Splitter
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.datasets.miscfx import get_samples_by_attr
from mvpa2.misc.attrmap import AttributeMap
from mvpa2.base.dochelpers import _str, _repr_attrs
from mvpa2.base.state import ConditionalAttribute, ClassWithCollections

from mvpa2.clfs.base import Classifier
from mvpa2.clfs.distance import cartesian_distance
from mvpa2.misc.transformers import first_axis_mean

from mvpa2.measures.base import \
    BoostedClassifierSensitivityAnalyzer, ProxyClassifierSensitivityAnalyzer, \
    MappedClassifierSensitivityAnalyzer, \
    FeatureSelectionClassifierSensitivityAnalyzer, \
    RegressionAsClassifierSensitivityAnalyzer, \
    BinaryClassifierSensitivityAnalyzer, \
    _dont_force_slaves

from mvpa2.base import warning

if __debug__:
    from mvpa2.base import debug


class BoostedClassifier(Classifier):
    """Classifier containing the farm of other classifiers.

    Should rarely be used directly. Use one of its children instead
    """

    # should not be needed if we have prediction_estimates upstairs
    raw_predictions = ConditionalAttribute(enabled=False,
        doc="Predictions obtained from each classifier")

    raw_estimates = ConditionalAttribute(enabled=False,
        doc="Estimates obtained from each classifier")


    def __init__(self, clfs=None, propagate_ca=True,
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
        if self.params.retrainable:
            self.__changedData_isset = False


    def _get_feature_ids(self):
        """Custom _get_feature_ids for `BoostedClassifier`
        """
        # return union of all used features by slave classifiers
        feature_ids = set([])
        for clf in self.__clfs:
            feature_ids = feature_ids.union(set(clf.ca.feature_ids))
        return list(feature_ids)


    def _predict(self, dataset):
        """Predict using `BoostedClassifier`
        """
        raw_predictions = [ clf.predict(dataset) for clf in self.__clfs ]
        self.ca.raw_predictions = raw_predictions
        assert(len(self.__clfs)>0)
        if self.ca.is_enabled("estimates"):
            if np.array([x.ca.is_enabled("estimates")
                        for x in self.__clfs]).all():
                estimates = [ clf.ca.estimates for clf in self.__clfs ]
                self.ca.raw_estimates = estimates
            else:
                warning("One or more classifiers in %s has no 'estimates' state" %
                        self + "enabled, thus BoostedClassifier can't have" +
                        " 'raw_estimates' conditional attribute defined")

        return raw_predictions


    def _set_classifiers(self, clfs):
        """Set the classifiers used by the boosted classifier

        We have to allow to set list of classifiers after the object
        was actually created. It will be used by
        MulticlassClassifier
        """
        # tuple to guarantee immutability since we are asssigning
        # __tags__ below and rely on having clfs populated already
        self.__clfs = tuple(clfs) if clfs is not None else tuple()
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

    def _untrain(self):
        """Untrain `BoostedClassifier`

        Has to untrain any known classifier
        """
        if not self.trained:
            return
        for clf in self.clfs:
            clf.untrain()
        super(BoostedClassifier, self)._untrain()

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

    def __str__(self, *args, **kwargs):
        return super(ProxyClassifier, self).__str__(
            str(self.__clf), *args, **kwargs)

    def summary(self):
        s = super(ProxyClassifier, self).summary()
        if self.trained:
            s += "\n Slave classifier summary:" + \
                 '\n + %s' % \
                 (self.__clf.summary().replace('\n', '\n |'))
        return s

    def _set_retrainable(self, value, force=False):
        # XXX Lazy implementation
        self.clf._set_retrainable(value, force=force)
        super(ProxyClassifier, self)._set_retrainable(value, force)
        if value and not (self.ca['retrained']
                          is self.clf.ca['retrained']):
            if __debug__:
                debug("CLFPRX",
                      "Rebinding conditional attributes from slave clf %s", (self.clf,))
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


    def _untrain(self):
        """Untrain ProxyClassifier
        """
        if not self.__clf is None:
            self.__clf.untrain()
        super(ProxyClassifier, self)._untrain()


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    def get_sensitivity_analyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer

        Parameters
        ----------
        slave_kwargs : dict
          Arguments to be passed to the proxied (slave) classifier
        **kwargs
          Specific additional arguments for the sensitivity analyzer
          for the class.  See documentation of a corresponding `.__sa_class__`.
        """
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
          conditional attributes (value's) instead of pure prediction's
        dataset : Dataset
          training data in this case
        """
        # TODO: implement stacking to help with resolving ties
        pass


    def __call__(self, clfs, dataset):
        """Call function

        Parameters
        ----------
        clfs : list of Classifier
          List of classifiers to combine. Has to be classifiers (not
          pure predictions), since combiner might use some other
          conditional attributes (value's) instead of pure prediction's
        """
        raise NotImplementedError



class MaximalVote(PredictionsCombiner):
    """Provides a decision using maximal vote rule"""

    predictions = ConditionalAttribute(enabled=True,
        doc="Voted predictions")
    estimates = ConditionalAttribute(enabled=False,
        doc="Estimates keep counts across classifiers for each label/sample")

    # TODO: Might get a parameter to use raw decision estimates if
    # voting is not unambigous (ie two classes have equal number of
    # votes

    def __init__(self, **kwargs):
        PredictionsCombiner.__init__(self, **kwargs)


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
            # Lets check first if necessary conditional attribute is enabled
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
                       not is_sequence_type(prediction):
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
                        "same maximal vote %d. XXX disambiguate. " % maxv +
                        "Meanwhile selecting the first in sorted order")
            predictions.append(sorted(maxk)[0])

        ca = self.ca
        ca.estimates = all_label_counts
        ca.predictions = predictions
        return predictions



class MeanPrediction(PredictionsCombiner):
    """Provides a decision by taking mean of the results
    """

    predictions = ConditionalAttribute(enabled=True,
        doc="Mean predictions")

    estimates = ConditionalAttribute(enabled=True,
        doc="Predictions from all classifiers are stored")

    def __call__(self, clfs, dataset):
        """Actual callable - perform meaning

        """
        if len(clfs)==0:
            return []                   # to don't even bother

        all_predictions = []
        for clf in clfs:
            # Lets check first if necessary conditional attribute is enabled
            if not clf.ca.is_enabled("predictions"):
                raise ValueError, "MeanPrediction needs learners (such " \
                      " as %s) with state 'predictions' enabled" % clf
            all_predictions.append(clf.ca.predictions)

        # compute mean
        all_predictions = np.asarray(all_predictions)
        predictions = np.mean(all_predictions, axis=0)

        ca = self.ca
        ca.estimates = all_predictions
        ca.predictions = predictions
        return predictions


class ClassifierCombiner(PredictionsCombiner):
    """Provides a decision using training a classifier on predictions/estimates

    TODO: implement
    """

    predictions = ConditionalAttribute(enabled=True,
        doc="Trained predictions")


    def __init__(self, clf, variables=None):
        """Initialize `ClassifierCombiner`

        Parameters
        ----------
        clf : Classifier
          Classifier to train on the predictions
        variables : list of str
          List of conditional attributes stored in 'combined' classifiers, which
          to use as features for training this classifier
        """
        PredictionsCombiner.__init__(self)

        self.__clf = clf
        """Classifier to train on `variables` ca of provided classifiers"""

        if variables == None:
            variables = ['predictions']
        self.__variables = variables
        """What conditional attributes of the classifiers to use"""


    def _untrain(self):
        """It might be needed to untrain used classifier"""
        if self.__clf:
            self.__clf._untrain()

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

    def __init__(self, clfs=None, combiner='auto', **kwargs):
        """Initialize the instance.

        Parameters
        ----------
        clfs : list of Classifier
          list of classifier instances to use
        combiner : PredictionsCombiner, optional
          callable which takes care about combining multiple results into a single
          one. If default ('auto') chooses `MaximalVote` for classification and
          `MeanPrediction` for regression. If None is provided -- no combination is
          done
        kwargs : dict
          dict of keyworded arguments which might get used
          by State or Classifier

        NB: `combiner` might need to operate not on 'predictions' discrete
            labels but rather on raw 'class' estimates classifiers
            estimate (which is pretty much what is stored under
            `estimates`)
        """
        if clfs == None:
            clfs = []

        BoostedClassifier.__init__(self, clfs, **kwargs)

        self.__combiner = combiner
        """Input argument describing which "combiner" to use to combine results of multiple classifiers"""

        self._combiner = None
        """Actual combiner which would be decided upon later"""


    def __repr__(self, prefixes=[]):
        """Literal representation of `CombinedClassifier`.
        """
        return super(CombinedClassifier, self).__repr__(
            ["combiner=%s" % repr(self.__combiner)] + prefixes)

    @property
    def combiner(self):
        # Decide either we are dealing with regressions
        # by looking at 1st learner
        if self._combiner is None:
            if isinstance(self.__combiner, basestring) and self.__combiner == 'auto':
                self._combiner = (
                    MaximalVote,
                    MeanPrediction)[int(self.clfs[0].__is_regression__)]()
            else:
                self._combiner = self.__combiner
        return self._combiner

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


    def _untrain(self):
        """Untrain `CombinedClassifier`
        """
        try:
            self._combiner.untrain()
        except:
            pass
        finally:
            self._combiner = None
        super(CombinedClassifier, self)._untrain()


    def _train(self, dataset):
        """Train `CombinedClassifier`
        """
        BoostedClassifier._train(self, dataset)

        # combiner might need to be defined and trained as well at this point
        if self.combiner is not None:
            self.combiner.train(self.clfs, dataset)


    def _predict(self, dataset):
        """Predict using `CombinedClassifier`
        """
        ca = self.ca
        predictions = BoostedClassifier._predict(self, dataset)
        if self.combiner is not None:
            cca = self.combiner.ca
            if ca.is_enabled("estimates"):
                cca.enable('estimates')

            # combiner will make use of conditional attributes instead of only predictions
            # returned from _predict
            predictions = self.combiner(self.clfs, dataset)
        else:
            cca = None

        ca.predictions = predictions

        if ca.is_enabled("estimates") and cca is not None:
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
                                 /     \
                            animate  inanimate
                             /            \
                           SVM            SMLR
                         /     \         / | \ \
                    human    animal     5  6 7  8
                     |          |
                    SVM        SVM
                   /   \       /  \
                 male female monkey dog
                  1      2    3      4

    If it is desired to have a trailing node with a single label and
    thus without any classification, such as in

                       SVM
                      /   \
                     g1   g2
                     /     \
                    1     SVM
                          /  \
                         2    3

    then just specify None as the classifier to use::

        TreeClassifier(SVM(),
           {'g1':  ((1,), None),
            'g2':  ((1,2,3,4), SVM())})

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
          See :class:`~mvpa2.clfs.meta.TreeClassifier` for example
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

    def __str__(self, *args, **kwargs):
        return super(TreeClassifier, self).__str__(
            ', '.join(['%s: %s' % i for i in self.clfs.iteritems()]),
            *args, **kwargs)

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
        targets_sa_name = self.get_space()    # name of targets sa
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
            debug('CLFTREE', "Training primary %s on %s with targets %s",
                  (clf, ds_group, ds_group.sa[targets_sa_name].unique))
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
            clf = clfs[gk]
            group_labels = groups_labels[gk]
            if clf is None: # Trailing node
                if len(group_labels) != 1:
                    raise ValueError(
                        "Trailing nodes with no classifier assigned must have "
                        "only a single label associated. Got %s defined in "
                        "group %r of %s"
                        % (group_labels, gk, self))
            else:
                # select samples per each group
                ids = get_samples_by_attr(dataset, targets_sa_name, groups_labels[gk])
                ds_group = dataset[ids]
                if __debug__:
                    debug('CLFTREE', "Training %s for group %s on %s",
                          (clfs[gk], gk, ds_group))
                # and train corresponding slave clf
                clf.train(ds_group)


    def _untrain(self):
        """Untrain TreeClassifier
        """
        super(TreeClassifier, self)._untrain()
        for clf in self.clfs.values():
            if clf is not None:
                clf.untrain()


    def _predict(self, dataset):
        """
        """
        # Local bindings
        clfs, index2group, groups = self.clfs, self._index2group, self._groups
        clf_predictions = np.asanyarray(ProxyClassifier._predict(self, dataset))
        if __debug__:
                debug('CLFTREE',
                      'Predictions %s',
                      (clf_predictions))
        # assure that predictions are indexes, ie int
        clf_predictions = clf_predictions.astype(int)

        # now for predictions pointing to specific groups go into
        # corresponding one
        predictions = np.zeros((len(dataset),),
                               dtype=self.ca.trained_targets.dtype)
        for pred_group in set(clf_predictions):
            gk = index2group[pred_group]
            clf_ = clfs[gk]
            group_indexes = (clf_predictions == pred_group)
            if __debug__:
                debug('CLFTREE',
                      'Predicting for group %s using %s on %d samples',
                      (gk, clf_, np.sum(group_indexes)))
            if clf_ is None:
                predictions[group_indexes] = groups[gk][0] # our only label
            else:
                predictions[group_indexes] = clf_.predict(dataset[group_indexes])
                
        return predictions


class BinaryClassifier(ProxyClassifier):
    """`ProxyClassifier` which maps set of two labels into +1 and -1
    """

    __sa_class__ = BinaryClassifierSensitivityAnalyzer

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

        # TODO: move to use AttributeMap
        #self._attrmap = AttributeMap(dict([(l, -1) for l in sneglabels] +
        #                                  [(l, +1) for l in sposlabels]))

        # check if there is no overlap
        overlap = set(poslabels).intersection(neglabels)
        if len(overlap)>0:
            raise ValueError("Sets of positive and negative labels for " +
                "BinaryClassifier must not overlap. Got overlap " %
                overlap)

        self.__poslabels = poslabels
        self.__neglabels = neglabels

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

    @property
    def poslabels(self):
        return self.__poslabels

    @property
    def neglabels(self):
        return self.__neglabels

    def __repr__(self, prefixes=[]):
        prefix = "poslabels=%s, neglabels=%s" % (
            repr(self.__poslabels), repr(self.__neglabels))
        return super(BinaryClassifier, self).__repr__([prefix] + prefixes)


    def _train(self, dataset):
        """Train `BinaryClassifier`
        """
        targets_sa_name = self.get_space()
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
                      "Created shallow copy with %d samples for binary "
                      "classification among labels %s/+1 and %s/-1",
                      (dataset.nsamples, self.__poslabels, self.__neglabels))
        else:
            datasetselected = dataset[[ x[0] for x in idlabels ]]
            if __debug__:
                debug('CLFBIN',
                      "Selected %d samples out of %d samples for binary "
                      "classification among labels %s/+1 and %s/-1. Selected %s",
                      (len(idlabels), dataset.nsamples,
                       self.__poslabels, self.__neglabels, datasetselected))

        # adjust the labels
        datasetselected.sa[targets_sa_name].value = [ x[1] for x in idlabels ]

        # now we got a dataset with only 2 labels
        if __debug__:
            assert(set(datasetselected.sa[targets_sa_name].unique) ==
                   set([-1, 1]))

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
    """Perform multiclass classification using a list of binary classifiers.

    Based on a `CombinedClassifier` for which it constructs a list of
    binary 1-vs-1 (ie in pairs like LIBSVM does) or 1-vs-all (which is
    yet to think about) classifiers.
    """

    raw_predictions_ds = ConditionalAttribute(enabled=False,
        doc="Wraps raw_predictions into a Dataset with .fa.(neg,pos) "
        "describing actual labels used in each binary classification task "
        "and samples containing actual decision labels per each input "
        "sample")

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

        # adhere to slave classifier capabilities
        if clf is not None:
            self.__tags__ += clf.__tags__
        if not 'multiclass' in self.__tags__:
            self.__tags__ += ['multiclass']

        # Some checks on known ways to do multiclass
        if bclf_type == "1-vs-1":
            pass
        elif bclf_type == "1-vs-all": # TODO
            raise NotImplementedError
        else:
            raise ValueError(
                  "Unknown type of classifier %s for " % bclf_type +
                  "MulticlassClassifier")
        self.__bclf_type = bclf_type

    # XXX fix it up a bit... it seems that MulticlassClassifier should
    # be actually ProxyClassifier and use BoostedClassifier internally
    def __repr__(self, prefixes=[]):
        prefix = "bclf_type=%s, clf=%s" % (repr(self.__bclf_type),
                                            repr(self.__clf))
        return super(MulticlassClassifier, self).__repr__([prefix] + prefixes)

    def _get_binary_pairs(self, dataset):
        """Return a list of pairs of categories lists to be used in binary classification
        """
        targets_sa_name = self.get_space()

        # construct binary classifiers
        ulabels = dataset.sa[targets_sa_name].unique

        if self.__bclf_type == "1-vs-1":
            # generate pairs and corresponding classifiers
            # could use _product but let's stay inline with previuos
            # implementation
            label_pairs = [([ulabels[i]], [ulabels[j]])
                           for i in xrange(len(ulabels))
                           for j in xrange(i+1, len(ulabels))]
            if __debug__:
                debug("CLFMC", "Created %d label pairs for original %d labels",
                      (len(label_pairs), len(ulabels)))
        elif self.__bclf_type == "1-vs-all":
            raise NotImplementedError

        return label_pairs

    def _train(self, dataset):
        """Train classifier
        """
        # construct binary classifiers
        biclfs = []
        for poslabels, neglabels in self._get_binary_pairs(dataset):
            biclfs.append(
                BinaryClassifier(self.__clf.clone(),
                                 poslabels=poslabels,
                                 neglabels=neglabels))
        self.clfs = biclfs                # need to be set after, not operated in-place
        # perform actual training
        CombinedClassifier._train(self, dataset)

    def _predict(self, dataset):
        ca = self.ca
        if ca.is_enabled("raw_predictions_ds"):
            ca.enable("raw_predictions")

        predictions = super(MulticlassClassifier, self)._predict(dataset)

        if ca.is_enabled("raw_predictions_ds") or self.combiner is None:
            if self.combiner is None:
                raw_predictions = predictions
            else:
                # we should fetch those from ca
                raw_predictions = ca.raw_predictions

            # assign pos and neg to fa while squeezing out
            # degenerate dimensions which are there to possibly accomodate
            # 1-vs-all cases

            # for consistency -- place into object array of tuples
            # (Sensitivity analyzers already do the same)
            pairs = zip(np.array([np.squeeze(clf.neglabels) for clf in self.clfs]).tolist(),
                        np.array([np.squeeze(clf.poslabels) for clf in self.clfs]).tolist())
            ca.raw_predictions_ds = raw_predictions_ds = \
                Dataset(np.array(raw_predictions).T, fa={self.space: asobjarray(pairs)})
        if self.combiner is None:
            return raw_predictions_ds
        else:
            return predictions


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

    stats = ConditionalAttribute(enabled=False,
        doc="Resultant confusion whenever classifier trained " +
            "on 1 part and tested on 2nd part of each split")

    splits = ConditionalAttribute(enabled=False, doc=
       """Store the actual splits of the data. Can be memory expensive""")

    # ??? couldn't be training_stats since it has other meaning
    #     here, BUT it is named so within CrossValidatedTransferError
    #     -- unify
    #  decided to go with overriding semantics tiny bit. For split
    #     classifier training_stats would correspond to summary
    #     over training errors across all splits. Later on if need comes
    #     we might want to implement global_training_stats which would
    #     correspond to overall confusion on full training dataset as it is
    #     done in base Classifier
    #global_training_stats = ConditionalAttribute(enabled=False,
    #    doc="Summary over training confusions acquired at each split")

    def __init__(self, clf, partitioner=NFoldPartitioner(),
                 splitter=Splitter('partitions', count=2), **kwargs):
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

        self.__partitioner = partitioner
        self.__splitter = splitter


    def _train(self, dataset):
        """Train `SplitClassifier`
        """
        targets_sa_name = self.get_space()

        # generate pairs and corresponding classifiers
        bclfs = []

        # local binding
        ca = self.ca

        clf_template = self.__clf
        if ca.is_enabled('stats'):
            ca.stats = clf_template.__summary_class__()
        if ca.is_enabled('training_stats'):
            clf_template.ca.enable(['training_stats'])
            ca.training_stats = clf_template.__summary_class__()

        clf_hastestdataset = hasattr(clf_template, 'testdataset')

        self.ca.splits = []

        for i, pset in enumerate(self.__partitioner.generate(dataset)):
            if __debug__:
                debug("CLFSPL_", "Deepcopying %s for %s",
                      (clf_template, self))
            clf = clf_template.clone()
            bclfs.append(clf)

            if __debug__:
                debug("CLFSPL", "Training classifier for split %d", (i,))

            # split partitioned dataset
            split = [d for d in self.__splitter.generate(pset)]

            if ca.is_enabled("splits"):
                self.ca.splits.append(split)

            clf = bclfs[i]

            # assign testing dataset if given classifier can digest it
            if clf_hastestdataset:
                clf.testdataset = split[1]

            clf.train(split[0])

            # unbind the testdataset from the classifier
            if clf_hastestdataset:
                clf.testdataset = None

            if ca.is_enabled("stats"):
                predictions = clf.predict(split[1])
                self.ca.stats.add(split[1].sa[targets_sa_name].value,
                                          predictions,
                                          clf.ca.get('estimates', None))
                if __debug__:
                    dact = debug.active
                    if 'CLFSPL_' in dact:
                        debug('CLFSPL_', 'Split %d:\n%s',
                              (i, self.ca.stats))
                    elif 'CLFSPL' in dact:
                        debug('CLFSPL', 'Split %d error %.2f%%',
                              (i, self.ca.stats.summaries[-1].error))

            if ca.is_enabled("training_stats"):
                # XXX this is broken, as it cannot deal with not yet set ca
                ca.training_stats += clf.ca.training_stats
        # need to be assigned after the entire list populated since
        # _set_classifiers places them into a tuple
        self.clfs = bclfs


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    def get_sensitivity_analyzer(self, slave_kwargs={}, **kwargs):
        """Return an appropriate SensitivityAnalyzer for `SplitClassifier`

        Parameters
        ----------
        combiner
          If not provided, `first_axis_mean` is assumed
        """
        return BoostedClassifierSensitivityAnalyzer(
                self, sa_attr='splits',
                analyzer=self.__clf.get_sensitivity_analyzer(
                    **_dont_force_slaves(slave_kwargs)),
                **kwargs)

    partitioner = property(fget=lambda x:x.__partitioner,
                        doc="Partitioner used by SplitClassifier")
    splitter = property(fget=lambda x:x.__splitter,
                        doc="Splitter used by SplitClassifier")


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
        if __debug__:
            debug('CLF', "Training %s having mapped dataset into %s",
                  (self, wdataset))
        ProxyClassifier._train(self, wdataset)


    def _untrain(self):
        """Untrain `FeatureSelectionClassifier`

        Has to untrain any known classifier
        """
        # untrain the mapper
        if self.__mapper is not None:
            self.__mapper.untrain()
        # let base class untrain as well
        super(MappedClassifier, self)._untrain()


    def _predict(self, dataset):
        """Predict using `MappedClassifier`
        """
        return ProxyClassifier._predict(self, self.__mapper.forward(dataset))


    def __repr__(self, prefixes=[]):
        return super(MappedClassifier, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['mapper']))

    def __str__(self, *args, **kwargs):
        return super(MappedClassifier, self).__str__(
              str(self.mapper), *args, **kwargs)

    mapper = property(lambda x:x.__mapper, doc="Used mapper")



class FeatureSelectionClassifier(MappedClassifier):
    """This is nothing but a `MappedClassifier`.

    This class is only kept for (temporary) compatibility with old code.
    """
    __tags__ = [ 'does_feature_selection', 'meta' ]


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

    distances = ConditionalAttribute(enabled=False,
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
        targets_sa_name = self.get_space()
        targets_sa = dataset.sa[targets_sa_name]

        # May be it is an advanced one needing training.
        if hasattr(self.distance_measure, 'train'):
            self.distance_measure.train(dataset)

        # Centroids
        ul = dataset.sa[targets_sa_name].unique
        if self.centroids is None:
            # setup centroids -- equidistant points
            # XXX we might preferred -1/+1 for binary...
            centers = np.arange(len(ul), dtype=float)
        else:
            # verify centroids and assign
            if not set(self.centroids.keys()).issuperset(ul):
                raise ValueError, \
                      "Provided centroids with keys %s do not cover all " \
                      "labels provided during training: %s" \
                      % (self.centroids.keys(), ul)
            # override with superset
            ul = self.centroids.keys()
            centers = np.array([self.centroids[k] for k in ul])

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
            = np.array([[distance_measure(s, c) for c in centers]
                       for s in regr_predictions])

        predictions = attrmap.to_literal(np.argmin(distances, axis=1))
        if __debug__:
            debug("CLF_", "Converted regression distances %s "
                  "into labels %s for %s", (distances, predictions, self))

        return predictions


    def _set_retrainable(self, value, **kwargs):
        if value:
            raise NotImplementedError, \
                  "RegressionAsClassifier wrappers are not yet retrainable"
        ProxyClassifier._set_retrainable(self, value, **kwargs)
