#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Abstract base class for all classifiers."""

from mvpa.misc.state import State

class Classifier(State):
    """
    Required behavior:

    For every classifier is has to be possible to be instanciated without
    having to specify the training pattern.

    Repeated calls to the train() method with different training data have to
    result in a valid classifier, trained for the particular dataset.

    It must be possible to specify all classifier parameters as keyword
    arguments to the constructor.

    Recommended behavior:

    Derived classifiers should provide access to *decision_values* -- i.e. that
    information that is finally used to determine the predicted class label.

    Michael: Maybe it works well if each classifier provides a 'decision_value'
             state member. This variable is a list as long as and in same order
             as Dataset.uniquelabels (training data). Each item in the list
             corresponds to the likelyhood of a sample to belong to the
             respective class. However the sematics might differ between
             classifiers, e.g. kNN would probably store distances to class-
             neighbours, where PLF would store the raw function value of the
             logistic function. So in the case of kNN low is predictive and for
             PLF high is predictive. Don't know if there is the need to unify
             that.

             As the storage and/or computation of this information might be
             demanding its collection should be switchable and off be default.
    """
    # Dict that contains the parameters of a classifier.
    # This shall provide an interface to plug generic parameter optimizer
    # on all classifiers (e.g. grid- or line-search optimizer)
    # A dictionary is used because Michael thinks that access by name is nicer.
    # Additonally Michael thinks ATM that additonal information might be
    # necessary in some situations (e.g. reasonably predefined parameter range,
    # minimal iteration stepsize, ...), therefore the value to each key should
    # also be a dict or we should use mvpa.misc.param.Parameter'...
    params = {}

    def __init__(self):
        """Cheap initialization.
        """
        State.__init__(self)


    def train(self, data):
        """
        """
        raise NotImplementedError


    def predict(self, data):
        """
        """
        raise NotImplementedError



# MultiClass
# One-vs-One   One-vs-All  TreeClassifier
#
# BoostedClassifier
#

"""
yoh's Thoughts:

MultiClass can actually delegate decision making to some
BoostedClassifier which works with a set of classifiers to derive the
final solution. Then the job of MultiClass parametrized with a
classifier to use, to decorate the classifier so that decorator
selects only 2 given classes for training.

So MultiClass.__init__ creates decorated classifiers and creates/initializes a
BoostedClassifier instance based on that set.

MultiClass.train() would simply call BoostedClassifier

The aspect to keep in mind is the resultant sensitivities
"""

class BoostedClassifier(Classifier, State):
    """

    """

    def __init__(self, clss, combiner, **kargs):
        """Initialize the instance.

        :Parameters:
          `clss` : list
            list of classifier instances to use
          `combiner`
            callable which takes care about combining multiple
            results into a single one (e.g. maximal vote)
          **kargs : dict
            dict of keyworded arguments which might get used
            by State or Classifier
        """
        Classifier.__init__(self)
        State.__init__(self, **kargs)
        self.__clss = clss
        """Classifiers to use"""

        self._registerState("predictions", enabled=True)


    def train(self, data):
        """
        """
        for cls in self.__clss:
            cls.train(data)


    def predict(self, data):
        """
        """
        predictions = [ cls.predict(data) for cls in self.__clss ]

        if self.isStateEnabled("predictions"):
            self["predictions"] = predictions

        return self.__combiner(predictions)

    classifiers = property(lambda x:x.__clss, doc="Used classifiers")


class MulticlassClassifier(Classifier):
    """ Classifier to perform multiclass classification using a set of simple classifiers

    such as 1-vs-1 or 1-vs-all
    """

    def __init__(self, cls, bcls):
        """Initialize the instance

        :Parameters:
          `clf` : `Classifier`
            classifier based on which multiple classifiers are created
            for multiclass
          `boostedcls` : `BoostedClassifier`
            classifier used to aggregate "pairClassifier"s
          """
        Classifier.__init__(self)
        self.__bcls = bcls


    def train(self, data):
        """
        """
        self.__bcls.train(data)


    def predict(self, data):
        """
        """
        return self.__bcls.predict(data)

    classifiers = property(lambda x:x.__bcls.__clss, doc="Used classifiers")
