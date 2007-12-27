#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA classifiers

Module Organization
===================
mvpa.clfs module contains various classifiers

.. packagetree::
   :style: UML

Classifiers can be grouped according to their function as
:group Basic: kNN svm
:group BoostedClassifiers -- use set of other classifier: BoostedClassifier CombinedClassifier MulticlassClassifier SplitClassifier
:group ProxyClassifiers -- use other classifier while altering input data: BinaryClassifier MappedClassifier FeatureSelectionClassifier
:group Combiners -- functors to group results for CombinedClassifier: Combiner MaximalVote
"""

__docformat__ = 'restructuredtext'

from classifier import *
from knn import *
from svm import *
