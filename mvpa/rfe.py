### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Recursive feature elimination
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import svm
import support


class RecursiveFeatureElimination( object ):
    """
    """
    def __init__( self, pattern, clf = svm.SVM, **kwargs ):
        self.__pattern = pattern
        self.__verbose = False

        self.__clf_class = clf
        if len( kwargs ) == 0:
            self.__clf_args = {'kernel_type': svm.libsvm.LINEAR}
        else:
            self.__clf_args = kwargs

        # train the classifier with the initial feature set
        self.__trainClassifier()


    def __trainClassifier(self):
        self.__clf = self.__clf_class(self.pattern, **self.__clf_args)
        self.__testClassifierOnTrainingSet()

        # feature rating: large values == important
        fr = self.__clf.getFeatureBenchmark()


    def __testClassifierOnTrainingSet(self):
        predictions = np.array(self.__clf.predict( self.pattern.pattern ))

        self.__training_confusion_mat = \
            support.buildConfusionMatrix( self.pattern.reglabels,
                                          self.pattern.reg,
                                          predictions )
        self.__training_perf = \
            float(self.trainconfmat.diagonal().sum()) / self.trainconfmat.sum()


    def selectFeatures( self, n, eliminate_by = 'fraction',
                        kill_per_iter = 0.1 ):
        """ Select n predictive features from the dataset.
        """
        # are we talking integers?
        n = int( n )

        # do it until there are more feature than there should be
        while n < self.pattern.nfeatures:
            # get importance of each remaining feature
            featrank = self.__clf.getFeatureBenchmark()

            # determine the number of features to be eliminated
            if eliminate_by == 'fraction':
                nkill = int( len(featrank) * kill_per_iter )
            elif eliminate_by == 'number':
                nkill = kill_per_iter
            else:
                raise ValueError, 'Unknown elimination method: ' \
                                  + str(eliminate_by)

            # limit number of eliminations if necessary (would have less than
            # n features after elimination)
            if nkill > self.pattern.nfeatures - n:
                nkill = self.pattern.nfeatures - n

            # make sure to remove one feature at least to prevent endless
            # loops
            if nkill < 1:
                nkill = 1

            if self.__verbose:
                print "Removing", nkill, "features."

            # eliminate the first 'nkill' features (with the lowest ranking)
            self.__pattern = \
                self.pattern.selectFeatures( 
                        (featrank.argsort()[nkill:]).tolist() )

            # retrain the classifier with the new feature set
            self.__trainClassifier()


    def killNFeatures( self, n,
                       eliminate_by = 'number',
                       kill_per_iter = 1 ):
        """ Eliminates the 'n' least predictive features.

        Please see the documentation of the selectFeatures() methods for
        information about the remaining arguments.
        """
        self.selectFeatures( self.pattern.nfeatures - n,
                             eliminate_by = eliminate_by,
                             kill_per_iter = kill_per_iter )


    def killFeatureFraction( self, frac,
                             eliminate_by = 'fraction',
                             kill_per_iter = 0.1 ):
        """ Eliminates the 100*frac percent least predictive features.

        Please see the documentation of the selectFeatures() methods for
        information about the remaining arguments.
        """
        nkill = int(frac * self.pattern.nfeatures)
        self.selectFeatures( self.pattern.nfeatures - nkill,
                             eliminate_by = eliminate_by,
                             kill_per_iter = kill_per_iter )


    def testSelection( self, pattern ):
        """
        'pattern' is an MVPAPattern object.
        """
        if not pattern.nfeatures == self.pattern.nfeatures:
            # select necessary features from the provided patterns
            masked = pattern.selectFeaturesByMask( 
                            self.pattern.getFeatureMask() )
        else:
            masked = pattern

        # get the predictions from the classifier
        predicted = self.clf.predict( masked.pattern )

        print self.pattern.reglabels
        confmat = support.buildConfusionMatrix( self.pattern.reglabels,
                                                masked.reg,
                                                predicted )

        perf = np.mean( predicted == masked.reg )

        return predicted, perf, confmat


    def setVerbosity( self, value ):
        self.__verbose = value


    # properties
    pattern = property( fget=lambda self: self.__pattern )
    verbosity = property( fget=lambda self: self.__verbose,
                          fset=setVerbosity )
    clf = property( fget=lambda self: self.__clf )
    trainconfmat = property( fget=lambda self: self.__training_confusion_mat )
    trainperf = property( fget=lambda self: self.__training_perf )
