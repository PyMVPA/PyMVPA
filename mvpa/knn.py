### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    k-Nearest-Neighbour classification
#
#    Copyright (C) 2006-2007 by
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

import numpy as N

try:
    import psyco
    psyco.profile()
except:
    pass

class kNN:
    """ k-nearest-neighbour classification.
    """
    def __init__(self, k=2):
        """
        Parameters:
          pattern: MVPAPattern object containing all data.
          k:       number of nearest neighbours to be used for voting
        """
        self.__k = k
        self.__votingfx = self.getWeightedVote
        self.verbose = False


    def train( self, data ):
        self.__data = data


    def predict(self, data):
        """ Predict the class labels for the provided data.

        Returns a list of class labels (one for each data sample).
        """
        # make sure we're talking about arrays
        data = N.array( data )

        if not data.ndim == 2:
            raise ValueError, "Data array must be two-dimensional."

        if not data.shape[1] == self.__data.nfeatures:
            raise ValueError, "Length of data samples (features) does " \
                              "not match the classifier."

        # predicted class labels will go here
        predicted = []

        # for all test pattern
        for p in data:
            # calc the euclidean distance of the pattern vector to all
            # patterns in the training data
            dists = N.sqrt(
                        N.sum(
                            N.abs( self.__data.pattern - p )**2, axis=1
                            )
                        )
            # get the k nearest neighbours from the sorted list of distances 
            knn = dists.argsort()[:self.__k]

            # finally get the class label
            predicted.append( self.__votingfx(knn) )

        return predicted


    def getMajorityVote(self, knn_ids):
        # create dictionary with an item for each condition
        votes = dict( zip ( self.__data.reglabels,
                            [0 for i in self.__data.reglabels ] ) )

        # add 1 to a certain condition per NN
        for nn in knn_ids:
            votes[self.__data.reg[nn]] += 1

        # find the condition with most votes
        best_cond = None; most_votes = None
        for cond, vote in votes.iteritems():
            if best_cond is None or vote > most_votes:
                best_cond = cond; most_votes = vote

        return best_cond


    def getWeightedVote(self, knn_ids):
        # create dictionary with an item for each condition
        votes = dict( zip ( self.__data.reglabels, 
                            [0 for i in self.__data.reglabels ] ) )
        weights = dict( zip ( self.__data.reglabels,
                    [ 1 - ( float( self.__data.reg.tolist().count(i) ) \
                      / len(self.__data.reg) )
                        for i in self.__data.reglabels ] ) )

        for nn in knn_ids:
            votes[self.__data.reg[nn]] += weights[self.__data.reg[nn]]

        # find the condition with most votes
        best_cond = None; most_votes = None
        for cond, vote in votes.iteritems():
            if best_cond is None or vote > most_votes:
                best_cond = cond; most_votes = vote

        return best_cond
