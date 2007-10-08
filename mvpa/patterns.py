### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Pattern handling and manipulation
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

import numpy
import operator
import random

class MVPAPattern(object):
    """ This class provides a container to store all necessary data to perform
    MVPA analyses. That is the actual pattern data, as well as the regressors
    associated with these patterns. Additionally the origin of each pattern is
    stored to be able to group patterns for cross-validation purposes.
    """

    def __init__( self, pattern, reg, origin = None, mask = None, **kwargs ):
        """ Initialize the pattern data.

        The pattern data is finally loaded by calling
        MVPAPattern.addPattern(). Please see the documentation of this method
        to learn what kind of data is required or supported.

        Parameters:
          pattern -
          reg     -
          origin  -
          mask    -

        Please ignore the additional keyword arguments. They are only necessary
        for internal stuff. You cannot trust them -- do not use them!
        """
        # initialize containers
        if mask == None:
            self.__mask = None
        elif isinstance( mask, numpy.ndarray):
            self.__mask = mask
        else:
            raise ValueError, "Mask has to be NumPy array."

        self.__patterns = None
        self.__regs = None
        self.__origins = None
        self.__origregs = None

        # when in internal mode bypass all checks and directly assign
        # the data. Otherwise call addPattern() that checks if the data
        # is in good shape
        if kwargs.has_key('internal'):
            self.__initDataNoChecks( pattern, reg, origin, **(kwargs) )
        else:
            self.addPattern( pattern, reg, origin )


    def __iadd__( self, other ):
        """ Merge the patterns of one MVPAPattern object to another (in-place).

        Please note that the patterns, regressors and origins are simply
        concatenated to create an MVPAPattern object that contains the
        patterns of both objects. No further processing is done. In particular
        the origin values are not modified: Patterns with the same origin from
        both MVPAPattern object will still share the same origin and will
        therefore be treated as they would belong together by the
        CrossValidation algorithm.
        """
        if not (self.__mask == other.__mask).all():
            raise ValueError, "Cannot add MVPAPattern, because pattern masks " \
                              "do not match."

        self.__patterns = \
            numpy.concatenate( ( self.__patterns, other.__patterns ), axis=0)
        self.__regs = \
            numpy.concatenate( ( self.__regs, other.__regs ), axis=0)
        self.__origins = \
            numpy.concatenate( ( self.__origins, other.__origins ), axis=0)

        return self


    def __add__( self, other ):
        """ Merge the patterns of two MVPAPattern objects.

        Please note that the patterns, regressors and origins are simply
        concatenated to create an MVPAPattern object that contains the
        patterns of both objects. No further processing is done. In particular
        the origin values are not modified: Patterns with the same origin from
        both MVPAPattern object will still share the same origin and will
        therefore be treated as they would belong together by the
        CrossValidation algorithm.
        """
        out = MVPAPattern( self.__patterns,
                           self.__regs,
                           self.__origins,
                           mask = self.__mask,
                           internal = True )

        out += other

        return out


    def addPattern( self, pattern, reg, origin = None ):
        """ Adds one or more pattern dataset(s).

        Pattern can be an array or sequence with an arbitrary number of
        dimensions. Internally the pattern data will be converted into a
        2d matrix ( patterns x features ). This data can later be accessed
        by using to 'pattern' property of this class.

        When adding more patterns to an object that already holds some
        patterns, the new pattern data has to match the shape of the ones that
        are already loaded. An example: one first loads pattern data with shape
        (10,2,3,4). This means 10 patterns shaped (2,3,4). Any pattern that
        shall be loaded later has to be shaped <number of patterns> x (2,3,4).

        If a feature mask is present the pattern will be masked and only
        features corresponding to non-zero mask elements will end up in the 2d
        pattern matrix.

        A 1d array or simple sequence passed as 'pattern' arguments is assumed
        to be a single pattern ( 1 x <number of features> ).

        'reg' can be a single scalar or a sequence of value. These regressors
        represent the target value for the classification, e.g. class labels.
        If 'reg' is a scalar this value is used as regressor for all patterns
        in 'pattern'. If 'reg' is a sequence its length has to match the number
        of patterns and the values in 'reg' are associated with the
        corresponding patterns.

        The 'origin' parameter is used to associate patterns with each other.
        This can be useful to exclude/include patterns set in a
        cross-validation run. 'origin' is treated similar to the 'reg'
        argument: scalars and sequences are supported. Additionally 'origin'
        can also be 'None'. In this case a unique origin value is automatically
        calculated for each pattern individually (this also takes already
        present patterns into account).
        """
        # 1d arrays or simple sequences are assumed to be a single pattern
        if (not isinstance(pattern, numpy.ndarray)) or pattern.ndim < 2:
            pattern = numpy.array( pattern, ndmin=2 )

        # store the shape of a single pattern for later coordinate
        # reconstruction
        if self.__mask == None:
            self.__mask = numpy.ones(pattern.shape[1:], dtype='bool')
        else:
            if self.__mask.shape != pattern.shape[1:]:
                raise ValueError, "Pattern shape does not match existing" \
                                  " patterns (exist: %s, new: %s)" \
                                  % ( str(self.__mask.shape),
                                      str(pattern.shape[1:]) )

        # now reshape into a 2d array
        if not pattern.ndim == 2:
            pattern = pattern.reshape( len( pattern ),
                                       numpy.prod( pattern.shape[1:] ) )

        # apply feature mask: use '>0' to cope with non-binary masks
        pattern = pattern[:, self.__mask.ravel() > 0]

        # simply assign or concatenate if already present
        if self.__patterns == None:
            self.__patterns = pattern
        else:
            self.__patterns = numpy.concatenate( (self.__patterns, pattern),
                                                axis=0 )

        # check if regs is supplied as a sequence
        try:
            if len( reg ) != len( pattern ):
                raise ValueError, "Length of 'reg' has to match the number" \
                                  " of patterns."
            # store the sequence as array
            reg = numpy.array( reg )

        except TypeError:
            # make sequence of identical value matching the number of patterns
            reg = numpy.repeat( reg, len( pattern ) )

        # simply assign or concatenate if already present
        if self.__regs == None:
            self.__regs = reg
        else:
            self.__regs = numpy.concatenate( (self.__regs, reg), axis=0 )

        # if no origin is given assume that every pattern has its own
        if origin == None:
            origin = numpy.arange( len( pattern ) )
            if self.__origins != None:
                origin += self.__origins.max() + 1
        else:
            try:
                if len( origin ) != len( pattern ):
                    raise ValueError, "Length of 'origin' has to match the" \
                                      " number of patterns."
                # store the sequence as array
                origin = numpy.array( origin )

            except TypeError:
                # make sequence of identical value matching the number of
                # patterns
                origin = numpy.repeat( origin, len( pattern ) )

        # simply assign or concatenate if already present
        if self.__origins == None:
            self.__origins = origin
        else:
            self.__origins = numpy.concatenate( (self.__origins, origin ),
                                                axis=0 )


    def __initDataNoChecks(self, pat, reg, orig, **kwargs):
        """ Do not use! """
        self.__patterns = pat
        self.__regs = reg
        self.__origins = orig


    def permutatedRegressors( self, status, perorigin = True ):
        """ Permutate the regressors.

        Calling this method with 'status' set to True, the regressors are
        permutated among all patterns.

        If 'perorigin' is True permutation is limited to patterns sharing the
        same origin value. Therefore only the association of a certain pattern
        with a regressor is permutated while keeping the absolute number of
        occurences of each regressor value with a certain origin constant.

        If 'status' is False the original regressors are restored.
        """

        if not status:
            # restore originals
            if self.__origregs == None:
                raise RuntimeError, 'Cannot restore regressors. ' \
                                    'randomizedRegressors() has never been ' \
                                    'called with status == True.'
            self.__regs = self.__origregs
            self.__origregs = None
        else:
            # permutate regs per origin

            # make a backup of the original regressors
            self.__origregs = self.__regs.copy()

            # now scramble the rest
            if perorigin:
                for o in self.originlabels:
                    self.__regs[self.__origins == o ] = \
                        numpy.random.permutation(
                                self.__regs[ self.__origins == o ] )
            else:
                self.__regs = numpy.random.permutation( self.__regs )


    def zscore( self, mean = None, std = None, origin=True ):
        """ Z-Score the pattern data.

        'mean' and 'std' can be used to pass custom values to the z-scoring.
        Both may be scalars or arrays.

        All computations are done in place. Data upcasting is done
        automatically if necessary.

        If origin is True patterns with the same origin are z-scored independent
        of patterns with other origin values, e.i. mean and standard deviation
        are calculated individually.
        """
        # cast to floating point datatype if necessary
        if str(self.__patterns.dtype).startswith('uint') \
           or str(self.__patterns.dtype).startswith('int'):
            self.__patterns = self.__patterns.astype('float64')

        def doit(pat, mean, std):
            # calculate mean if necessary
            if not mean:
                mean = pat.mean(axis=0)

            # calculate std-deviation if necessary
            if not std:
                std = pat.std(axis=0)

            # do the z-scoring
            pat -= mean
            pat /= std

            return pat

        if origin:
            for o in self.originlabels:
                self.__patterns[self.__origins == o] = \
                    doit( self.__patterns[self.__origins == o], mean, std )
        else:
            doit( self.__patterns, mean, std )


    def getPatternSample( self, nperreg ):
        """ Select a random sample of patterns.

        If 'nperreg' is an integer value, the specified number of patterns is
        randomly choosen from the group of patterns sharing a unique regressor
        value ( total number of selected patterns: nperreg x len(reglabels).

        If 'nperreg' is a list its length has to match the number of unique
        regressor labels. In this case 'nperreg' specifies the number of
        patters that shall be selected from the patterns with the corresponding
        regressor label.

        The method returns a MVPAPattern object containing the selected
        patterns.
        """
        # if interger is given take this value for all classes
        if isinstance(nperreg, int):
            nperreg = [ nperreg for i in self.reglabels ]

        sample = []
        # for each available class
        for i,r in enumerate(self.reglabels):
            # get the list of pattern ids for this class
            sample += random.sample( (self.reg == r).nonzero()[0],
                                     nperreg[i] )

        return self.selectPatterns( sample )


    def selectPatterns( self, mask ):
        """ Choose a subset of patterns.

        Returns a new MVPAPattern object containing the selected pattern
        subset.
        """
        # without having a sequence a index the masked pattern array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        return MVPAPattern( self.__patterns[mask,],
                            self.__regs[mask,],
                            self.__origins[mask,],
                            mask = self.__mask,
                            internal=True )


    def buildFeatureMaskFromIds( self, ids ):
        """ Returns a mask with a features in ids selected from the
        current feature set.
        """
        # sort to preserve feature order (otherwise mask gets corrupted)
        ids = sorted( ids )
        mask = numpy.zeros( self.__mask.shape, dtype='bool' )
        mask[ tuple( numpy.transpose(self.__mask.nonzero())[ids].T ) ] = True

        return mask


    def selectFeaturesById( self, ids, maintain_mask = True ):
        """ Select a number of features from the current set.

        Parameters:
            ids           - list of feature IDs
            maintain_mask - This will take the current feature mask and remove
                            all unselected features from it. This modified mask
                            becomes the feature mask of the returned
                            MVPAPattern object. This is useful if it is
                            necessary to reconstruct the location of certain
                            features in the original data space.
                            If this is not required it can be disabled to speed
                            up the selection process.

        Returns a new MVPAPattern object with a view of the original pattern
        array (no copying is performed).
        """
       # no default mask
        new_mask = None
        # reconstruct an updated feature mask if requested
        # only keep the selected non-zero features of the current mask
        if maintain_mask:
            new_mask = self.buildFeatureMaskFromIds( ids )

        return MVPAPattern( self.__patterns[:, ids],
                            self.__regs,
                            self.__origins,
                            mask = new_mask,
                            internal = True )


    def selectFeaturesByMask( self, mask ):
        """ Use a mask array to select features from the current set.

        The mask array shape must match the current feature mask (data shape).
        The final selection mask only contains features that are present in the
        current feature mask AND the selection mask passed to this method.

        Returns a new MVPAPattern object with a view of the original pattern
        array (no copying is performed).
        """
        if not mask.shape == self.__mask.shape:
            raise ValueError, 'Selection mask shape has to match the original' \
                              + ' mask.'

        # always copy to prevent confusion
        mask = mask.copy()

        # final selection mask
        mask *= self.__mask > 0

        # use '>0' to deal with integer masks
        return MVPAPattern( self.__patterns[:, mask[self.__mask>0].ravel()>0],
                            self.__regs,
                            self.__origins,
                            mask = mask,
                            internal = True )


    def buildFeatureMaskFromGroupIds( self, group_ids):
        """ 'group_ids' is a sequence of mask values where each unique mask
        value forms a single group.
        """
        # build boolean filer
        if len(group_ids) > 1:
            # all features in any of the given groups
            filter = \
                numpy.logical_or( *([self.__mask == id for id in group_ids]) )
        else:
            filter = self.__mask == group_ids[0]

        # preserve ROI ids for selected features
        return filter * self.__mask


    def selectFeaturesByGroup( self, group_ids ):
        return self.selectFeaturesByMask(
            self.buildFeatureMaskFromGroupIds( group_ids ) )


    def getFeatureId( self, coord ):
        """ Translate a feature mask coordinate into a feature ID.

        Warning: This method is painfully slow, avoid if possible!
        """
        coord = list(coord)

        featcoords = numpy.transpose(self.__mask.nonzero()).tolist()

        for i, c in enumerate( featcoords ):
            if c == coord:
                return i

        raise ValueError, "There is no used feature at this mask coordinate."


    def getCoordinate( self, feature_id ):
        """ Returns a features coordinate in the original data space
        for a given feature id.
        """
        return self.getFeatureCoordinates()[feature_id]


    def getFeatureCoordinates( self ):
        """ Returns a 2d array where each row contains the coordinate of the
        feature with the corresponding id.
        """
        return numpy.transpose(self.__mask.nonzero())


    def getFeatureMask(self, copy = True):
        """By default returns a copy of the current binary feature mask.

        If 'copy' is set to False a reference to the mask is returned instead.
        This shared mask must not be modified! 
        """
        if copy:
            return self.__mask.copy()
        else:
            return self.__mask


    def getNumberOfPatterns( self ):
        """ Currently available number of patterns.
        """
        return self.pattern.shape[0]


    def getNumberOfFeatures( self ):
        """ Number of features per pattern.
        """
        return self.pattern.shape[1]


    def getPatterns( self ):
        """ Returns the pattern matrix.

        Please note, that the size of the pattern matrix depends on the
        current feature and pattern mask.
        """
        return self.__patterns


    def getRegs( self ):
        """ Returns the regressors vector.

        Please note, that the size of the vector depends on the current
        pattern mask.
        """
        return self.__regs


    def getOrigins( self ):
        """ Returns the origin vector.

        Please note, that the size of the vector depends on the current
        pattern mask.
        """
        return self.__origins


    def getRegLabels( self ):
        """ Returns an array with all unique class labels in the regressors
        vector.
        """

        return numpy.unique( self.reg )


    def getOriginLabels( self ):
        """ Returns an array with all unique labels in the origin vector.
        """

        return numpy.unique( self.origin )


    def getPatternsPerRegLabel( self ):
        """ Returns the number of patterns per regressor label.
        """
        return [ len(self.__patterns[self.__regs == r]) \
                    for r in self.reglabels ]


    # read-only class properties
    pattern =   property( fget=getPatterns )
    reg =       property( fget=getRegs )
    origin =    property( fget=getOrigins )
    npatterns = property( fget=getNumberOfPatterns )
    nfeatures = property( fget=getNumberOfFeatures )
    origshape = property( fget=lambda self: self.__mask.shape )
    reglabels = property( fget=getRegLabels )
    originlabels = property( fget=getOriginLabels )
    patperreg = property( fget=getPatternsPerRegLabel )

