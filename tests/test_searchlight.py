### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA searchlight algorithm
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import unittest
import mvpa.searchlight as sl
import mvpa.algorithms
import numpy

class SearchlighTests(unittest.TestCase):

    def testSearchlight(self):
        data = numpy.random.standard_normal(( 100, 3, 6, 6 ))
        reg = numpy.concatenate( ( numpy.repeat( 0, 50 ),
                                   numpy.repeat( 1, 50 ) ) )
        orig = numpy.repeat( range(5), 10 )
        origin = numpy.concatenate( (orig, orig) )
        pattern = mvpa.MVPAPattern( data, reg, origin )
        slight = sl.Searchlight( pattern,
                                 numpy.ones( (3, 6, 6) ),
                                 3.0,
                                 elementsize = (3,3,3),
                                 forcesphere = True,
                                 classifier = mvpa.kNN,
                                 k = 5 )

        # check virgin results
        self.failUnless( (slight.perfmean == 0).all() )
        self.failUnless( (slight.perfvar == 0).all() )
        self.failUnless( (slight.perfmin == 0).all() )
        self.failUnless( (slight.perfmax == 0).all() )
        self.failUnless( (slight.spheresize == 0).all() )

        self.failUnless( slight.ncvfolds == 5 )

        # run searchlight
        slight.run(verbose=True)

        # check that something happened
        self.failIf( (slight.perfmean == 0).all() )
        self.failIf( (slight.perfvar == 0).all() )
        self.failIf( (slight.perfmin == 0).all() )
        self.failIf( (slight.perfmax == 0).all() )
        self.failIf( (slight.spheresize == 0).all() )


def suite():
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':
    unittest.main()

