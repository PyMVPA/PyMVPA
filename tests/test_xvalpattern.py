### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA pattern handling
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

import mvpa
import mvpa.xvalpattern as xvalpattern
import unittest
import numpy as np


class CrossValidationTests(unittest.TestCase):

    def setUp(self):
        self.data = \
            mvpa.MVPAPattern(np.random.normal(size=(100,10)),
            [ i%4 for i in range(100) ],
            [ i/10 for i in range(100) ] )



    def testSimplestCVPatGen(self):
        # create the generator
        cvpg = xvalpattern.CrossvalPatternGenerator(self.data)

        # now get the xval pattern sets One-Fold CV)
        xvpat = [ (train, test) for (train,trs,test,tes) in cvpg(1) ]

        self.failUnless( len(xvpat) == 10 )

        for i,p in enumerate(xvpat):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].npatterns == 90 )
            self.failUnless( p[1].npatterns == 10 )
            self.failUnless( p[1].origin[0] == i )


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    unittest.main()

