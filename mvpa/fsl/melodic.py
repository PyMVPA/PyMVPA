### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Wrapper around the output of MELODIC (part of FSL)
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

import numpy as N
import nifti
import os

class MelodicResults( object ):
    """ Easy access to MELODIC output.

    Only important information is available (important as judged by the
    author).
    """
    def __init__( self, path ):
        """ Reads all information from the given MELODIC output path.
        """
        self.__outputpath = path
        self.__icapath = os.path.join( path, 'filtered_func_data.ica' )
        self.__ic = \
            nifti.NiftiImage( os.path.join( self.__icapath,
                                            'melodic_IC' ) )
        self.__funcdata = \
            nifti.NiftiImage( os.path.join( self.__outputpath,
                                            'filtered_func_data' ) )
        self.__tmodes = N.fromfile( os.path.join( self.__icapath,
                                                  'melodic_Tmodes' ),
                                    sep = ' ' ).reshape( self.tr, self.nic )
        self.__smodes = N.fromfile( os.path.join( self.__icapath,
                                                  'melodic_Smodes' ),
                                    sep = ' ' )
        self.__icstats = N.fromfile( os.path.join( self.__icapath,
                                                   'melodic_ICstats' ),
                                     sep = ' ' ).reshape( self.nic, 4 )

    # properties
    path     = property( fget=lambda self: self.__respath )
    ic       = property( fget=lambda self: self.__ic )
    nic      = property( fget=lambda self: self.ic.extent[3] )
    funcdata = property( fget=lambda self: self.__funcdata )
    tr       = property( fget=lambda self: self.funcdata.extent[3] )
    tmodes   = property( fget=lambda self: self.__tmodes )
    smodes   = property( fget=lambda self: self.__smodes )
    icastats = property( fget=lambda self: self.__icstats )
    relvar_per_ic  = property( fget=lambda self: self.__icstats[:,0] )
    truevar_per_ic = property( fget=lambda self: self.__icstats[:,1] )
