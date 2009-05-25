# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrapper around the output of MELODIC (part of FSL)"""

__docformat__ = 'restructuredtext'

import os
import numpy as N

from mvpa.base import externals
if externals.exists('nifti', raiseException=True):
    import nifti


class MelodicResults( object ):
    """Easy access to MELODIC output.

    Only important information is available (important as judged by the
    author).
    """
    def __init__( self, path ):
        """Reads all information from the given MELODIC output path.
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
    relvar_per_ic  = property( fget=lambda self: self.__icstats[:, 0] )
    truevar_per_ic = property( fget=lambda self: self.__icstats[:, 1] )
