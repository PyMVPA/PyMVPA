# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Coordinate transformations"""

import numpy as np

if __debug__:
    from mvpa2.base import debug

class TypeProxy:
    """
    Simple class to convert from and then back to original type
    working with list, tuple, ndarray and having

    XXX Obsolete functionality ??
    """
    def __init__(self, value, toType=np.array):
        if   isinstance(value, list): self.__type = list
        elif isinstance(value, tuple): self.__type = tuple
        elif isinstance(value, np.ndarray): self.__type = np.array
        else:
            raise IndexError("Not understood format of coordinates '%s' for the transformation" % `coord`)

    def __call__(self, value):    return self.__type(value)
#   def __getitem__(self, value): return self.__type(value)


class TransformationBase:
    """
    Basic class to describe a transformation. Pretty much an interface
    """

    def __init__(self, previous=None):
        self.previous = previous

    def __getitem__(self, icoord):
        """
        Obtain coordinates, apply the transformation and spit out in the same
        format (list, tuple, numpy.array)
        """

        # remember original type
        #speed origType = TypeProxy(coord)

        # just in case it is not an ndarray, and to provide a copy to manipulate with
        coord = np.array(icoord)

        # apply previous transformation if such defined
        if self.previous:
            # if __debug__: debug('ATL__', "Applying previous transformation on `%s`" % `coord`)
            coord = self.previous[coord]

        #speed if __debug__: debug('ATL__', "Applying main transformation on `%s`" % `coord`)
        # apply transformation
        coord_out = self.apply(coord)
        #speed if __debug__: debug('ATL__', "Applied and got `%s`" % `coord_out`)

        #speed return origType(coord_out)
        return coord_out

    def __call__(self, coord):
        return self[coord]

    def apply(self, coord):
        return coord


class SpaceTransformation(TransformationBase):
    """
    To perform transformation from Voxel into Real Space.
    Simple one -- would subtract the origin and multiply by voxelSize.
    if to_real_space is True then on call/getitem converts to RealSpace
    """
    def __init__(self, voxelSize=None, origin=None, to_real_space=True,
                 *args, **kwargs):

        TransformationBase.__init__(self, *args, **kwargs)

        if voxelSize is not None: self.voxelSize = np.asarray(voxelSize)
        else: self.voxelSize = 1

        if origin is not None: self.origin = np.asarray(origin)
        else: self.origin = 0

        if to_real_space:
            self.apply = self.to_real_space
        else:
            self.apply = self.to_voxel_space

    ##REF: Name was automagically refactored
    def to_real_space(self, coord):
        #speed if not self.origin is None:
        coord -= self.origin
        #speed if not self.voxelSize is None:
        coord *= self.voxelSize
        return coord

    ##REF: Name was automagically refactored
    def to_voxel_space(self, coord):
        #speed if not self.voxelSize is None:
        coord /= self.voxelSize
        #speed if not self.origin is None:
        coord += self.origin
        return map(lambda x:int(round(x)), coord)


class Linear(TransformationBase):
    """
    Simple linear transformation defined by a matrix
    """
    def __init__(self, transf=np.eye(4), **kwargs):
        transf = np.asarray(transf)  # assure that we have arrays not matrices
        prev = kwargs.get('previous', None)
        if prev is not None and isinstance(prev, Linear):
            if prev.N == transf.shape[0] -1:
                if __debug__: debug('ATL__', "Colliding 2 linear transformations into 1")
                transf = np.dot(transf, prev.M)
                # reassign previous transformation to the current one
                kwargs['previous'] = prev.previous
        TransformationBase.__init__(self, **kwargs)
        self.M = transf
        self.N = self.M.shape[0] - 1

    def apply(self, coord):
        #speed if len(coord) != self.__N:
        #speed  raise ValueError("Transformation operates on %dD coordinates" \
        #speed                   % self.__N )
        #speed if __debug__: debug('ATL__', "Applying linear coord transformation + %s" % self.__M)
        # Might better come up with a linear transformation
        coord_ = np.r_[coord, [1.0]]
        result = np.dot(self.M, coord_)
        return result[0:-1]


class MNI2Tal_MatthewBrett(TransformationBase):
    """
    Transformation to bring MNI coordinates into MNI space

    Apparently it is due to Matthew Brett
    http://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach
    """

    _UPPER = np.array([ [0.9900, 0, 0, 0 ],
                        [0, 0.9688, 0.0460, 0 ],
                        [0,-0.0485, 0.9189, 0 ],
                        [0, 0, 0, 1.0000] ] )
    _LOWER = np.array( [ [0.9900, 0, 0, 0 ],
                         [0, 0.9688, 0.0420, 0 ],
                         [0,-0.0485, 0.8390, 0 ],
                         [0, 0, 0, 1.0000] ] )

    def __init__(self, *args, **kwargs):
        TransformationBase.__init__(self, *args, **kwargs)
        self.__upper = Linear(self._UPPER)
        self.__lower = Linear(self._LOWER)

    def apply(self, coord):
        return {True: self.__upper,
                False: self.__lower}[coord[2]>=0][coord]


class Tal2MNI_MatthewBrett(TransformationBase):
    """
    Inverse of MNI2Tal_MatthewBrett

    """

    def __init__(self, *args, **kwargs):
        TransformationBase.__init__(self, *args, **kwargs)
        self.__upper = Linear(np.linalg.inv(MNI2Tal_MatthewBrett._UPPER))
        self.__lower = Linear(np.linalg.inv(MNI2Tal_MatthewBrett._LOWER))

    def apply(self, coord):

        return {True: self.__upper,
                False: self.__lower}[coord[2]>=0][coord]

def mni_to_tal_meyer_lindenberg98 (*args, **kwargs):
    """
    Due to Andreas Meyer-Lindenberg
    Taken from
    http://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach
    """

    return Linear( np.array([
        [    0.88,   0,  0,  -0.8],
        [    0,   0.97,  0,  -3.32],
        [    0,   0.05,  0.88,   -0.44],
        [    0.00000,   0.00000,   0.00000,   1.00000] ]), *args, **kwargs )


def mni_to_tal_yohflirt (*args, **kwargs):
    """Transformations obtained using flirt from Talairach to Standard

    Transformations were obtained by registration of
    grey/white matter image from talairach atlas to FSL's standard
    volume. Following sequence of commands was used:

    fslroi /usr/share/rumba/atlases/data/talairach_atlas.nii.gz talairach_graywhite.nii.gz 3 1
    flirt -in talairach_graywhite.nii.gz \
     -ref /usr/apps/fsl.4.1/data/standard/MNI152_T1_1mm_brain.nii.gz \
     -out talairach2mni.nii.gz -omat talairach2mni.mat \
     -searchrx -20 20 -searchry -20 20 -searchrz -20 20 -coarsesearch 10 -finesearch 6 -v
    flirt -datatype float -in talairach_graywhite.nii.gz -init talairach2mni.mat \
     -ref /usr/apps/fsl.4.1/data/standard/MNI152_T1_1mm_brain.nii.gz \
     -out talairach2mni_fine1.nii.gz -omat talairach2mni_fine1.mat \
     -searchrx -10 10 -searchry -10 10 -searchrz -10 10 -coarsesearch 5 -finesearch 1 -v
    convert_xfm -inverse -omat mni2talairach.mat talairach2mni_fine1.mat
    """
    return Linear(
        t=np.array([
        [ 1.00448,  -0.00629625,  0.00741359,  0.70565,  ],
        [ 0.0130797,  0.978238,  0.0731315,  -3.8354,  ],
        [ 0.000248407,  -0.0374777,  0.838311,  18.6202,  ],
        [ 0,  0,  0,  1,  ],
        ])
                   , *args, **kwargs )


def tal_to_mni_yohflirt (*args, **kwargs):
    """See mni_to_tal_yohflirt doc
    """
    return Linear( np.array([
        [    1.00452,    0.00441281,  -0.011011,  -0.943886],
        [   -0.0141149,  1.00867,     -0.169177,  14.7016],
        [    0.00250222, 0.0920984,    1.18656,  -33.922],
        [    0.00000,   0.00000,   0.00000,   1.00000] ]), *args, **kwargs )



def mni_to_tal_lancaster07_fsl (*args, **kwargs):
    return Linear( np.array([
        [  0.9464, 0.0034, -0.0026, -1.0680],
        [ -0.0083, 0.9479, -0.0580, -1.0239],
        [  0.0053, 0.0617,  0.9010,  3.1883],
        [  0.0000, 0.0000,  0.0000,  1.0000] ]), *args, **kwargs )


def tal_to_mni_lancaster07_fsl (*args, **kwargs):
    return Linear( np.array([
        [ 1.056585, -0.003972,  0.002793,  1.115461],
        [ 0.008834,  1.050528,  0.067651,  0.869379],
        [-0.00682 , -0.071916,  1.105229, -3.60472 ],
        [ 0.      ,  0.      ,  0.      ,  1.      ]]), *args, **kwargs )


def mni_to_tal_lancaster07pooled (*args, **kwargs):
    return Linear( np.array([
        [    0.93570,   0.00290,  -0.00720,  -1.04230],
        [   -0.00650,   0.93960,  -0.07260,  -1.39400],
        [    0.01030,   0.07520,   0.89670,   3.64750],
        [    0.00000,   0.00000,   0.00000,   1.00000] ]), *args, **kwargs )


def tal_to_mni_lancaster07pooled (*args, **kwargs):
    return Linear( np.array([
        [  1.06860,  -0.00396,   0.00826,   1.07816],
        [  0.00640,   1.05741,   0.08566,   1.16824],
        [ -0.01281,  -0.08863,   1.10792,  -4.17805],
        [  0.00000,   0.00000,   0.00000,   1.00000] ]), *args, **kwargs )


if __name__ == '__main__':
    #t = Tal2Mni
    tl = tal_to_mni_lancaster07_fsl()
    tli = mni_to_tal_lancaster07_fsl()
    tml = mni_to_tal_meyer_lindenberg98()
    #print t[1,3,2]
    print tl[(1,3,2)]
    print tli[[1,3,2]]
    print tml[[1,3,2]]
    t = MNI2Tal_MatthewBrett()([10, 12, 14])
    print t, Tal2MNI_MatthewBrett()(t)
#   print t[(1,3,2,2)]
