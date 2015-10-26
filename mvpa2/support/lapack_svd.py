"""ctypes wrapper for LAPACK svd implementation - DGESVD"""
import numpy

from mvpa2.base import externals
from mvpa2.base.types import as_char

if externals.exists('ctypes', raise_=True):
    from ctypes import cdll, c_char, c_int, c_double, c_void_p, byref

from numpy.linalg import LinAlgError

if externals.exists('liblapack.so'):
    lapacklib = cdll.LoadLibrary('liblapack.so')

__all__ = ['svd']

def svd(a, full_matrices=True, algo='svd', **kwargs):
    """ ctypes wrapper for LAPACK SVD (DGESVD)
    Factorizes the matrix a into two unitary matrices U and Vh and
    an 1d-array s of singular values (real, non-negative) such that
    a == U S Vh  if S is an suitably shaped matrix of zeros whose
    main diagonal is s.

    Parameters
    ----------
    a : array, shape (M, N)
        Matrix to decompose
    full_matrices : boolean [Default is True]
        If true,  U, Vh are shaped  (M,M), (N,N)
        If false, the shapes are    (M,K), (K,N) where K = min(M,N)
    algo : 'svd' or 'sdd'
    Returns
    -------
    U:  array, shape (M,M) or (M,K) depending on full_matrices
    s:  array, shape (K,)
        The singular values, sorted so that s[i] >= s[i+1]. K = min(M, N)
    Vh: array, shape (N,N) or (K,N) depending on full_matrices

    Raises LinAlgError if SVD computation does not converge
    """
    if full_matrices:
        flag='A'
    else:
        flag='S'
    jobu=c_char(as_char(flag))
    jobv=c_char(as_char(flag))
    info=c_int(0)
    x, y = a.shape
    m=c_int(x)
    n=c_int(y)
    lda=c_int(x)
    s=(c_double*min(x,y))()
    ldu=c_int(x)


    if full_matrices:
        ldvt = c_int(y)
        u  = numpy.zeros((x,x),dtype=float)
        vt = numpy.zeros((y,y),dtype=float)
    else:
        ldvt = c_int(min(x,y))
        u  = numpy.zeros((x,min(x,y)),dtype=float)
        vt = numpy.zeros((min(x,y),y),dtype=float)


    if algo == 'svd':
        lwork=c_int(7*max(x,y))
        work = (c_double*7*min(x,y))()
        lapacklib.dgesvd_(byref(jobu), byref(jobv), byref(m), byref(n), 
                a.ctypes.data_as(c_void_p), byref(lda), s, 
                u.ctypes.data_as(c_void_p), byref(ldu), vt.ctypes.data_as(c_void_p), 
                byref(ldvt), work, byref(lwork), byref(info))
    else:
        lwork=c_int(7*min(x,y)+4*min(x,y)*min(x,y))
        work = (c_double*(7*min(x,y)+4*min(x,y)*min(x,y)))()
        iwork = (c_int*8*min(x,y))()
        lapacklib.dgesdd_(byref(jobu), byref(m), byref(n), 
                a.ctypes.data_as(c_void_p), byref(lda), s, 
                u.ctypes.data_as(c_void_p), byref(ldu), vt.ctypes.data_as(c_void_p), 
                byref(ldvt), work, byref(lwork), iwork, byref(info))
        
    if info.value >= 1:
        print "DBSQR did not converge for %i superdiagonals"%(info.value)
    #    raise LinAlgError
    #if info.value <= -1:
    #    print "Interesting!!! \nQuick, go find swaroop"
    if info.value == 0:
        return vt, numpy.frombuffer(s), u
