# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
GIFTI surface functions (wrapper) using nibabel.gifti
'''
from mvpa2.base import externals

if externals.exists("nibabel", raise_=True):
    from nibabel.gifti import gifti, giftiio

import numpy as np, os, datetime, re

from mvpa2.support.nibabel import surf
import io


def _get_single_array(g, intent):
        ar = g.getArraysFromIntent(intent)
        n = len(ar)
        if n != 1:
            len_str = 'no' if n == 0 else '%d' % n
            raise ValueError('Found %s arrays matching %s, expected 1' %
                                (len_str, intent))
        return ar[0]

def read(fn):
    '''Reads a GIFTI surface file

    Parameters
    ----------
    fn: str
        Filename

    Returns
    -------
    surf_: surf.Surface
        Surface

    Notes
    -----
    Any meta-information stored in the GIFTI file is not present in surf_.
    '''

    g = giftiio.read(fn)

    vertices = _get_single_array(g, 'NIFTI_INTENT_POINTSET').data
    faces = _get_single_array(g, 'NIFTI_INTENT_TRIANGLE').data

    return surf.Surface(vertices, faces)


def filename2vertices_faces_metadata(fn):
    '''Attempts to get meta data based on the filename

    Parameters
    ----------
    fn: str
        Filename

    Returns
    -------
    meta: tuple
        Tuple with two gifti.GiftiMetaData objects for vertices
        and faces. If the filename contains exactly one of 'lh', 'rh', or
        'mh' then it is assumed to be of left, right or merged hemispheres.
        If the filename contains exactly one of 'pial,'smoothwm',
        'intermediate',''inflated','sphere','flat', then the geometric
        type is set
    '''
    _, fn = os.path.split(fn)

    vertex_map = dict(AnatomicalStructurePrimary=dict(
                                lh='CortexLeft',
                                rh='CortexRight',
                                mh='CortexRightLeft'),
                      AnatomicalStructureSecondary=dict(
                                pial='Pial',
                                smoothwm='GrayWhite',
                                intermediate='MidThickness'),
                      GeometricType=dict(
                                pial='Anatomical',
                                smoothwm='Anatomical',
                                intermediate='Anatomical',
                                inflated='Inflated',
                                sphere='Spherical',
                                flat='Flat'))

    def just_one(dict_, fn=fn):
        vs = [v for k, v in dict_.iteritems() if k in fn]
        return vs[0] if len(vs) == 1 else None

    v_meta = [gifti.GiftiNVPairs('Name', fn)]

    for key, dict_ in vertex_map.iteritems():
        v = just_one(dict_)
        if not v is None:
            v_meta.append(gifti.GiftiNVPairs(key, v))


    f_meta = [gifti.GiftiNVPairs('Name', fn)]
    # XXX maybe also closed or open topology? that's a bit tricky though

    v = gifti.GiftiMetaData()
    v.data.extend(v_meta)

    f = gifti.GiftiMetaData()
    f.data.extend(f_meta)

    return v, f

def to_gifti_image(s, add_indices=False, swap_LPI_RAI=False):
    '''
    Converts a surface to nibabel's gifti format.

    Parameters
    ----------
    s: surf
        Input surface
    add_indices: True or False (default: False)
        if True then indices of the nodes are added.
        Note: caret may not be able to read these
    swap_LPI_RAI: True or False (default: False)
        If True then the diagonal elements of the xform matrix
        are set to [-1,-1,1,1], otherwise to [1,1,1,1].


    Returns
    -------
    img: gifti.GiftiImage
        Surface representated as GiftiImage
    '''

    vertices = gifti.GiftiDataArray(np.asarray(s.vertices, np.float32))
    vertices.intent = gifti.intent_codes.field1['pointset']
    vertices.datatype = 16 # this is what gifti likes

    if add_indices:
        nvertices = s.nvertices
        indices = gifti.GiftiDataArray(np.asarray(np.arange(nvertices), np.int32))
        indices.datatype = 8 # this is what gifti likes
        indices.coordsys = None # otherwise SUMA might complain
        indices.intent = gifti.intent_codes.field1['node index']


    faces = gifti.GiftiDataArray(np.asarray(s.faces, np.int32))
    faces.intent = gifti.intent_codes.field1['triangle']
    faces.datatype = 8 # this is what gifti likes
    faces.coordsys = None # otherwise SUMA might complain

    # set some fields common to faces and vertices
    for arr in (vertices, faces) + ((indices,) if add_indices else ()):
        arr.ind_ord = 1
        arr.encoding = 3
        arr.endian = 'LittleEndian' # XXX this does not work (see below)
        arr.dims = list(arr.data.shape)
        arr.num_dim = len(arr.dims)

    # make the image
    meta = gifti.GiftiMetaData()
    labeltable = gifti.GiftiLabelTable()

    img = gifti.GiftiImage(meta=meta, labeltable=labeltable)

    if swap_LPI_RAI:
        xform = np.asarray(vertices.coordsys.xform)
        xform[0, 0] = -1
        xform[1, 1] = -1
        vertices.coordsys.xform = xform

    if add_indices:
        img.add_gifti_data_array(indices)
    img.add_gifti_data_array(vertices)
    img.add_gifti_data_array(faces)

    return img


def to_xml(img, meta_fn_hint=None):
    '''Converts to XML

    Parameters
    ----------
    img: gifti.GiftiImage or surf
        Input surface
    meta_fn_hint: str or None
        If not None, it should be a string (possibly a filename that
        describes what kind of surface this is.
        See filename2vertices_faces_metadata.

    Returns
    -------
    xml: bytearray
        Representation of input surface in XML format
    '''

    if isinstance(img, surf.Surface):
        img = to_gifti_image(img)

    if not meta_fn_hint is None:
        vertices = _get_single_array(img, 'pointset')
        faces = _get_single_array(img, 'triangle')
        vertices.meta, faces.meta = \
                        filename2vertices_faces_metadata(meta_fn_hint)

    # XXX FIXME from here on it's a bit of a hack
    # The to_xml() method adds newlines in <DATA>...</DATA> segments
    # and also writes GIFTI_ENDIAN_LITTLE instead of LittleEndian.
    # For now we just replace these offending parts
    # TODO: report issue to nibabel developers

    xml = img.to_xml().encode('utf-8')

    # split by data segements. Odd elements are data, even are surroudning
    sps = re.split(b'([<]Data[>][^<]*?[<][/]Data[>])', xml, re.DOTALL)

    # fix the LittleEndian issue for even segments and newline for odd ones
    fix_odd_even = lambda x, i: x.replace(b'\n', b'') \
                                if i % 2 == 1 \
                                else x.replace(b'Endian="GIFTI_ENDIAN_LITTLE"',
                                               b'Endian="LittleEndian"')

    xml_fixed = b''.join(fix_odd_even(sp, i) for i, sp in enumerate(sps))

    return xml_fixed

def write(fn, s, overwrite=True):
    '''Writes a GIFTI surface file

    Parameters
    ----------
    fn: str
        Filename
    s: surf.Surface
        Surface
    overwrite: bool (default: False)
        If set to False an error is raised if the file exists
    '''

    if not overwrite and os.path.exists(fn):
        raise ValueError("Already exists: %s" % fn)

    EXT = '.surf.gii'
    if not fn.endswith(EXT):
        raise ValueError("Filename %s does not end with required"
                         " extension %s" % (fn, EXT))


    xml = to_xml(s, fn)

    with io.FileIO(fn, 'wb') as f:
        n = f.write(xml)
    if n != len(xml):
        raise ValueError("Not all bytes written to %s" % fn)
