# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Type definitions for AFNI NIML format
Taken from afni_ni_defs() function in AFNI matlab library
which is based on niml.h

Created on Feb 16, 2012

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)
'''
import numpy as np, sys

'''Currently no support for RGB and RGBA'''

np_types = [np.byte, np.int16, np.int32,
            np.float32, np.float64, np.complex64,
            None, None, str]

np_bytecounts = [1, 2, 4, 4, 8, 16, None, None, None]

python_types = [int, int, int,
                float, float, complex,
                None, None, str]

type_names = ['byte', 'short', 'int',
              'float', 'double', 'complex',
              'rgb', 'rgba', 'String']

type_alias = ['uint8', 'int16', 'int32',
              'float32', 'float64', 'complex64',
              'rgb8', 'rgba8', 'CString']
type_sep = ","


def code2python_convertor(i):
    if i in [0, 1, 2]:
        return int
    if i in [3, 4, 5]:
        return float
    if i in [8]:
        return lambda x: x.strip('"')  # remove quotes
    return None


def numpy_type2bytecount(tp):
    for i, t in enumerate(np_types):
        if t is tp:
            return np_bytecounts[i]
    return None


def numpy_type2name(tp):
    code = numpy_type2code(tp)
    return _one_code2str(code)


def numpy_data_isint(data):
    return type(data) is np.ndarray and np.issubdtype(data.dtype, int)


def numpy_data_isfloat(data):
    return type(data) is np.ndarray and np.issubdtype(data.dtype, float)


def numpy_data_isdouble(data):
    return type(data) is np.ndarray and np.issubdtype(data.dtype, np.double)


def numpy_data_isstring(data):
    return type(data) is np.ndarray and \
           (np.issubdtype(data.dtype, np.str) or data.dtype.kind in 'US')


def numpy_data2printer(data):
    tp = type(data)
    if tp is list:
        return map(numpy_data2printer, data)
    elif tp is str:
        return lambda x: '"%s"' % x
    elif tp == np.ndarray:
        if numpy_data_isint(data):
            return lambda x: '%d' % x
        elif numpy_data_isdouble(data):
            return str
        elif numpy_data_isfloat(data):
            return lambda x: '%f' % x
        elif numpy_data_isstring(data):
            return lambda x: '"%s"' % x

    raise ValueError("Not understood type %r in %r" % (tp, data))


def code2python_type(i):
    if type(i) is list:
        return map(code2python_type, i)
    else:
        return python_types[i]


def nimldataassupporteddtype(data):
    tp = type(data)

    if tp is list:
        return map(nimldataassupporteddtype, data)

    if not type(data) is np.ndarray:
        return data

    tp = data.dtype
    if numpy_data_isfloat(data) and not np.issubdtype(tp, np.float32):
        return np.asarray(data, np.float32)

    if numpy_data_isint(data) and not np.issubdtype(tp, np.int32):
        return np.asarray(data, np.int32)

    return data


def numpy_type2code(tp):
    if type(tp) is list:
        return map(numpy_type2code, tp)
    else:
        for i, t in enumerate(np_types):
            if t == tp:
                return i

        # hack because niml does not support int64
        if tp == np.int64:
            return 2

        # bit of a hack to get string arrays converted properly 
        # XXX should we do this for other types as well?
        if isinstance(tp, np.dtype) and tp.char in ('S', 'a', 'U'):
            return 8

        raise ValueError("Unknown type %r" % tp)


def code2numpy_type(i):
    if type(i) is list:
        return map(code2numpy_type, i)
    else:
        return np_types[i]


def num_codes():
    return len(type_names)


def _one_str2code(name):
    lname = name.lower()
    for lst in [type_names, type_alias]:
        for i, v in enumerate(lst):
            if v.lower() == lname:
                return i
    return None


def _one_code2str(code):
    return type_alias[code]


def sametype(p, q):
    ascode = lambda x: _one_str2code(x) if type(x) is str else x
    pc, qc = ascode(p), ascode(q)

    if pc is None or qc is None:
        raise ValueError("Illegal type %r or %r " % (p, q))

    return pc == qc


def codes2str(codes):
    if not type(codes) is list:
        codes = [codes]
    names = [_one_code2str(code) for code in codes]
    return type_sep.join(names)


def byteorder_from_niform(niform, dtype):
    if not (niform and type(niform) is str):
        return None
    if not type(dtype) is np.dtype:
        raise ValueError("Expected numpy.dtype")

    split = niform.split(".")
    ns = len(split)
    if ns == 1:
        prefix = niform
        byteorder = 'msbfirst'  # the default
    elif ns == 2:
        prefix, byteorder = split
    else:
        raise ValueError('Not understood niform')

    if prefix in ['binary', 'base64']:
        d = dict(lsbfirst='<', msbfirst='>')
        order = d.get(byteorder, None)
        return order and dtype.newbyteorder(order)
    else:
        raise ValueError("Prefix %s not understood" % prefix)

    raise ValueError('Not understood niform')


def data2ni_form(data, form):
    if (type(data) is np.ndarray and
            (numpy_data_isint(data) or numpy_data_isfloat(data))):
        byteorder = data.dtype.byteorder

        if not form in ['binary', 'base64']:
            raise ValueError('illegal form %s' % form)

        if byteorder == '=':  # native order
            byteorder = '<' if sys.byteorder == 'little' else '>'

        if byteorder in '<>':
            return '%s.%s' % (form, 'lsbfirst' if byteorder == '<' else 'msbfirst')
        else:
            raise ValueError("Unrecognized byte order %s" % byteorder)

    return None


def str2codes(names):
    parts = names.lower().split(type_sep)
    codes = []
    for part in parts:
        # support either 'float' or '4*float'
        p = part.split('*')
        np = len(p)

        if np == 1:
            fac = 1  # how many
        elif np == 2:
            fac = int(p[0])
        else:
            raise ValueError("Not understood: %s" % part)

        code = _one_str2code(p[-1])  # last element for type
        codes.extend([code] * fac)

    return codes


def findonetype(tps):
    '''tps is a list of vec_typ'''
    typeorder = str2codes(type_sep.join(['string', 'int', 'float']))

    # find correct type, default to float if not string or int
    for tp in typeorder:
        if all([i == tp for i in tps]):
            return tp

    return None
