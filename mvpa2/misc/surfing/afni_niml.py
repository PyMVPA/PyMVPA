# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
General AFNI NIML I/O support

Created on Feb 16, 2012

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)

This function reads a NIML file and returns a dict that contains all 
NIML information in a tree-like structure (dicts for which some values
are dicts themselves). Branches are stored in a 'nodes' field.

For specific types of data, consider afni_niml_annot or afni_niml_annot
files which provide easier access to the data.

'''

import re, numpy as np, random, os, time, sys, base64, copy

import mvpa2.misc.surfing.afni_niml_types as types
_RE_FLAGS = re.DOTALL # regular expression matching spans across new lines


_TEXT_ROWSEP = "\n"
_TEXT_COLSEP = " "

_ESCAPE = {'&lt;':'<',
         '&gt;':'>',
         '&quot;':'"',
         '&amp;':'&',
         '&apos;':"'"}

def decode_escape(s):
    for k, v in _ESCAPE.iteritems():
        s = s.replace(k, v)
    return s

def encode_escape(s):
    for k, v in _ESCAPE.iteritems():
        s = s.replace(v, k)
    return s

def _parse_nameheaderbody(s):
    '''parse string in the form <NAME HEADER>BODY</NAME>'''

    e = r'<(?P<name>\w+)\W(?P<header>.*?)>[\s"]*(?P<body>.*?)["\s]*</\1>'
    return re.findall(e, s, _RE_FLAGS)

def _parse_keyvalues(s):
    '''parse K0=V0 K1=V1 ... and return a dict(K0=V0,K1=V1,...)'''

    e = r'\s*(?P<lhs>\w+)\s*=\s*"(?P<rhs>[^"]+)"'
    m = re.findall(e, s, _RE_FLAGS)
    return dict([(k, v) for k, v in m])

def string2rawniml(s):
    nhbs = _parse_nameheaderbody(s) # name, header, body triples
    if not nhbs:
        raise ValueError("Did not understand %s" % s)

    nimls = [] # output array

    for nhb in nhbs:
        name, header, body = nhb # only one element
        niml = _parse_keyvalues(header) # header
        niml['name'] = name # name of the group

        if niml.get('ni_form', None) == 'ni_group':
            niml['nodes'] = string2rawniml(body)
        else:
            datatypes = niml['ni_type']
            niml['vec_typ'] = types.str2codes(datatypes)
            niml['vec_len'] = int(niml['ni_dimen'])
            niml['vec_num'] = len(niml['vec_typ'])
            niml['data'] = _datastring2rawniml(body, niml)

        nimls.append(niml)

    return nimls

def _mixedtypes_datastring2rawniml(s, niml):
    tps = niml['vec_typ']
    ncols = len(tps)
    nrows = niml['vec_len']

    lines = s.strip().split(_TEXT_ROWSEP)
    if len(lines) != nrows:
        raise ValueError("Expected %d rows, but found %d" % (nrows, len(lines)))

    elems = map(lambda x : x.strip().split(_TEXT_COLSEP), lines)
    fs = map(types.code2python_convertor, tps)

    data = []
    for col in xrange(ncols):
        f = fs[col]
        if types.sametype(tps[col], 'String'):
            d = map(f, [elems[r][col] for r in xrange(nrows)])
        else:
            tp = types.code2numpy_type(tps[col])
            niform = niml.get('ni_form', None)
            if not niform is None:
                raise ValueError('Not supported: have ni_form with mixed types')

            d = np.zeros((nrows,), dtype=tp) # allocate one-dimensional array
            for r in xrange(nrows):
                d[r] = f(elems[r][col])

        data.append(d)

    return data


def _datastring2rawniml(s, niml):
    tps = niml['vec_typ']

    onetype = types.findonetype(tps)

    if onetype is None:
        return _mixedtypes_datastring2rawniml(s, niml)

    if [onetype] == types.str2codes('string'):
        return decode_escape(s) # do not string2rawniml

    # numeric, either int or float
    ncols = niml['vec_num']
    nrows = niml['vec_len']
    tp = types.code2numpy_type(onetype)

    niform = niml.get('ni_form', None)

    if not niform or niform == 'text':
        data = np.zeros((nrows, ncols), dtype=tp) # allocate space for data CHECKME was tp=tp
        convertor = types.code2python_convertor(onetype) # string to type convertor 

        vals = s.split(None) # split by whitespace seperator
        if len(vals) != ncols * nrows:
            raise ValueError("unexpected number of elements")

        for i, val in enumerate(vals):
            data[i / ncols, i % ncols] = convertor(val)

    else:
        dtype = np.dtype(tp)
        dtype = types.byteorder_from_niform(niform, dtype)

        if 'base64' in niform:
            s = base64.decodestring(s)
        elif not 'binary' in niform:
            raise ValueError('Illegal niform %s' % niform)

        data_1d = np.fromstring(s, dtype=tp)
        data = np.reshape(data_1d, (nrows, ncols))

    return data

def getnewidcode():
    return ''.join(map(chr, [random.randint(65, 65 + 25) for _ in xrange(24)]))

def setnewidcode(s):
    tp = type(s)
    if tp is list:
        for v in s:
            setnewidcode(v)
    elif tp is dict:
        key = 'self_idcode'
        for k, v in s.iteritems():
            if k == key:
                s[key] = getnewidcode()
            else:
                setnewidcode(v)

def rawniml2string(p, form='text'):
    if type(p) is list:
        return "\n".join(rawniml2string(v, form) for v in p)

    if not form in ['text', 'binary', 'base64']:
        raise ValueError("Illegal form %s" % form)

    q = p.copy() # make a shallow copy


    if 'nodes' in q:
        s_body = rawniml2string(q.pop('nodes'), form) # recursion
    else:
        data = q.pop('data')
        data = types.nimldataassupporteddtype(data) # ensure the data format is supported by NIML

        s_body = _data2string(data, form)

        if form == 'text':
            q.pop('ni_form', None) # defaults to text, remove if already there
        else:
            byteorder = types.data2ni_form(data, form)
            if byteorder:
                q['ni_form'] = byteorder

        # remove some unncessary fields
        for f in ['vec_typ', 'vec_len', 'vec_num']:
            q.pop(f, None)

    s_name = q.pop('name', None)
    s_header = _header2string(q)

    return '<%s\n%s >%s</%s>' % (s_name, s_header, s_body, s_name)

def _data2string(data, form):
    if type(data) is str:
        return '"%s"' % encode_escape(data)
    elif type(data) is np.ndarray:
        if form == 'text':
            f = types.numpy_data2printer(data)
            nrows, ncols = data.shape
            return _TEXT_ROWSEP.join([_TEXT_COLSEP.join([f(data[row, col])
                                                         for col in xrange(ncols)])
                                                         for row in xrange(nrows)])
        elif form == 'binary':
            return str(data.data)
        elif form == 'base64':
            return base64.encodestring(data)
        else:
            raise ValueError("illegal format %s" % format)

    elif type(data) is list:
        # mixed types, each column in its own container
        # always use text output format, even if requested form is binary of base64

        ncols = len(data)
        if ncols == 0:
            return ""
        else:
            nrows = len(data[0])

            # separate formatter functions for each column
            # if list of strings then take first element of the list to get a string formattr
            # else use the entire np array to get a numeric formatter
            fs = [types.numpy_data2printer(d[0] if type(d) is list else d) for d in data]

            return _TEXT_ROWSEP.join([_TEXT_COLSEP.join([fs[col](data[col][row])
                                                         for col in xrange(ncols)])
                                                         for row in xrange(nrows)])

    else:
        raise TypeError("Unknown type %r" % type(data))

def _header2string(p, keyfirst=['dset_type', 'self_idcode', 'filename', 'data_type'], keylast=['ni_form']):
    otherkeys = list(set(p.keys()) - (set(keyfirst) | set(keylast)))

    added = set()
    keyorder = [keyfirst, otherkeys, keylast]
    kvs = []
    for keys in keyorder:
        for k in keys:
            if k in p and not k in added:
                kvs.append((k, p[k]))
                added.add(k)

    rs = map(lambda x : '   %s="%s"' % x, kvs)
    return "\n".join(rs)

def read(fn, itemifsingletonlist=True, postfunction=None):
    f = open(fn)
    s = f.read()
    f.close()

    r = string2rawniml(s)
    if not postfunction is None:
        r = postfunction(r)

    if itemifsingletonlist and type(r) is list and len(r) == 1:
        return r[0]
    else:
        return r

def write(fnout, niml, form='binary', prefunction=None):
    if not prefunction is None:
        niml = prefunction(niml)

    s = rawniml2string(niml, form=form)

    f = open(fnout, 'w')
    f.write(s)
    f.close()

