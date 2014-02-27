#!/usr/bin/python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Wrapper for 3dSetupGrpInCorr for surface-based dataets in SUMA (in NIML format)
also makes input dataset 'full' (i.e. so that all nodes have a value)

Requires by default only the input datasets from individual participants, as long
as they are named with something like "ico32_lh" in the name. Otherwise it's 
required to provide ico_ld as well to make the datasets 'full'

This code is *experimental*

NNO May 2012
'''

import numpy as np
import os
import re
import argparse

from mvpa2.support.nibabel import afni_niml_dset
from mvpa2.support.afni import afni_utils as utils


def afni_niml_zscore_data(dset, axis=1):
    data = dset['data']
    data[np.isnan(data)] = 0
    sh = data.shape

    # mean and standard deviation
    m = np.mean(data, axis)
    sd = np.std(data, axis)

    mvec = np.reshape(m, (-1, 1))
    sdvec = np.reshape(sd, (-1, 1))

    dataz = (data - mvec) / sdvec
    dataz[np.isnan(dataz)] = 0

    dset['data'] = dataz

    return dset

def _smart_filename_decode(fn, w):
    if type(fn) is list:
        return _list_same_element([_smart_filename_decode(f, w) for f in fn])

    [_, fn] = os.path.split(fn)

    # patterns that might be used as postfix. Try one by one, return if one matches

    if w == 'postfix':
        pats = [r'.*(_ico[\d]+_[lr]h_\d+vx).*',
              r'.*(_ico[\d]+_[lr]h_\d+mm).*',
              r'.*(_ico[\d]+_[lr]h_).*']
        postproc = lambda x:x
    elif w == 'pad_to_ico_ld':
        pats = ['_ico(\d+)_']
        postproc = int
    else:
        raise ValueError("illegal tag %r" % w)

    for pat in pats:
        r = re.findall(pat, fn)
        if len(r) == 1:
          return postproc(r[0].strip("_"))
    return None

def _list_same_element(xs):
    # if a list only contains one element, return that element; None otherwise
    if len(xs) == 0:
        raise ValueError("No elements?")

    return xs[0] if [xs[0]] * len(xs) == xs else None



def afni_niml_zscore_makefull(fnin, fnout, pad_to_ico_ld=None, pad_to_node=None):
    dset = afni_niml_dset.read(fnin)

    dset_z = afni_niml_zscore_data(dset)

    if pad_to_ico_ld or pad_to_node:
        dset_z_full = afni_niml_dset.sparse2full(dset_z, pad_to_ico_ld=pad_to_ico_ld, pad_to_node=pad_to_node)
    else:
        dset_z_full = dset_z

    afni_niml_dset.write(fnout, dset_z_full)
    return fnout

def afni_niml_zscore_makefull_wizard(cfg):
    # ensure we have access to the AFNI binary
    instacorrbin = '3dSetupGroupInCorr'
    if utils.which(instacorrbin) is None:
        raise ValueError("could not locate binary %r" % instacorrbin)

    # get configuration values
    fns = cfg['filenames']
    prefix = cfg['prefix']

    group_prefix = cfg['group_prefix']
    group_postfix = cfg.get('grouppostfix', None)
    if group_postfix is None:
        group_postfix = _smart_filename_decode(fns, 'postfix')
        if not group_postfix is None:
            print "Using automatic postfix %s" % group_postfix

    overwrite = cfg['overwrite']

    # figure out to where to pad to node to
    pad_to_node = cfg.get('pad_to_node', None)

    if pad_to_node is None:
        pad_to_ico_ld = cfg.get('pad_to_ico_ld', None)
        if pad_to_ico_ld is None:
            pad_to_ico_ld = _smart_filename_decode(fns, 'pad_to_ico_ld')
            if not pad_to_ico_ld is None:
                pad_to_node = pad_to_ico_ld * pad_to_ico_ld * 10 + 2
                print "Using automatic pad_to_ico_ld=%r, pad_to_node=%r" % (pad_to_ico_ld, pad_to_node)

    if pad_to_node:
        pad_to_node = int(pad_to_node)

    # process each of the input files            
    fnouts = []
    for fn in fns:
        [pth, nm] = os.path.split(fn)

        fnout = os.path.join(pth, prefix + nm)

        if os.path.exists(fnout) and not overwrite:
            print("Output file %s already exists, skipping (use '--overwrite' to override)" % fnout)
        else:
            afni_niml_zscore_makefull(fn, fnout, pad_to_node=pad_to_node)
            print "Converted %s -> %s (in %s)" % (nm, prefix + nm, pth)

        fnouts.append(fnout)

    [pth, nm] = os.path.split(group_prefix)

    ext = '.niml.dset'

    if ext.endswith(ext):
        ext = ext[:(len(nm) - len(ext))]

    fullprefix = nm if group_postfix is None else '%s%s' % (nm, group_postfix)
    fullname = os.path.join(pth, fullprefix)



    cmds = ['cd "%s"; ' % pth]

    groupfnsout = ['%s.grpincorr.%s' % (fullname, ext) for ext in ['niml', 'data']]

    if any(map(os.path.exists, groupfnsout)):
        if overwrite:
            cmds.extend('rm %s;' % fn for fn in groupfnsout)
        else:
            print("Some or all of output files (%s) already exists (use '--overwrite' to override)" % (" ".join(groupfnsout)))

    cmds.append('%s -prefix ./%s' % (instacorrbin, fullprefix))
    cmds.extend(' %s' % fn for fn in fnouts)


    cmd = "".join(cmds)

    utils.run_cmds(cmd)

    msg = ("\n\nTo view the results in SUMA:\n"
    "- open a terminal window\n"
    "- run 'cd %s; 3dGroupInCorr -setA %s.grpincorr.niml -suma'\n"
    "- open a second terminal window\n"
    "- cd to a directory with surfaces, run 'suma -spec SPECFILE.spec -niml '\n"
    "- in SUMA, select a node while holding down ctrl+shift" %
    (pth, fullprefix))

    print msg



def _full_path(fn):
    if type(fn) is list:
        return map(_full_path, fn)

    return os.path.abspath(fn)

def _fns():
    d = '/Users/nick/organized/_tmp/andyak12/groupdata/'
    s = "ab00_ico32_lh_100vx.niml.dset ad01_ico32_lh_100vx.niml.dset as00_ico32_lh_100vx.niml.dset er00_ico32_lh_100vx.niml.dset hb00_ico32_lh_100vx.niml.dset hm00_ico32_lh_100vx.niml.dset ls00_ico32_lh_100vx.niml.dset mg00_ico32_lh_100vx.niml.dset ok00_ico32_lh_100vx.niml.dset pc01_ico32_lh_100vx.niml.dset pk00_ico32_lh_100vx.niml.dset vm00_ico32_lh_100vx.niml.dset"

    return [d + i for i in s.split(" ")]

def _get_options():
    yesno = ["yes", "no"]
    parser = argparse.ArgumentParser('Pre-processing for group-based connectivity/similarity analysis using AFNI and SUMA\nNikolaas N. Oosterhof Jan 2012')
    parser.add_argument("-i", "--input", required=False, nargs='*')
    parser.add_argument("more_input", nargs='*')
    parser.add_argument('-I', "--pad_to_ico_ld", required=False, default=None)
    parser.add_argument('-N', "--pad_to_node", required=False, default=None)
    parser.add_argument('-p', '--prefix', default='z_full_')
    parser.add_argument('-g', '--group_prefix', default='ALL_')
    parser.add_argument('-G', '--group_postfix', default=None)
    parser.add_argument("-o", "--overwrite", required=False, action='store_true', help="Overwrite existing files")

    namespace = parser.parse_args(None)
    return vars(namespace)

def _augment_config(cfg):
    inputfns = cfg.get('input', []) or []
    inputfns.extend(cfg.get('more_input', []))

    if len(inputfns) == 0:
        raise ValueError('no input files?')

    cfg['filenames'] = _full_path(inputfns)
    cfg['group_prefix'] = _full_path(cfg['group_prefix'])


if __name__ == '__main__':
    cfg = _get_options()
    _augment_config(cfg)
    afni_niml_zscore_makefull_wizard(cfg)
