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

attempt for simple alphasim based on simple 2nd level residuals (a la SPM)
uses SUMA's SurfClust, SurfFWHM, and a couple of other AFNI programs

Note: this method has not been validated properly yet.

NNO Oct 2012
'''

import os, fnmatch, datetime, re, argparse, math

from mvpa2.support.afni import afni_utils as utils


def _fn(config, infix, ext=None):
    '''Returns a file name with a particular infix'''
    if ext is None:
        ext = _ext(config)
    return './%s%s%s' % (config['prefix'], infix, ext)

def _is_surf(config):
    '''Returns True iff we are on the surface'''
    return 'surface_file' in config and config['surface_file']

def _ext(config, for1D=False):
    '''Returns the extension for a file name'''
    if _is_surf(config):
        return '.1D.dset' if for1D else '.niml.dset'
    else:
        fn = config['data_files'][0]
        return ''.join(utils.afni_fileparts(fn)[2:])

def _mask_expr(config):
    '''returns an expression that can be used as an infix in
    running other commands (depending on whether in volume or on surface)'''
    m = config['mask']
    if not m:
        return ''
    else:
        if _is_surf(config):
            return '-b_mask %s' % m
        else:
            return '-mask %s' % m

def compute_fwhm(config):
    # helper function - called by critical_clustersize
    # computes FWHM of residuals of input data and stores in config
    output_dir = c['output_dir']

    is_surf = _is_surf(config)
    ext, ext1D = _ext(config), _ext(config, for1D=True)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    cmds = ['cd "%s"' % output_dir]

    # if surfaces and needs padding, do that first
    pad_to_node = config['pad_to_node']
    if is_surf and pad_to_node:
        data_files = []

        for i, fn in enumerate(c['data_files']):
            fn_pad = 'pad_%d%s' % (i, ext)
            cmds.append("; ConvertDset -overwrite -pad_to_node %d -input %s'[%d]' -prefix ./%s" %
                                (pad_to_node, fn, config['brik_index'], fn_pad))
            data_files.append(fn_pad)
        pad_files = data_files
        brik_index = 0
    else:
        data_files = c['data_files']
        pad_files = []
        brik_index = c['brik_index']

    # bucket data from all participants into a single file
    buck_fn = _fn(config, 'buck')

    cmds.append('; 3dbucket -overwrite -prefix %s' % buck_fn)
    for fn in data_files:
        cmds.append(" %s'[%d]'" % (fn, brik_index))

    # also store as 1D (won't hurt)
    if is_surf:
        buck_fn_1D = _fn(config, 'buck', ext1D)
        cmds.append('; ConvertDset -overwrite -o_1D -prefix %s -input %s' %
                    (buck_fn_1D, buck_fn))
    else:
        buck_fn_1D = buck_fn

    # compute group mean
    mean_fn = _fn(config, 'mean')
    cmds.append('; 3dTstat -overwrite -prefix %s %s' % (mean_fn, buck_fn))

    # compute residuals, and estimate FWHM for each of them
    # store FWHM output in fwhm_fn
    fwhm_fn = os.path.join(output_dir, _fn(config, 'fwhm', '.1D'))
    cmds.append('; echo > "%s"' % fwhm_fn)

    resid_fns = []
    for i in xrange(len(c['data_files'])):
        fn = _fn(config, 'resid_%d' % i)
        cmds.append("; 3dcalc -overwrite -prefix %s -a %s -b %s'[%d]' -expr 'a-b'"
                    % (fn, mean_fn, buck_fn, i))
        msk = _mask_expr(config)
        if is_surf:
            surf_fn = c['surface_file']
            cmds.append("; SurfFWHM %s -input %s -i_fs %s"
                        "| grep ^FWHM  | cut -f2 -d'=' >> '%s'" %
                        (msk, fn, surf_fn, fwhm_fn))
        else:
            cmds.append('; 3dFWHMx %s %s | cut -c18- >> %s' % (msk, fn, fwhm_fn))
        resid_fns.append(fn)

    cmd = ''.join(cmds)
    utils.run_cmds(cmd)

    # read FWHM values and store in config
    with open(fwhm_fn) as f:
        fwhms = f.read().split()

    print fwhms
    print fwhm_fn

    config['all_fwhms'] = fwhms # all FWHMs (for each participant)
    config['fwhm'] = sum(map(float, fwhms)) / len(fwhms) # average FWHM
    config['buck_fn'] = buck_fn
    config['buck_fn_1D'] = buck_fn_1D

    mean_fwhm_fn = os.path.join(output_dir, _fn(config, 'mean_fwhm', '.1D'))
    with open(mean_fwhm_fn, 'w') as f:
        f.write('%.3f\n' % config['fwhm'])

    tmpfns = resid_fns + pad_files + [mean_fn]
    print "TEMP"
    print tmpfns
    _remove_files(config, tmpfns)

def null_clustersize(config):
    # helper function - called by critical_clustersize
    # computes maxmimum cluster size of a single null permutation
    output_dir = config['output_dir']
    tthr = config['tthr']
    fwhm = config['fwhm']
    buck_fn_1D = config['buck_fn_1D']
    msk = _mask_expr(config)
    is_surf = _is_surf(config)
    if is_surf:
        surf_fn = config['surface_file']
    ext, ext1D = _ext(config), _ext(config, for1D=True)

    ns = len(config['data_files'])
    cmds = ['cd "%s"' % output_dir]

    # generate N random data files (N=number of participants)
    # use the output bucket to get datasets with the right size
    null_fns = []
    for i in xrange(ns):
        fn = _fn(config, 'rand_%d' % i, ext1D)
        if is_surf:
            cmds.append("; 1deval -ok_1D_text -a %s'[0]' -expr 'gran(0,1)' > '%s'" % (buck_fn_1D, fn))
        else:
            cmds.append("; 3dcalc -overwrite -prefix %s -a %s'[0]' -expr 'gran(0,1)'" % (fn, buck_fn_1D))
        null_fns.append(fn)

    # bucket random data
    buck_fn = _fn(config, 'rand_buck', ext1D)
    null_fns_list = ' '.join(null_fns)
    if is_surf:
        cmds.append('; 1dcat %s > "%s"' % (null_fns_list, buck_fn))
    else:
        cmds.append('; 3dbucket -overwrite -prefix %s %s' % (buck_fn, null_fns_list))

    # smooth all data at once, using estimated FWHM
    smooth_fn = _fn(config, 'rand_buck_smooth', ext1D)
    if is_surf:
        if config['sigma'] > 0:
            sigma_str = '-sigma %s' % config['sigma']
        else:
            sigma_str = ''
        cmds.append('; SurfSmooth -overwrite %s -met HEAT_07 -i_fs %s -input %s '
                    ' -fwhm %f -output %s %s' % (msk, surf_fn, buck_fn, fwhm, smooth_fn, sigma_str))
    else:
        cmds.append('; 3dBlurInMask -overwrite %s -FWHM %f -prefix %s -input %s' %
                      (msk, fwhm, smooth_fn, buck_fn))

    # run ttest
    if is_surf:
        msk = '' # cannot use mask on surface, but that's fine
               # as it was used in SurfSmooth
    ttest_fn = _fn(config, 'rand_buck_smooth_t', ext1D)
    cmds.append('; 3dttest++ %s -overwrite -prefix %s -setA %s' %
                    (msk, ttest_fn, smooth_fn))


    # extract maximum cluster size (in mm^2 or number of voxels) from output
    # and pipe into size_fn

    size_fn = _fn(config, 'rand_size', '.1D')
    if is_surf:
        postfix = "| grep --invert-match '#' | head -1 | cut -c 18-28"
        cmds.append('; SurfClust -i_fs %s -input %s 1 -rmm -1 '
                    ' -thresh %f -thresh_col 1 %s > "%s"' %
                        (surf_fn, ttest_fn, tthr, postfix, size_fn))
    else:
        postfix = " | grep --invert-match '#' | head -1 | cut -c1-8"
        cmds.append("; 3dclust -quiet -1noneg -1clip %f 0 0 %s'[1]' %s > '%s'" %
                     (tthr, ttest_fn, postfix, size_fn))

    utils.run_cmds(''.join(cmds))

    # read maximum cluster size form size_fn
    sz_str = None
    with open(os.path.join(output_dir, size_fn)) as f:
        sz_str = f.read()

    try:
        sz = float(sz_str)
    except:
        sz = 0. # CHECKME whether this makes sense

    print "Null data: maximum size %f" % sz

    if is_surf:
        smoothing_fn_rec = os.path.join(output_dir, _fn(config, 'rand_buck_smooth', '.1D.dset.1D.smrec'))
        if not os.path.exists(smoothing_fn_rec):
            raise ValueError("Smoothing did not succeed. Please check the error"
                             " messaged. You may have to set sigma manually")
        with open(smoothing_fn_rec) as f:
            s = f.read()

        final_fwhm = float(s.split()[-2])
        ratio = fwhm / final_fwhm
        thr = 0.9
        if ratio < thr or 1. / ratio < thr:
            raise ValueError('FWHM converged to %s but expected %s. Consider '
                             'setting sigma manually' % (final_fwhm, fwhm))

    # clean up - remove all temporary files

    tmpfns = null_fns + [buck_fn, smooth_fn, ttest_fn, size_fn]
    _remove_files(config, tmpfns)

    return sz

def _remove_files(config, list_of_files):
    # removes a list of files, if config allows for it
    # in the case of AFNI volume files, it removes HEAD and BRIK files
    if not config['keep_files']:
        for fn in list_of_files:
            fp = utils.afni_fileparts(fn)

            if fp[2]:
                # AFNI HEAD/BRIK combination
                # ensure we delete all of them
                exts = ['.HEAD', '.BRIK', '.BRIK.gz']
                fn = ''.join(fp[1:3])
            else:
                exts = ['']

            for ext in exts:
                full_fn = os.path.join(config['output_dir'], fn + ext)
                if os.path.exists(full_fn):
                    os.remove(full_fn)


def critical_clustersize(config):
    '''computes the critical cluster sizes
    it does so by calling compute_fwhm and null_clustersize

    config['max_size'] is a list with he maximum cluster size of
    each iteration'''

    compute_fwhm(config)

    # this takes a long time
    niter = config['niter']
    sz = []
    for i in xrange(niter):
        sz.append(null_clustersize(config))
        print "Completed null iteration %d / %d" % (i + 1, niter)

    config['max_size'] = sz

    # store the results in a file
    clsize_fn = _fn(config, 'critical_cluster_size', '.1D')
    with open(os.path.join(config['output_dir'], clsize_fn), 'w') as f:
        f.write('# Critical sizes for tthr=%.3f, fwhm=%.3f, %d files\n' % (
                    config['tthr'], config['fwhm'], len(config['data_files'])))
        for s in sz:
            f.write('%.5f\n' % s)
    return sz

def _critical_size_index(config):
    '''computes the index of the critical cluster size
    (assuming that these sizes are sorted)'''
    pthr = config['pthr']
    nsize = config['niter']

    idx = math.ceil((1. - pthr) * nsize) # index of critical cluster size
    if idx >= nsize or idx == 0:
        raise ValueError("Illegal critical index (p=%s): %s; "
                            "consider increasing --niter" % (pthr, idx))

    return int(idx)


def apply_clustersize(config):
    # applies the critical cluster size to the original data
    #
    # assumes that critical_clustersize(config) has been run

    output_dir = config['output_dir']
    pthr = config['pthr']
    tthr = config['tthr']
    niter = config['niter']
    buck_fn_1D = config['buck_fn_1D']
    is_surf = _is_surf(config)

    if is_surf:
        surf_fn = config['surface_file']

    cmds = ['cd "%s"' % output_dir]

    # run ttest on original data
    infix = 'ttest_t%(tthr)s' % config
    ttest_fn = _fn(config, infix)
    msk = _mask_expr(config)

    # NOTE: for surfaces, apply mask below (SurfClust)
    #       but in volumes, apply it here
    if is_surf:
        cmds.append('; 3dttest++ -ok_1D_text -overwrite -prefix %s -setA %s' % (ttest_fn, buck_fn_1D))
    else:
        cmds.append('; 3dttest++ %s -overwrite -prefix %s -setA %s' % (msk, ttest_fn, buck_fn_1D))

    # sort cluster sizes
    clsize = list(config['max_size'])
    clsize.sort()

    # get critical cluster size
    idx = _critical_size_index(config)
    critical_size = clsize[idx]

    print "critical size %s (p=%s)" % (critical_size, pthr)

    # apply critical size to t-test of original data
    infix += '_clustp%s_%dit' % (pthr, niter)

    if not is_surf:
        # for surfaces the size is included in the filename automatically
        infix += '_%svx' % critical_size

    # set file names
    dset_out = _fn(config, infix)
    log_out = _fn(config, infix, '.txt')

    if is_surf:
        cmds.append('; SurfClust %s -i_fs %s -input %s 1 -rmm -1 '
                    ' -thresh %f -thresh_col 1 -amm2 %f -out_clusterdset -prefix %s > %s' %
                        (msk, surf_fn, ttest_fn, tthr, critical_size, dset_out, log_out))
    else:
        dset_out_msk = _fn(config, infix + '_msk')
        cmds.append("; 3dclust -overwrite -1noneg -1clip %f  "
                    " -prefix %s -savemask %s 0 -%f %s'[1]' > %s" %
                    (tthr, dset_out, dset_out_msk, critical_size, ttest_fn, log_out))

    cmd = "".join(cmds)
    utils.run_cmds(cmd)

def run_all(config):
    '''main function to estimate critical cluster size
    and apply to group t-test result'''
    critical_clustersize(config)
    apply_clustersize(config)


def get_testing_config():
    # for testing only
    in_vol = True
    c = dict(mask='')

    if in_vol:
        d = '/Users/nick/organized/210_smoothness/afnidata/glm/'
        sids = ['%02d' % i for i in xrange(1, 13)]
        fn_pat = 'glm_SUB%s_REML+tlrc.HEAD'
        fns = ['%s/%s' % (d, fn_pat % sid) for sid in sids]
        c['output_dir'] = '/Users/nick/organized/210_smoothness/_tst'
        c['brik_index'] = 2
        c['mask'] = d + '../../all/mask+tlrc.HEAD'
        c['pad_to_node'] = None
    else:


        d = '/Users/nick/organized/212_raiders_fmri/ref'
        sids = ['ab', 'ag', 'aw', 'jt']
        fn = 'cross_ico16-128_mh_100vx_8runs_t3.niml.dset'
        fns = ['%s/%s/%s' % (d, sid, fn) for sid in sids]

        surffn = '%s/%s/ico16_mh.intermediate_al.asc' % (d, sids[0])

        c['output_dir'] = '/Users/nick//tmp/alphasim/'

        c['surface_file'] = surffn
        c['brik_index'] = 0
        c['ext'] = '.niml.dset'
        c['pad_to_node'] = 5124 - 1



    c['data_files'] = fns
    c['tthr'] = 4.4
    c['niter'] = 1000
    c['pthr'] = .05
    c['prefix'] = 'alphasim_'
    c['keep_files'] = False


    return c


def get_options():
    description = '''
    Experimental implementation of alternative AlphaSim for volumes and surfaces.\n

    Currently only supports group analysis with t-test against 0.

    Input paramaters are an uncorrected t-test threshold (tthr)
    and cluster-size corrected p-value (a.k.a. alpha level) (pthr).

    This program takes the following steps:
    (0) Participant's map are t-tested against zero, thresholded by tthr and clustered.
    (1) residuals are computed by subtracting the group mean from each participants' map
    (2) the average smoothness of these residual maps is estimated
    (3) null maps are generated from random gaussian noise that is smoothened
    with the estimated smoothness from step (2), thresholded by tthr, and clustered.
    The maximum cluster size is stored for each null map.
    (4) A cluster from (0) survive pthr if a smaller ratio than pthr of the
    null maps generated in (3) have a maximum cluster size.

    Copyright 2012 Nikolaas N. Oosterhof <nikolaas.oosterhof@unitn.it>
    '''

    epilog = '''This function is *experimental* and may delete files in output_dir or elsewhere.
    As of Oct 2012 it has not been validated properly against other
    approaches to correct for multiple comparisons. Also, it is slow.'''


    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("-d", "--data_files", required=True,
                            help=("Data input files (that are tested against "
                                "0 with a one-sample t-test)"), nargs='+')
    parser.add_argument("-o", "--output_dir", required=True,
                            help="Output directory")
    parser.add_argument("-s", "--surface_file", required=False,
                            help=("Anatomical surface file (.asc); required"
                                    " for surface-based analysis"),
                            default=None)
    parser.add_argument("-t", "--tthr", required=True,
                            help="t-test uncorrected threshold", type=float)
    parser.add_argument("-p", "--pthr", required=False,
                            help="p-value for corrected threshold",
                            type=float, default=.05)
    parser.add_argument("-n", "--niter", required=False,
                            help="Number of iterations", type=int,
                            default=1000)
    parser.add_argument("-i", "--brik_index", required=False,
                            help="Brik index in input files",
                            type=int, default=0)
    parser.add_argument("-P", "--prefix", required=False,
                            help="Prefix for data output",
                            default='alphasim_')
    parser.add_argument("-m", "--mask", required=False,
                            help="Mask dataset", default=None)
    parser.add_argument("-k", "--keep_files", required=False,
                            help="Keep temporary files", default=False,
                            action="store_true")
    parser.add_argument("-N", "--pad_to_node", required=False,
                            help="pad_to_node (for surfaces)",
                            default=0, type=int)
    parser.add_argument('-S', "--sigma", required=False,
                            help=('sigma (smoothing bandwidth) for SurfSmooth. '
                                    'If smoothing of surface '
                                    'data takes a long time, '
                                    'set this value to a value between 1 and '
                                    '1.5 and see how many smoothing '
                                    'iterations are performed. '
                                    'Ideally, the number of '
                                    'smoothing iterations should be between '
                                    '10 and 20'),
                            default=0., type=float)

    args = None
    namespace = parser.parse_args(args)
    return vars(namespace)

def fix_options(config):
    # makes everything an absolute path
    # also verifies a few input parameters
    def f(x, check=True):
        y = os.path.abspath(x)
        if check and not os.path.exists(y):
            raise ValueError("Not found: %s" % x)
        return y

    for i in xrange(len(config['data_files'])):
        full_path=f(config['data_files'][i])

        if _extension_indicates_surface_file(full_path) and \
                                            not _is_surf(config):
            raise ValueError("Input file %d (%s) indicates surface-based "
                                "input but '--surface_file' was not specified."
                                " To use this function for surface-based "
                                "analysis, supply an anatomical surface "
                                "file (preferably the intermediate surface "
                                "[=the node-wise average of the pial and "
                                "white surface], or alternatively, a pial "
                                "or white surface) using the --surface_file "
                                "option." % (i+1, full_path))

        config['data_files'][i] = f(config['data_files'][i])

    config['output_dir'] = f(config['output_dir'], check=False)

    if _is_surf(config):
        config['surface_file'] = f(config['surface_file'])

    if config['mask']:
        config['mask'] = f(config['mask'])

    p = config['pthr']
    if p <= 0 or p >= 1:
        raise ValueError('Require 0 < pthr < 1')
    _critical_size_index(config) # raises error if p too small


def _extension_indicates_surface_file(fn):
    surface_extensions=['.gii','.niml.dset','.1D','.1D.dset']

    return any(fn.endswith(ext) for ext in surface_extensions)



if __name__ == '__main__':
    #c = get_testing_config()
    c = get_options()
    fix_options(c)

    run_all(c)
