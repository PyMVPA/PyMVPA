# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
This is a library for anatomical preprocessing for surface-based voxel
selection. Typically it is used by a wrapper function prep_afni_surf.

%s

Created on Jan 29, 2012

@author: nick
'''

__usage_doc__ = """
It provides functionality to:

- convert FreeSurfer surfaces to AFNI/SUMA format (using SUMA_Make_Spec_FS).

- resample surfaces to standard topology (using MapIcosahedron) at various
resolutions.

- generate additional surfaces by averaging existing ones.

- coregistration of freesurfer output to AFNI/SUMA anatomical or functional
volume (using align_epi_anat.py).

- run @AddEdge to visualize coregistration.

- merge left (lh) and right (rh) hemispheres into single files (mh).

- generate various views of left+right inflated surfaces.

- generate SUMA specification files, and see_suma* shell scripts.

This script assumes a processing pipeline with FreeSurfer for surface
reconstruction, and AFNI/SUMA for coregistration and visualization.
Specifically it is assumed that surfaces have been reconstructed
using FreeSurfer's recon-all. More details can be found in the
documentation available at http://surfing.sourceforge.net

If EPIs from multiple sessions are aligned, this script should use
different directories for refdir for each session, otherwise naming
conflicts may occur.

This function does not resample or transform any functional data. Instead,
surfaces are transformed to be in alignment with ANATVOL or EPIVOL.

For typical usage it requires three arguments:
(1) "-e epi_filename"  or  "-a anat_filename"
(2) "-d freesurfer/directory/surf"
(3) "-r outputdir"'
(*) "-T" [if "epi_filename" / "anat_filename" is in template (MNI/Tal) space"

Notes:

- Please check alignment visually.

- Nifti (.nii or .nii.gz) files are supported, but AFNI may not be able
to read the {S,Q}-form information properly. If functional data was
preprocessed with a program other than AFNI, please check alignment
visually with another program than AFNI, such as MRIcron.

"""

__doc__ %= __usage_doc__


from mvpa2.support.nibabel import surf, afni_suma_spec
from mvpa2.support.afni import afni_utils as utils


import os
import fnmatch
import datetime
import re
import argparse
import sys

_VERSION = "0.1"
__all__ = ['run_afni_anat_preproc']

def getdefaults():
    '''set up default parameters - for testing now'''
    d = { #  options that must be set properly

       # usually these defaults are fine
       'overwrite':False, # delete directories and files before running commands - BE CAREFUL with this option
       'hemi':'l+r', # hemispheres to process, usually l+r
       'mi_icopat':'ico%d_', # pattern for icosehedron output; placeholder is for ld
       'expvol_ss':True,
       'verbose':True,
       'al2expsuffix':'_al2exp',
       'sssuffix':'_ss',
       'alsuffix':'_al',
       'hemimappingsuffix':'hemi_correspondence.1D',
       'outvol_space':None, # output space, one of 'MNI', 'TLRC', 'MNI_ANAT', None, 'ORIG'
       'outvol_ext': None, # '+tlrc','+orig','.nii','.nii.gz'
       'outvol_fullext':None
       }

    return d

def format2extension(fmt):
    if type(fmt) is dict:
        fmt = fmt['surfformat'] # config
    return dict(ascii='.asc', gifti='.surf.gii')[fmt]

def format2type(fmt):
    if type(fmt) is dict:
        fmt = fmt['surfformat'] # config
    return dict(ascii='fs', gifti='gii')[fmt]

def format2spectype(fmt):
    if type(fmt) is dict:
        fmt = fmt['surfformat'] # config
    return dict(ascii='FreeSurfer', gifti='GIFTI')[fmt]

def checkconfig(config):
    # for now, assume input is valid

    pass

def augmentconfig(c):
    '''add more configuration values that are *derived* from the current configuration,
    and also checks whether some options are set properly'''

    # ensure surfdir is set properly.
    # normally it should end in 'surf'

    # try to be smart and deduce subject id from surfdir (if not set explicitly)
    surfdir = c['surfdir']
    if surfdir:
        surfdir = os.path.abspath(surfdir)
        parent, nm = os.path.split(surfdir)

        if nm != 'surf':
            jn = os.path.join(surfdir, 'surf')
            if os.path.exists(jn):
                surfdir = jn
        elif nm == 'SUMA':
            surfdir = parent

        if not (os.path.exists(surfdir) or os.path.split(surfdir)[1] == 'surf'):
            print('Warning: surface directory %s not does exist or does not end in "surf"' % surfdir)
            surfdir = None

        c['surfdir'] = surfdir

        # set SUMADIR as well
        c['sumadir'] = '%(surfdir)s/SUMA/' % c

    # derive subject id from surfdir

    sid = os.path.split(os.path.split(surfdir)[0])[1] if surfdir else None

    if c.get('sid') is None:
        c['sid'] = sid

    if c['sid'] is None:
        print"Warning: no subject id specified"


    c['prefix_sv2anat'] = 'SurfVol2anat'

    # update steps
    if c.get('steps', 'all') == 'all':
        c['steps'] = 'toafni+mapico+moresurfs+skullstrip+align+makespec+makespecboth+makesurfmasks'



    hasanatvol = 'anatvol' in c and c['anatvol']
    hasepivol = 'epivol' in c and c['epivol']
    hasexpvol = 'expvol' in c and c['expvol']
    hasisepi = 'isepi' in c

    if hasexpvol:
        if hasanatvol or hasepivol:
            raise Exception("expvol specified, but also anatvol or epivol - illegal!")
        if not hasisepi:
            raise Exception("not specified whether expvol is EPI (yes) or anat (no)")

    else:
        if hasanatvol:
            if 'epivol' in c and c['epivol']:
                raise Exception("Cannot have both anatvol and epivol")
            else:
                c['expvol'] = c['anatvol']
                c['isepi'] = False
                del(c['anatvol'])
        else:
            if hasepivol:
                c['expvol'] = c['epivol']
                c['isepi'] = True
                del(c['epivol'])
            else:
                print("Warning: no anatomical or functional experimental voume defined")

    def yesno2bool(d, k): # dict, key
        '''Assuming d[k] contains the word 'yes' or 'no', makes d[k] a boolean
        (True=='yes',False=='no'); otherwise an exception is thrown. The dict is updated'''
        val = d[k]
        if val is None:
            b = False
        elif type(val) == bool:
            b = val
        else:
            v = val.lower()
            if v == 'yes':
                b = True
            elif v == 'no':
                b = False
            else:
                raise Exception("Not yes or no: %s" % val)
        d[k] = b

    yesno2bool(c, 'AddEdge')

    if c['identity']:
        c['expvol_ss'] = c['anatval_ss'] = False
    else:
        yesno2bool(c, 'expvol_ss')
        yesno2bool(c, 'isepi')

    # see if we can get the fs_sid
    # (only if surfdir is set properly)
    # XXX not sure if this still makes sense
    c['fs_sid'] = None
    surfdir = c.get('surfdir', None)
    if not surfdir is None and os.path.exists(surfdir):
        fs_log_fn = os.path.join(surfdir, '..', 'scripts', 'recon-all.done')
        print "Looking in %s" % fs_log_fn
        if os.path.exists(fs_log_fn):
            with open(fs_log_fn) as f:
                lines = f.read().split('\n')
                for line in lines:
                    if line.startswith('SUBJECT'):
                        fs_sid = line[8:]
                        c['fs_sid'] = fs_sid
                        print "Found Freesurfer sid %s" % fs_sid
                        break

    if c['fs_sid'] is None:
        c['fs_sid'] = sid
        print "Unable to find proper Freesurfer sid"

    pathvars = ['anatvol', 'expvol', 'epivol', 'refdir', 'surfdir']
    for pathvar in pathvars:
        if pathvar in c and c[pathvar]:
            c[pathvar] = os.path.abspath(c[pathvar])
            print "Set absolute path for %s: %s" % (pathvar, c[pathvar])

    if c['template'] and c['notemplate']:
        error('Cannot have both template and notemplate')

    if 'expvol' in c:
        p, n, o, e = utils.afni_fileparts(c['expvol'])

        if c.get('outvol_space', None) is None:
            cmd = '3dAttribute TEMPLATE_SPACE %s' % c['expvol']
            outvol_space = utils.cmd_capture_output(cmd)
            outvol_space = outvol_space.split('~')[0].strip()
            if len(outvol_space) and not c['notemplate'] and outvol_space.lower() != 'orig':
                print "Detected TEMPLATE_SPACE=%s" % outvol_space
                c['outvol_space'] = outvol_space
                if o == '+orig':
                    o = '+tlrc'
                    print "Template space '%s' detected: output has extension %s" % (outvol_space, o)
                c['template'] = True
            else:
                c['outvol_space'] = '+orig'

        if len(o): #'+orig' or '+tlrc'
            c['outvol_ext'] = o
            c['outvol_fullext'] = o + e
            c['outvol_view'] = o[1:]
        else:
            # For NIFTI - output in orig or tlrc
            c['outvol_view'] = 'tlrc' if c['template'] else 'orig'
            c['outvol_ext'] = '+' + c['outvol_view']
            c['outvol_fullext'] = c['outvol_ext'] + '.HEAD'


    return c

def getenv():
    '''returns the path environment
    As a side effect we ensure to set for FreeSurfer's HOME'''
    env = os.environ

    if 'FREESURFER_HOME' not in env:
        env['FREESURFER_HOME'] = env['HOME'] # FreeSurfer requires this var, even though we don't use it

    return env

def run_toafni(config, env):
    '''convert surfaces to AFNI (or SUMA, rather) format'''
    cmds = []

    sd = config['sumadir']
    sid = config['sid']

    if sid is None:
        raise ValueError("Subject id is not set, cannot continue")
    fs_sid = config['fs_sid']

    # files that should exist if Make_Spec_FS was run successfully
    checkfns = ['brainmask.nii',
              'T1.nii',
              'aseg.nii']

    filesexist = all([os.path.exists('%s/%s' % (sd, fn)) for fn in checkfns])

    if config['overwrite'] or not filesexist:
        if config['overwrite']:
            if filesexist:
                cmds.append('rm -rf "%s"' % sd)
        cmds.append('cd %(surfdir)s;@SUMA_Make_Spec_FS -sid %(sid)s -no_ld' % config)
        utils.run_cmds(cmds, env)
    else:
        print "SUMA conversion appears to have been performed already for %s in %s" % (sid, sd)



def run_mapico(config, env):
    '''run MapIcosehedron to convert surfaces to standard topology'''
    sumadir = config['sumadir']
    firstcmd = 'cd "%s" || exit 1' % sumadir
    cmds = []
    icolds, hemis = _get_hemis_icolds(config)
    sid = config['sid'] # subject id
    ext = '.asc' # input is always ascii
    for icold in icolds:
        icoprefix = config['mi_icopat'] % icold
        spherefns = []
        for hemi in hemis:
            if not config['overwrite']:
                # the last file that is generated by MapIcosahedron
                lastsurffn = '%s/%s%sh.sphere.reg%s' % (sumadir, config['mi_icopat'] % icold, hemi, ext)
                spherefns.append(lastsurffn)
                if os.path.exists(lastsurffn):
                    print("Seems MapIcosahedron was already run for %sh with ld=%d" % (hemi, icold))
                    continue

            cmd = ('MapIcosahedron -overwrite -spec %s_%sh.spec -ld %d -fix_cut_surfaces -prefix %s' %
                       (sid, hemi, icold, icoprefix))
            cmds.append(cmd)
        if cmds:
            cmd = '%s;%s' % (firstcmd, ';'.join(cmds))
            utils.run_cmds(cmd, env)
            cmds = []
        if len(spherefns) == 2 and 'l' in hemis and 'r' in hemis:
            spheres = map(surf.read, spherefns)

            mapfn = (config['mi_icopat'] % icold) + config['hemimappingsuffix']
            mappathfn = os.path.join(sumadir, mapfn)


            if config['overwrite'] or not os.path.exists(mappathfn):
                eps = .001
                print "Computing bijection between nodes (ico=%d) - this may take a while" % icold
                bijection = surf.get_sphere_left_right_mapping(spheres[0],
                                                                spheres[1],
                                                                eps)

                with open(mappathfn, 'w') as f:
                    f.write('\n'.join(map(str, bijection)))

                    print "Written bijection to %s" % mappathfn




def run_moresurfs(config, env):
    '''Generates additional surfaces in the SUMA dir by averaging existing ones. '''

    # surface1, surface2, and output (averaged) output
    moresurfs = [("smoothwm", "pial", "intermediate"),
                ("intermediate", "inflated", "semiinflated"),
                ("semiinflated", "inflated", "tqinflated")]

    icolds, hemis = _get_hemis_icolds(config)
    d = config['sumadir']
    ext = '.asc' # always AFNI in this stage
    for icold in icolds:
        icoprefix = config['mi_icopat'] % icold
        for hemi in hemis:
            for surftriple in moresurfs:
                fns = ['%s/%s%sh.%s%s' % (d, icoprefix, hemi, surfname, ext)
                     for surfname in surftriple]
                if config['overwrite'] or not os.path.exists(fns[2]):
                    average_fs_asc_surfs(fns[0], fns[1], fns[2])
                else:
                    print "%s already exists" % fns[2]

def _set_vol_space_cmd(fn, config):
    param = ''
    del_cmd = ''

    p, n, o, e = utils.afni_fileparts(fn)

    space = config.get('outvol_space', None)
    if space is not None:
        param += ' -space %s' % space

    view = config.get('outvol_view')
    if view is not None:
        param += ' -view %s' % view

        if len(o) and o[0] == '+' and o[1:].lower() != view.lower():
            trgfn = '%s/%s+%s' % (p, n, view.lower())
            del_cmd = 'rm %s*; ' % trgfn


    if param:
        p, n, o, e = utils.afni_fileparts(fn)
        cmd = 'cd %s; %s 3drefit -overwrite %s %s' % (p, del_cmd, param, n + o + e)
    else:
        cmd = 'echo "Not changing view or space for %s"' % fn
    return cmd

def _convert_vol_space_to_orig_cmd(fn):
    p, n, o, e = utils.afni_fileparts(fn)
    if o == '+orig':
        return "echo '%s' is already in orig space - no conversion" % fn

    t = '__tmp_foo_'
    while os.path.exists(p + '/' + t + o + e):
        t += 'x'

    cmds = ['cd %s' % p,
          '3drename -overwrite %s%s %s%s' % (n, o, t, o),
          '3drefit -view orig %s%s' % (t, o),
          '3drename -overwrite %s+orig %s+orig' % (t, n)]

    return ';'.join(cmds)



def run_skullstrip(config, env):
    fullext = config['outvol_fullext']

    overwrite = config['overwrite']
    refdir = config['refdir']
    cmds = []
    if not os.path.exists(refdir):
        cmds.append('mkdir %(refdir)s' % config)

    sumadir = config['sumadir']
    sid = config['sid']
    fs_sid = config['fs_sid']

    if not sid:
        raise ValueError("Subject id is not set, cannot continue")

    # process the surfvol anatomical.
    # because it's already skull stripped by freesurfer
    # simply copy it over; rename brain.nii to surfvol_ss
    surfvol_srcs = ['%s/%s' % (sumadir, fn)
                  for fn in ['brain.nii',
                             'T1.nii']]

    surfvol_trgs = ['%s/%s' % (refdir, fn)
                  for fn in ['%s_SurfVol_ss%s' % (sid, fullext),
                             '%s_SurfVol%s' % (sid, fullext)]]

    for src, trg in zip(surfvol_srcs, surfvol_trgs):
        if os.path.exists(trg) and not overwrite:
            print '%s already exists' % trg
        else:
            t_p, t_n, t_o, t_e = utils.afni_fileparts(trg)
            trg_short = '%s%s' % (t_n, t_o)
            cmds.append('cd "%s"; 3dresample -overwrite -orient LPI -inset %s -prefix ./%s' %
                        (refdir, src, trg_short))
            cmds.append(_set_vol_space_cmd('%s/%s+orig' % (refdir, t_n), config))

    # process experimental volume.
    expvol_src = config['expvol']
    do_ss = config['expvol_ss']
    [e_p, e_n, e_o, e_e] = utils.afni_fileparts(expvol_src)

    expvol_trg_prefix = '%s%s' % (e_n, config['sssuffix'] if do_ss else '')
    expvol_trg_tmp_prefix = '__tmp_%s' % expvol_trg_prefix

    expvol_trg = '%s/%s%s' % (refdir, expvol_trg_prefix, fullext)

    print "Attempt %s -> %s" % (expvol_src, expvol_trg)

    ext = config['outvol_ext']

    if overwrite or not utils.afni_fileexists(expvol_trg):
        if do_ss:
            cmds.append('cd "%s";3dSkullStrip -overwrite -prefix ./%s%s -input %s' %
                            (refdir, expvol_trg_tmp_prefix, ext, expvol_src))
        else:
            cmds.append('cd "%s";3dbucket -overwrite -prefix ./%s%s %s' %
                            (refdir, expvol_trg_tmp_prefix, ext, expvol_src))
        cmds.append('cd "%s"; 3dresample -overwrite -orient LPI -prefix %s -inset %s%s' %
                            (refdir, expvol_trg_prefix, expvol_trg_tmp_prefix, ext))
        cmds.append('rm %s/%s*' % (refdir, expvol_trg_tmp_prefix))
        cmds.append(_set_vol_space_cmd(expvol_trg, config))
    else:
        print "No skull strip because already exists: %s%s" % (expvol_trg_prefix, ext)

    utils.run_cmds(cmds, env)

def run_alignment(config, env):
    '''Aligns anat (which is assumed to be aligned with EPI data) to FreeSurfer SurfVol

    This function strips the anatomicals (by default), then uses align_epi_anat.py
    to estimate the alignment, then applies this transformation to the non-skull-stripped
    SurfVol and also to the surfaces. Some alignment headers will be nuked'''
    overwrite = config['overwrite']
    alignsuffix = config['al2expsuffix']
    refdir = config['refdir']

    fullext = config['outvol_fullext']
    ext = config['outvol_ext']

    if config['sid'] is None:
        raise ValueError('Need sid')

    cmds = []
    if not os.path.exists(config['refdir']):
        cmds.append('mkdir %(refdir)s' % config)

    # two volumes may have to be stripped: the inpput anatomical, and the surfvol.
    # put them in a list here and process them similarly
    surfvol = '%(refdir)s/%(sid)s_SurfVol%(outvol_fullext)s' % config
    surfvol_ss = '%(refdir)s/%(sid)s_SurfVol%(sssuffix)s%(outvol_fullext)s' % config

    e_p, e_n, _, _ = utils.afni_fileparts(config['expvol'])
    if config['expvol_ss']:
        e_n = '%s%s' % (e_n, config['sssuffix'])
    expvol = '%s/%s%s' % (refdir, e_n, fullext)

    volsin = [surfvol_ss, expvol]
    for volin in volsin:
        if not os.path.exists(volin):
            raise ValueError('File %s does not exist' % volin)

    a_n = utils.afni_fileparts(volsin[0])[1] # surfvol input root name
    ssalprefix = '%s%s' % (a_n, alignsuffix)

    unity = "1 0 0 0 0 1 0 0 0 0 1 0" # we all like unity, don't we?

    fullmatrixfn = '%s_mat.aff12.1D' % ssalprefix
    aloutfns = ['%s%s' % (ssalprefix, fullext), fullmatrixfn] # expected output files if alignment worked
    if config['overwrite'] or not all([os.path.exists('%s/%s' % (refdir, f)) for f in aloutfns]):
        alignedfn = '%s/%s%s' % (refdir, ssalprefix, fullext)

        if config['identity']:
            fullmatrix_content = '"MATRIX(%s)"' % unity.replace(" ", ",")

            cmd = 'cd "%s"; cat_matvec %s > %s; 3dcopy -overwrite %s%s %s%s%s' % (refdir, fullmatrix_content, fullmatrixfn, a_n, ext, a_n, alignsuffix, ext)
        else:
            # use different inputs depending on whether expvol is EPI or ANAT
            twovolpat = ('-anat %s -epi %s -anat2epi -epi_base 0 -anat_has_skull no -epi_strip None' if config['isepi']
                       else '-dset1 %s -dset2 %s -dset1to2 -dset1_strip None -dset2_strip None')
            # use this pattern to generate a suffix
            twovolsuffix = twovolpat % (volsin[0], volsin[1])

            aea_opts = config['aea_opts']
            if config['template']:
                aea_opts += " -Allineate_opts '-maxrot 10 -maxshf 10 -maxscl 1.5'"
            # align_epi_anat.py
            cmd = 'cd "%s"; align_epi_anat.py -overwrite -suffix %s %s %s' % (refdir, alignsuffix, twovolsuffix, aea_opts)

        cmds.append(cmd)

        cmds.append(_set_vol_space_cmd(alignedfn, config))

        utils.run_cmds(cmds, env)

    else:
        print "Alignment already done - skipping"

        # run these commands first, then check if everything worked properly


    cmds = []

    # see if the expected transformation file was found
    if not config['identity'] and not os.path.exists('%s/%s' % (refdir, fullmatrixfn)):
        raise Exception("Could not find %s in %s" % (fullmatrixfn, refdir))

    # now make a 3x4 matrix
    matrixfn = '%s%s.A2E.1D' % (a_n, alignsuffix)
    if overwrite or not os.path.exists('%s/%s' % (refdir, matrixfn)):
        cmds.append('cd "%s"; cat_matvec %s > %s || exit 1' % (refdir, fullmatrixfn, matrixfn))


    # make an aligned, non-skullstripped version of SurfVol in refdir
    alprefix = '%s_SurfVol%s' % (config['sid'], alignsuffix)
    svalignedfn = '%s/%s%s' % (refdir, alprefix, fullext)

    newgrid = 1 # size of anatomical grid in mm. We'll have to resample, otherwise 3dWarp does
              # not respect the corners of the volume (as of April 2012)

    if overwrite or not os.path.exists(svalignedfn):
        #if not config['fs_sid']:
        #    raise ValueError("Don't have a freesurfer subject id - cannot continue")

        #surfvolfn = '%s/%s_SurfVol+orig' % (config['sumadir'], config['fs_sid'])
        surfvolfn = '%s/T1.nii' % config['sumadir']
        cmds.append('cd "%s";3dWarp -overwrite -newgrid %f -matvec_out2in `cat_matvec -MATRIX %s` -prefix ./%s %s' %
                    (refdir, newgrid, matrixfn, alprefix, surfvolfn))
        cmds.append(_set_vol_space_cmd('%s/%s+orig' % (refdir, alprefix), config))

    else:
        print '%s already exists - skipping Warp' % svalignedfn

    utils.run_cmds(cmds, env)
    cmds = []

    # nuke afni headers
    headernukefns = ['%s%s' % (f, fullext) for f in [ssalprefix, alprefix]]
    headernukefields = ['ALLINEATE_MATVEC_B2S_000000',
                      'ALLINEATE_MATVEC_S2B_000000',
                      'WARPDRIVE_MATVEC_FOR_000000',
                      'WARPDRIVE_MATVEC_INV_000000']

    for fn in headernukefns:
        for field in headernukefields:
            # nuke transformation - otherwise AFNI does this unwanted transformation for us
            fullfn = '%s/%s' % (refdir, fn)

            if not (os.path.exists(fullfn) or config['identity']):
                raise ValueError("File %r does not exist" % fullfn)

            refitcmd = "3drefit -atrfloat %s '%s' %s" % (field, unity, fn)

            # only refit if not already in AFNI history (which is stored in HEADfile)
            cmd = 'cd "%s"; m=`grep "%s" %s | wc -w`; if [ $m -eq 0 ]; then %s; else echo "File %s seems already 3drefitted"; fi' % (refdir, refitcmd, fn, refitcmd, fn)
            cmds.append(cmd)
    utils.run_cmds('; '.join(cmds), env)
    cmds = []

    # run AddEdge so that volumes can be inspected visually for alignment
    if config['AddEdge']:
        use_ss = config['expvol_ss']

        # ae_{e,s}_n are AddEdge names for expvol and surfvol
        ae_e_n = utils.afni_fileparts(config['expvol'])[1]
        if use_ss:
            ae_e_n += config['sssuffix']
        ae_s_n = ssalprefix #if use_ss else alprefix

        # *_ne have the output extension as well
        ae_e_ne = ae_e_n + ext
        ae_s_ne = ae_s_n + ext

        addedge_fns = ['%s/_ae.ExamineList.log' % refdir]

        exts = ['HEAD', 'BRIK']
        orig_ext = '+orig'
        addedge_rootfns = ['%s_%s%%s' % (ae_e_n, postfix)
                            for postfix in ['e3', 'ec', ae_s_n + '_ec']]
        addedge_rootfns.extend(['%s_%s%%s' % (ae_s_n, postfix)
                            for postfix in ['e3', 'ec']])

        addedge_fns_pat = ['%s.%s' % (fn, e) for fn in addedge_rootfns for e in exts]

        addegde_pathfns_orig = map(lambda x:os.path.join(refdir, x % '+orig'), addedge_fns_pat) + addedge_fns
        addegde_pathfns_ext = map(lambda x:os.path.join(refdir, x % ext), addedge_fns_pat)
        addegde_exists = map(os.path.exists, addegde_pathfns_ext)
        if overwrite or not all(addegde_exists):
            ae_ns = (ae_e_n, ae_s_n)

            cmds.extend(map(lambda fn : 'if [ -e "%s" ]; then rm "%s"; fi' % (fn, fn), addegde_pathfns_orig + addegde_pathfns_ext))
            cmds.append(';'.join(['cd %s' % refdir] +
                                 [_convert_vol_space_to_orig_cmd('%s/%s%s' % (refdir, n, ext))
                                            for n in ae_ns] +
                                 ['\@AddEdge %s+orig %s+orig' % ae_ns]))

            set_space_fns = addegde_pathfns_orig + ['%s/%s%s.%s' % (refdir, fn, orig_ext, exts[0]) for fn in ae_ns]

            for fn in set_space_fns: #['%s/%s' % (refdir, fn % orig_ext) for fn in addedge_fns_pat]:
                if fn.endswith('.log'):
                    continue
                cmds.append('if [ -e %s ]; then %s; fi' % (fn, _set_vol_space_cmd(fn, config)))

            utils.run_cmds(cmds, env)
            cmds = []

        else:
            print "AddEdge seems to have been run already"

        sid = config['sid']
        plot_slice_fns = [(ae_e_n + '_e3', ae_s_n + '_e3', '%s_qa_e3.png' % sid),
                          (None, ae_e_n + '_' + ae_s_n + '_ec', '%s_qa_ec.png' % sid)]


        plot_slice_imgfns = ['%s/%s' % (refdir, fn) for fn in plot_slice_fns]
        if overwrite or not all(map(os.path.exists, plot_slice_imgfns)):
            slice_dims = [0, 1, 2]
            slice_pos = [.35, .45, .55, .65]
            for fns in plot_slice_fns:
                input_fns = []
                for i, fn in enumerate(fns):
                    if fn is not None:
                        fn = '%s/%s' % (refdir, fn)
                        if i <= 1:
                            fn += ext
                    input_fns.append(fn)

                fn1, fn2, fnout = input_fns
                if not os.path.exists(fnout):
                    _make_slice_plot(fn1, fn2, fnout)
                    print "QA Image saved to %s" % fnout
                else:
                    print "Already exists: %s" % fnout
        else:
            print "QA images already exist"



    # because AFNI uses RAI orientation but FreeSurfer LPI, make a new
    # affine transformation matrix in which the signs of
    # x and y coordinates are negated before and after the transformation
    matrixfn_LPI2RAI = '%s.A2E_LPI.1D' % ssalprefix
    if overwrite or not os.path.exists('%s/%s' % (refdir, matrixfn_LPI2RAI)):
        lpirai = '"MATRIX(-1,0,0,0,0,-1,0,0,0,0,1,0)"'
        cmd = ('cd %s; cat_matvec -ONELINE %s `cat_matvec -MATRIX %s` %s > %s' %
             (refdir, lpirai, matrixfn, lpirai, matrixfn_LPI2RAI))
        cmds.append(cmd)

    # apply transformation to surfaces
    [icolds, hemis] = _get_hemis_icolds(config)
    sumadir = config['sumadir']
    sumafiles = os.listdir(sumadir)


    origext = '.asc'
    ext = format2extension(config)
    tp = format2type(config)
    # process all hemispheres and ld values
    for icold in icolds:
        for hemi in hemis:
            pat = '%s%sh.?*%s' % (config['mi_icopat'] % icold, hemi, origext)
            for sumafile in sumafiles:
                if fnmatch.fnmatch(sumafile, pat):
                    if not sumafile.endswith(origext):
                        raise ValueError("%s does not end with %s" % (sumafile, origext))
                    #s = sumafile.split(".")
                    #s[len(s) - 2] += config['alsuffix'] # insert '_al' just before last dot
                    #alsumafile = ".".join(s)
                    extsumafile = sumafile[:-len(origext)]
                    alsumafile = extsumafile + config['alsuffix'] + ext

                    if config['overwrite'] or not os.path.exists('%s/%s' % (refdir, alsumafile)):
                        # now apply transformation
                        cmd = 'cd "%s";ConvertSurface -overwrite -i_fs %s/%s -o_%s ./%s -ixmat_1D %s' % \
                              (refdir, sumadir, sumafile, tp, alsumafile, matrixfn_LPI2RAI)
                        cmds.append(cmd)

                    # as of June 2012 copy the original sphere.reg (not aligned) as well
                    if sumafile == ('%s.sphere.reg%s' % (pat, ext)):
                        sumaout = '%s/%s' % (refdir, extsumafile + ext)
                        if config['overwrite'] or not os.path.exists(sumaout):
                            s = surf.read('%s/%s' % (sumadir, sumafile))
                            surf.write(s, sumaout)
                            #cmds.append('cp %s/%s %s/%s' % (sumadir, sumafile, refdir, sumafile))


        mapfn = (config['mi_icopat'] % icold) + config['hemimappingsuffix']
        srcpathfn = os.path.join(sumadir, mapfn)

        if os.path.exists(srcpathfn):
            trgpathfn = os.path.join(refdir, mapfn)
            if not os.path.exists(trgpathfn) or config['overwrite']:
                cmds.append('cp %s %s' % (srcpathfn, trgpathfn))

    utils.run_cmds(cmds, env)

def _make_slice_plot(ulay, olay, fnout, raise_=False):
    if raise_:
        from mvpa2.support.afni import lib_plot_slices
    else:
        try:
            from mvpa2.support.afni import lib_plot_slices
        except:
            print "No slice plotting supported"
            return

    slice_dims = [0, 1, 2]
    slice_pos = [.35, .45, .55, .65]

    lib_plot_slices.make_plot(ulay, olay, slice_dims, slice_pos, output_fn=fnout)


def run_makespec(config, env):
    '''Generates the SUMA specifcation files for all hemispheres and ld values'''
    refdir = config['refdir']
    icolds, hemis = _get_hemis_icolds(config)
    for icold in icolds:
        for hemi in hemis:

            # make spec file
            surfprefix = '%s%sh' % (config['mi_icopat'] % icold, hemi)
            specfn = afni_suma_spec.canonical_filename(icold, hemi,
                                                       config['alsuffix'])
            specpathfn = os.path.join(refdir, specfn)

            if config['overwrite'] or not os.path.exists(specpathfn):
                suma_makespec(refdir, surfprefix, config['surfformat'], specpathfn, removepostfix=config['alsuffix'])
            else:
                print "Skipping spec for %s" % specpathfn

            # make simple script to run afni and suma
            runsumafn = '%s/%sh_ico%d_runsuma.sh' % (refdir, hemi, icold)
            surfvol = '%(sid)s_SurfVol%(al2expsuffix)s+orig' % config

            if config['overwrite'] or not os.path.exists(runsumafn):
                suma_makerunsuma(runsumafn, specfn, surfvol)

def run_makespec_bothhemis(config, env):
    refdir = config['refdir']
    overwrite = config['overwrite']
    icolds, hemis = _get_hemis_icolds(config)

    ext = format2extension(config)

    if hemis != ['l', 'r']:
        raise ValueError("Cannot run without left and right hemisphere")

    for icold in icolds:
        specs = []
        for hemi in hemis:
            #surfprefix = '%s%sh' % (config['mi_icopat'] % icold, hemi)
            specfn = afni_suma_spec.canonical_filename(icold, hemi,
                                                       config['alsuffix'])
            specpathfn = os.path.join(refdir, specfn)
            s = afni_suma_spec.read(specpathfn)

            specs.append(afni_suma_spec.read(specpathfn))

        add_states = ['inflated', 'full.patch.flat', 'sphere.reg']
        add_states_required = [True, False, True] # flat surface is optional
        for add_state, is_req in zip(add_states, add_states_required):
            has_state = all([len(spec.find_surface_from_state(add_state)) == 1
                                    for spec in specs])

            if not has_state:
                if is_req:
                    error('cannot find state %s' % add_state)
                else:
                    # skip this state
                    print "Optional state %s not found - skipping" % add_state
                    continue

            specs = afni_suma_spec.hemi_pairs_add_views(specs,
                            add_state, ext, refdir, overwrite=overwrite)


        spec_both = afni_suma_spec.combine_left_right(specs)


        # generate spec files for both hemispheres
        hemiboth = 'b'
        specfn = afni_suma_spec.canonical_filename(icold, hemiboth, config['alsuffix'])
        specpathfn = os.path.join(refdir, specfn)
        spec_both.write(specpathfn, overwrite=overwrite)

        # merge left and right into one surface
        # and generate the spec files as well
        hemimerged = 'm'
        specfn = afni_suma_spec.canonical_filename(icold, hemimerged, config['alsuffix'])
        specpathfn = os.path.join(refdir, specfn)

        if config['overwrite'] or not os.path.exists(specpathfn):
            spec_merged, surfs_to_join = afni_suma_spec.merge_left_right(spec_both)
            spec_merged.write(specpathfn, overwrite=overwrite)

            full_path = lambda x:os.path.join(refdir, x)
            for fn_out, fns_in in surfs_to_join.iteritems():
                surfs_in = [surf.read(full_path(fn)) for fn in fns_in]

                if all(['full.patch.flat' in fn for fn in fns_in]):
                    # left hemi of flat; rotate 180 degrees, reposition again
                    surfs_in[0] = surfs_in[0] * [-1, -1, 1]
                    surfs_in = surf.reposition_hemisphere_pairs(surfs_in[0], surfs_in[1], 'm')

                surf_merged = surf.merge(*surfs_in)

                if config['overwrite'] or not os.path.exists(full_path(fn_out)):
                    surf.write(full_path(fn_out), surf_merged)
                    print "Merged surfaces written to %s" % fn_out

def run_makesurfmasks(config, env):
    refdir = config['refdir']
    overwrite = config['overwrite']

    if config['sid'] is None:
        raise ValueError('Need sid')

    sumfn = '%s_qa_surf_mask' % config['sid'] # output file

    fullext = config['outvol_fullext']
    volor = config['outvol_ext']

    sumfn_path = '%s/%s%s' % (refdir, sumfn, fullext)
    qafn_path = '%s/%s.png' % (refdir, sumfn)
    checkfn_paths = (sumfn_path, qafn_path)
    if all(map(os.path.exists, checkfn_paths)) and not overwrite:
        print "Already exist: %s" % (", ".join(checkfn_paths))
        return

    icolds, hemis = _get_hemis_icolds(config)

    volexts = ['%s%s' % (volor, e) for e in '.HEAD', '.BRIK*']




    sssuffix = config['sssuffix'] if config['expvol_ss'] else ''
    expvol_fn = '%s%s%s' % (utils.afni_fileparts(config['expvol'])[1],
                            sssuffix,
                            volor)


    #if overwrite or not os.path.exists('%s/%s' % (refdir, sv_al_nii_fn)):
    #    cmd = 'cd %s; 3dcopy -overwrite %s %s' % (refdir, sv_al_orig_fn, sv_al_nii_fn)
    #    utils.run_cmds(cmd, env)


    if hemis != ['l', 'r']:
        raise ValueError("Cannot run without left and right hemisphere")

    icold = max(icolds)

    oneDfn = '__t.1D'
    oneDtfn = '__tt.1D' # transposed
    cmds = ['cd %s' % refdir,
             '1deval -1D: -num %d -expr 1 > %s' % (icold ** 2 * 10 + 1, oneDfn),
             '1dtranspose %s > %s' % (oneDfn, oneDtfn)]

    utils.run_cmds(';'.join(cmds), env)


    tmpfns = [oneDfn, oneDtfn]

    s2v_cmd = ('3dSurf2Vol -map_func mask2 -data_expr "a*%%d" -spec %%s %%s -sv %s'
             ' -grid_parent %s. -prefix %%s -sdata %s -overwrite') % \
                                (expvol_fn, expvol_fn, oneDtfn)

    infix2val = {'-surf_A pial':1,
               '-surf_A smoothwm':2,
               '-surf_A smoothwm -surf_B pial -f_steps 20': 4}

    volfns = []
    for hemi in hemis:
        specfn = afni_suma_spec.canonical_filename(icold, hemi,
                                                       config['alsuffix'])

        for infix, val in infix2val.iteritems():
            fnprefix = '__m%d_%sh' % (val, hemi)
            cmd = s2v_cmd % (val, specfn, infix, fnprefix)
            utils.run_cmds('cd %s;%s' % (refdir, cmd))
            tmpfns.extend(['%s%s' % (fnprefix, e) for e in volexts])
            volfns.append(fnprefix + volor)

    cmds = ['cd %s' % refdir]
    catfn = '__cat'
    cmds.extend(['3dTcat -overwrite -prefix %s %s' % (catfn, ' '.join(volfns)),
                 '3dTstat -overwrite -sum -prefix %s %s%s' % (sumfn, catfn, volor)])
    tmpfns.extend(['%s%s' % (catfn, e) for e in volexts])


    cmds.extend('rm %s' % fn for fn in tmpfns)
    cmds.append('echo "Surface mask in %s"' % sumfn)

    utils.run_cmds(';'.join(cmds), env)

    # make plot
    if overwrite or not os.path.exists(qafn_path):
        expvol_path = '%s/%s' % (refdir, expvol_fn)
        _make_slice_plot(expvol_path,
                         sumfn_path,
                         qafn_path)


def suma_makerunsuma(fnout, specfn, surfvol):
    '''Generate a simple script to launch AFNI and SUMA with NIML enabled
    Scripts can be run with ./lh_ico100_seesuma.sh (for left hemisphere and ld=100)'''

    shortspecfn = os.path.split(specfn)[1] # remove path

    lines = ['export SUMA_AllowDsetReplacement=YES',
           'killall afni',
           'afni -niml &'
           'suma -spec %s -sv %s' % (shortspecfn, surfvol)]

    with open(fnout, 'w') as f:
        f.write('\n'.join(lines))
        f.close()
        os.chmod(fnout, 0777)


    print 'Generated run suma file in %s' % fnout


def suma_makespec(directory, surfprefix, surf_format, fnout=None, removepostfix=''):
    '''Generates a SUMA specification file that contains information about
    the different surfaces'''



    postfix = format2extension(surf_format)
    tp = format2type(surf_format)
    pat = '%s.?*%s' % (surfprefix, postfix)

    #removepostfix = config['alsuffix']

    fns = os.listdir(directory)
    surfname2filename = dict()
    for fn in fns:
        if fnmatch.fnmatch(fn, pat):
            surfname = fn[len(surfprefix) + 1:(len(fn) - len(postfix))]

            if surfname.endswith(removepostfix):
                surfname = surfname[:-len(removepostfix)]
            surfname2filename[surfname] = fn

    # only include these surfaces
    usesurfs = ['smoothwm', 'intermediate', 'pial', 'semiinflated',
                 'tqinflated', 'inflated', 'full.patch.flat', 'sphere.reg']
    isanatomical = dict(zip(usesurfs, [True, True, True] + [False] * 5))


    # make the spec file
    lines = []
    lines.append('# Created %s' % str(datetime.datetime.now()))
    lines.append('Group = all')
    lines.append('')

    lines.extend('StateDef = %s' % f for f in usesurfs if f in surfname2filename)

    lines.append('')
    localdomainparent = surfname2filename.get('smoothwm', None)

    ndim = lambda x:2 if 'full.patch' in x else 3

    for surfname in usesurfs:
        if surfname in surfname2filename:
            ldp = ('SAME' if (not localdomainparent or
                              localdomainparent == surfname2filename[surfname])
                         else localdomainparent)
            lines.extend(['NewSurface',
                          'SurfaceFormat = %s' % surf_format.upper(),
                          'SurfaceType = %s' % format2spectype(surf_format),
                          'FreeSurferSurface = %s' % surfname2filename[surfname],
                          'LocalDomainParent = %s' % ldp,
                          'LocalCurvatureParent = %s' % ldp,
                          'SurfaceState = %s' % surfname,
                          'EmbedDimension = %d' % ndim(surfname),
                          'Anatomical = %s' % ('Y' if isanatomical[surfname] else 'N'),
                         ''])
            if localdomainparent is None:
                localdomainparent = surfname2filename[surfname]



    if fnout:
        f = open(fnout, 'w')
        f.write('\n'.join(lines))
        f.close()
        print 'Generated SUMA spec file in %s' % fnout
    else:
        print "No output"




def average_fs_asc_surfs(fn1, fn2, fnout):
    '''averages two surfaces'''
    surf1 = surf.read(fn1)
    surf2 = surf.read(fn2)
    surfavg = surf1 * .5 + surf2 * .5
    surf.write(fnout, surfavg)

def _get_hemis_icolds(config):
    '''returns the icolds (as a list of integers) and the hemispheres (as a list of single characters)'''
    icolds = [int(v) for v in config['ld'].split('+')] # linear divisions for MapIcosehedron
    hemis = config['hemi'].split('+') # list of hemis (usually ['l','r'])
    return (icolds, hemis)


def run_all(config, env):
    '''run commands from all steps specified in config'''
    cmds = []

    print config

    steps = config['steps'].split('+')
    step2func = {'toafni':run_toafni,
               'mapico':run_mapico,
               'moresurfs':run_moresurfs,
               'skullstrip':run_skullstrip,
               'align':run_alignment,
               'makespec':run_makespec,
               'makespecboth':run_makespec_bothhemis,
               'makesurfmasks':run_makesurfmasks}

    if not all(s in step2func for s in steps):
        raise Exception("Illegal step in %s" % steps)

    for step in steps:
        if step in step2func:
            print "Running: %s" % step
            step2func[step](config, env)
        else:
            raise ValueError('Step not recognized: %r' % step)

    return cmds

def getparser():
    description = '''
Anatomical preprocessing to align FreeSurfer surfaces with AFNI data.

%s

Copyright 2010-2012 Nikolaas N. Oosterhof <nikolaas.oosterhof@unitn.it>

''' % __usage_doc__

    epilog = '''
This function is *experimental*; using the --overwrite option may
remove and/or overwrite existing files.'''

    yesno = ["yes", "no"]
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='prep_afni_surf %s' % _VERSION)
    parser.add_argument("-s", "--sid", required=False, help="subject id used in @SUMA_Make_Spec_FS ")
    parser.add_argument("-d", "--surfdir", required=False, help="FreeSurfer surf/ directory")
    parser.add_argument("-a", "--anatvol", required=False, help="Anatomical that is assumed to be in alignment with the EPI data of interest")
    parser.add_argument("-e", "--epivol", required=False, help="EPI data of interest")
    parser.add_argument('-x', "--expvol", required=False, help="Experimental volume to which SurfVol is aligned")
    parser.add_argument('-E', "--isepi", required=False, choices=yesno, help="Is the experimental volume an EPI (yes) or anatomical (no)")
    parser.add_argument('-r', '-o', "--refdir", required=True, help="Output directory in which volumes and surfaces are in reference to ANATVOL or EPIVOL")
    parser.add_argument("-p", "--steps", default='all', help='Processing steps separated by "+"-characters. "all" is the default and equivalent to "toafni+mapico+moresurfs+skullstrip+align+makespec+makespecboth+makesurfmasks"')
    parser.add_argument("-l", "--ld", default="4+8+16+32+64+128", help="MapIcosahedron linear devisions, e.g. 80, or 16+96 (for both 16 or 96). The default is 4+8+16+32+64+128")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Overwrite existing files")
    parser.add_argument('--hemi', default='l+r', choices=['l', 'r', 'l+r'], help='Hemispheres to process ([l+r])')
    parser.add_argument('-S', "--expvol_ss", default='yes', choices=yesno, help='Skull strip experimental volume ([yes],no)')
    parser.add_argument('--aea_opts', default='-cmass cmass+xyz -big_move', help="Options given to align_epi_anat ([-cmass cmass+xyz -big_move])")
    parser.add_argument('-I', '--identity', action="store_true", default=False, help="Use identity transformation between SurfVol and anat/epivol (no alignment)")
    parser.add_argument('-A', '--AddEdge', default='yes', choices=yesno, help="Run AddEdge on aligned volumes ([yes])")
    parser.add_argument('-f', '--surfformat', default='ascii', choices=['gifti', 'ascii'], help="Output format of surfaces: 'ascii' (default - for now) or 'gifti'")
    parser.add_argument('-T', '--template', action="store_true", default=False, help="Indicate that the experimental volume (suppplied by '-e', '-a', or '-x') is in template space. This will add \"-Allineate_opts '-maxrot 10 -maxshf 10 -maxscl 1.5'\" to --aea_opts")
    parser.add_argument('-t', '--notemplate', action="store_true", default=False, help="Indicate that the experimental volume (suppplied by '-e', '-a', or '-x') is not in template space. ")

    # expvol_space  (template space) MNI, TLRC, MNI_ANAT, None
    # expvol_ext      +tlrc, +orig, .nii, .nii.gz
    # expvol_fullext  +tlrc.HEAD (set based on

    return parser

def getoptions():
    parser = getparser()
    args = None

    namespace = parser.parse_args(args)
    return vars(namespace)

def _test_me(config):
    datadir = os.path.abspath('.') + '/' #/Users/nick/Downloads/subj1/'
    refdir = datadir + '_test_ref'
    surfdir = datadir + '/subj1/surf'

    refs = ['-e bold_mean.nii', '-a anat.nii', '-e bold_mean+orig', '-a anat+orig',
            '-e bold_mean_ss.nii', '-a anat_ss+orig']


    for i, ref in enumerate(refs):
        tp, fn = ref.split(' ')
        do_ss = not ('_ss' in fn)
        c = getdefaults()

        if 'refdir' in config:
            refdir = os.path.abspath('%s%s_%d' % (datadir, config['refdir'], i))

        c.update(dict(isepi=tp == '-e', refdir=refdir, expvol=datadir + fn, surfdir=surfdir,
                      identity=False, expvol_ss=do_ss, AddEdge=True, steps='all', ld="4+32",
                      aea_opts='-cmass cmass+xyz -big_move', alsuffix='_al', verbose=True))
        #c.update(config)

        c['overwrite'] = False # i == 0 and utils.which('mris_convert')

        env = getenv()
        c = augmentconfig(c)
        print c
        run_all(c, env)

def run_prep_afni_surf(config_dict):
    config = getdefaults()
    config.update(config_dict) # overwrite default input arguments

    checkconfig(config)
    augmentconfig(config)

    environment = getenv()
    run_all(config, environment)

# this is a little hack so that python documentation
# is added from the parser defined above
def _set_run_surf_anat_preproc_doc():
    import textwrap
    p = getparser()
    aa = p._actions
    ds = []
    for a in aa:
        if a.nargs != 0:
            ch = 'str'
            if a.choices:
                ch = ' or '.join('%r' % c for c in a.choices)

            if a.default:
                ch += ' [%r]' % a.default

            bd = map(lambda x:'    ' + x, textwrap.wrap(a.help))
            ds.append('%s: %s\n%s' % (a.dest, ch, '\n'.join(bd)))

    # text to include in between the modules' docstring and the
    # generated parameter documentation
    intermediate = '''
Parameters
----------
'''

    run_prep_afni_surf.__doc__ = __doc__ + intermediate + '\n'.join(ds)

# apply setting the documentation
_set_run_surf_anat_preproc_doc()



