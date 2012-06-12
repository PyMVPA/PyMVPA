'''
Created on Jan 29, 2012

@author: nick

Purpose: anatomical preprocessing for surface-based voxel selection
Provides functionality to:
- convert freesurfer surfaces to AFNI/SUMA format (using SUMA_Make_Spec_FS)
- resample surfaces to standard topology (using MapIcosahedron)
- generate additional surfaces by averaging existing ones
- find the transformation from freesurfer 
- make SUMA specification files, and see_suma* shell scripts

At the moment we assume a processing stream with freesurfer for surface 
reconstruction, and AFNI/SUMA for coregistration and visualization.

If EPIs from multiple sessions are aligned, this script should use different directories 
for refdir for each session, otherwise naming conflicts may occur
'''

import os, fnmatch, datetime, re, argparse, utils, surf_fs_asc, collections, copy, surf, afni_suma_spec
from utils import afni_fileparts

def getdefaults():
    '''set up default parameters - for testing now'''
    rootdir = '/Users/nick/Downloads/fingerdata-0.2/'
    d = { #  options that must be set properly
       'sid':'s88', # subject id
       'ld': '4+8+16+32+64+128', # (list of) number of linear divisions for mapicoshedron
       'steps':'all', # one or more of: toafni+mapico+moresurfs+align+makespec
                      # or 'all' to run all steps

       # usually these defaults are fine
       'overwrite':False, # delete directories and files before running commands - BE CAREFUL with this option
       'hemi':'l+r', # hemispheres to process, usually l+r
       'mi_icopat':'ico%d_', # pattern for icosehedron output; placeholder is for ld
       'usenifti':False, # maybe provide some nifti support in the future
       'expvol_ss':True,
       'verbose':True,
       'al2expsuffix':'_al2exp',
       'sssuffix':'_ss'
       }

    return d

def checkconfig(config):
    # for now, assume input is valid

    pass

def augmentconfig(c):
    '''add more configuration values that are *derived* from the current configuration,
    and also checks whether some options are set properly'''

    # ensure surfdir is set properly.
    # normally it should end in 'surf'
    surfdir = os.path.abspath(c['surfdir'])
    parent, nm = os.path.split(surfdir)

    if nm != 'surf':
        jn = os.path.join(surfdir, 'surf')
        if os.path.exists(jn):
            surfdir = jn
    elif nm == 'SUMA':
        surfdir = parent

    if not (os.path.exists(surfdir) and os.path.split(surfdir)[1] == 'surf'):
        print('Warning: surface directory %s not does exist or does not end in "surf"' % surfdir)
        surfdir = None

    c['surfdir'] = surfdir

    # set SUMADIR as well
    c['sumadir'] = '%(surfdir)s/SUMA/' % c

    # derive subject id from surfdir

    sid = os.path.split(os.path.split(surfdir)[0])[1] if surfdir else None
    c['sid'] = c.get('sid', sid)
    if c['sid'] is None:
        print"Warning: no subject id specified"


    c['prefix_sv2anat'] = 'SurfVol2anat'

    # update steps
    if config.get('steps', 'all') == 'all':
        config['steps'] = 'toafni+mapico+moresurfs+skullstrip+align+makespec+makespecboth'

    if c['identity']:
        c['expvol_ss'] = c['anatval_ss'] = c['AddEdge'] = False
    else:
        hasanatvol = 'anatvol' in config and config['anatvol']
        hasepivol = 'epivol' in config and config['epivol']
        hasexpvol = 'expvol' in config and config['expvol']
        hasisepi = 'isepi' in config and config['isepi']

        if hasexpvol:
            if hasanatvol or hasepivol:
                print config
                print hasanatvol, hasepivol, hasexpvol, hasisepi
                raise Exception("expvol specified, but also anatvol or epivol - illegal!")
            if not hasisepi:
                raise Exception("not specified whether expvol is EPI (yes) or anat (no)")

        else:
            if hasanatvol:
                if 'epivol' in config and config['epivol']:
                    raise Exception("Cannot have both anatvol and epivol")
                else:
                    config['expvol'] = config['anatvol']
                    config['isepi'] = False
                    del(config['anatvol'])
            else:
                if hasepivol:
                    config['expvol'] = config['epivol']
                    config['isepi'] = True
                    del(config['epivol'])
                else:
                    print("Warning: no anatomical or functional experimental voume defined")

        def yesno2bool(d, k): # dict, key
            '''Assuming d[k] contains the word 'yes' or 'no', makes d[k] a boolean 
            (True=='yes',False=='no'); otherwise an exception is thrown. The dict is updated'''
            val = d[k]
            if type(val) == bool:
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

        yesno2bool(c, 'expvol_ss')
        yesno2bool(c, 'isepi')
        yesno2bool(c, 'AddEdge')


    pathvars = ['anatvol', 'expvol', 'epivol', 'refdir', 'surfdir']
    for pathvar in pathvars:
        if pathvar in c and c[pathvar]:
            c[pathvar] = os.path.abspath(c[pathvar])
            print "Set absolute path for %s: %s" % (pathvar, c[pathvar])

    return c

def getenv():
    '''returns the path environment
    As a side effect we ensure to set for Freesurfer's HOME'''
    env = os.environ
    #p=env.get('PATH','')

    # add for both local machine (NNO) and Hydra. This is a bit clumsy at the moment
    # FIXME
    #p+=':/sw/afni:/Applications/Freesurfer/bin:/Applications/MATLAB_R2011a.app/bin/:/apps/freesurfer/bin:/apps/afni'
    #env['PATH']=p
    if 'FREESURFER_HOME' not in env:
        env['FREESURFER_HOME'] = env['HOME'] # Freesurfer requires this var, even though we don't use it

    return env

def run_toafni(config, env):
    '''convert surfaces to AFNI (or SUMA, rather) format'''
    cmds = []

    sd = config['sumadir']
    sid = config['sid']

    if sid is None:
        raise ValueError("Subject id is not set, cannot continue")

    # files that should exist if Make_Spec_FS was run successfully    
    checkfns = ['brainmask.nii',
              'T1.nii',
              'aseg.nii',
              '%s_SurfVol+orig.HEAD' % sid]

    filesexist = all([os.path.exists('%s/%s' % (sd, fn)) for fn in checkfns])

    if config['overwrite'] or not filesexist:
        if config['overwrite']:
            if filesexist:
                cmds.append('rm -rf "%s"' % sd)
            else:
                raise Exception("Dir %s exists but does not seem to be SUMA dir for subject %s, don't know what do to now" % (sd, sid))
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
    for icold in icolds:
        icoprefix = config['mi_icopat'] % icold
        for hemi in hemis:
            if not config['overwrite']:
                # the last file that is generated by MapIcosahedron
                lastsurffn = '%s/%s%sh.sphere.reg.asc' % (sumadir, config['mi_icopat'] % icold, hemi)
                if os.path.exists(lastsurffn):
                    print("Seems MapIcosahedron was already run for %sh with ld=%d" % (hemi, icold))
                    continue

            cmd = ('MapIcosahedron -overwrite -spec %s_%sh.spec -ld %d -prefix %s' %
                       (sid, hemi, icold, icoprefix))
            cmds.append(cmd)
    if cmds:
        cmd = '%s;%s' % (firstcmd, ';'.join(cmds))
        utils.run_cmds(cmd, env)



def run_moresurfs(config, env):
    '''Generates additional surfaces in the SUMA dir by averaging existing ones. '''

    # surface1, surface2, and output (averaged) output
    moresurfs = [("smoothwm", "pial", "intermediate"),
                ("intermediate", "inflated", "semiinflated"),
                ("semiinflated", "inflated", "tqinflated")]

    icolds, hemis = _get_hemis_icolds(config)
    d = config['sumadir']
    for icold in icolds:
        icoprefix = config['mi_icopat'] % icold
        for hemi in hemis:
            for surftriple in moresurfs:
                fns = ['%s/%s%sh.%s.asc' % (d, icoprefix, hemi, surfname)
                     for surfname in surftriple]
                if config['overwrite'] or not os.path.exists(fns[2]):
                    average_fs_asc_surfs(fns[0], fns[1], fns[2])
                else:
                    print "%s already exists" % fns[2]

def run_skullstrip(config, env):

    if config['identity']:
        return

    overwrite = config['overwrite']
    refdir = config['refdir']
    cmds = []
    if not os.path.exists(refdir):
        cmds.append('mkdir %(refdir)s' % config)

    sumadir = config['sumadir']
    sid = config['sid']

    # process the surfvol anatomical.
    # because it's already skull stripped by freesurfer
    # simply copy it over; rename brain.nii to surfvol_ss 
    surfvol_srcs = ['%s/%s' % (sumadir, fn)
                  for fn in ['brain.nii',
                             '%s_SurfVol+orig.HEAD' % sid]]

    surfvol_trgs = ['%s/%s' % (refdir, fn)
                  for fn in ['%s_SurfVol_ss+orig.HEAD' % sid,
                             '%s_SurfVol+orig.HEAD' % sid]]

    for src, trg in zip(surfvol_srcs, surfvol_trgs):
        if os.path.exists(trg) and not overwrite:
            print '%s already exists' % trg
        else:
            t_p, t_n, t_o, t_e = utils.afni_fileparts(trg)
            trg_short = '%s%s' % (t_n, t_o)
            cmds.append('cd "%s"; 3dcopy -overwrite %s ./%s' %
                        (refdir, src, trg_short))

    # process experimental volume. 
    expvol_src = config['expvol']
    do_ss = config['expvol_ss']
    [e_p, e_n, e_o, e_e] = utils.afni_fileparts(expvol_src)

    if do_ss:
        expvol_trg_prefix = '%s%s' % (e_n, config['sssuffix'])
        cmd = '3dSkullStrip'
        input = '-input'
    else:
        expvol_trg_prefix = e_n
        cmd = '3dbucket'
        input = ''

    if 'nii' in e_e:
        if overwrite or not utils.afni_fileexists('%s/%s+orig.HEAD' % (refdir, e_n)):
            print "Converting %s from NIFTI to AFNI format" % e_n
            cmds.append('cd "%s"; 3dcopy %s ./%s+orig' % (refdir, expvol_src, e_n))

    if overwrite or not utils.afni_fileexists('%s/%s+orig.HEAD' % (refdir, expvol_trg_prefix)):
        cmds.append('cd "%s";%s -overwrite -prefix ./%s+orig %s %s+orig' % (refdir, cmd, expvol_trg_prefix, input, e_n))
    else:
        print "%s already exists" % expvol_trg_prefix

    utils.run_cmds(cmds, env)

def run_alignment(config, env):
    '''Aligns anat (which is assumed to be aligned with EPI data) to Freesurfer SurfVol
    
    This function strips the anatomicals (by default), then uses @SUMA_AlignToExperiment 
    to estimate the alignment, then applies this transformation to the non-skull-stripped
    SurfVol and also to the surfaces. Some alignment headers will be nuked'''
    overwrite = config['overwrite']
    alignsuffix = config['al2expsuffix']
    refdir = config['refdir']

    cmds = []
    if not os.path.exists(config['refdir']):
        cmds.append('mkdir %(refdir)s' % config)

    # two volumes may have to be stripped: the inpput anatomical, and the surfvol.
    # put them in a list here and process them similarly
    surfvol = '%(refdir)s/%(sid)s_SurfVol+orig.HEAD' % config
    surfvol_ss = '%(refdir)s/%(sid)s_SurfVol_ss+orig.HEAD' % config

    e_p, e_n, _, _ = utils.afni_fileparts(config['expvol'])
    if config['expvol_ss']:
        e_n = '%s%s' % (e_n, config['sssuffix'])
    expvol = '%s/%s+orig.HEAD' % (refdir, e_n)

    volsin = [surfvol_ss, expvol]
    for volin in volsin:
        if not os.path.exists(volin):
            raise ValueError('File %s does not exist' % volin)
    '''    

    allvols = [surfvol, config['expvol']]
    allvols_ss = [config[s] for s in ['surfvol_ss', 'expvol_ss']] # skulls strip?
    myvolsin = []

    

    alignsuffix = config['al2expsuffix']
    volsin = [] # keeps the 2 output files (surfvol, experimental vol) used for alignment 

    # determine which volume to use. if there is a skull stripped version, use that one. Otherwise
    # take the original volume and assume it is skull stripped

    for volfn in allvols:
        if volfn is None:
            if not config['identity']:
                raise ValueError("Empty volfn")
            volfn_in = None
        else:
            [a_p, a_n, a_o, a_e] = utils.afni_fileparts(volfn)
            volfn_ss = '%s_ss%s%s' % (a_n, a_o, a_e) # skull stripped version
            shortvolfn = '%s%s%s' % (a_n, a_o, a_e) # ensure there is a HEAD extension

            # make list of potential file names, in order of preference
            fullfns = ['%s/%s' % (refdir, volfn_ss), '%s/%s' % (a_p, shortvolfn)]
            print fullfns
            volfn_in = None

            for fullfn in fullfns:
                if os.path.exists(fullfn):
                    volfn_in = os.path.split(fullfn)[1]
                    break

            if not volfn_in:
                utils.run_cmds('ls', env)
                utils.run_cmds('ls %s' % volfn, env)
                raise Exception("Did not find volume for %s" % volfn)
        volsin.append(volfn_in)
    # our two volumes are skull stripped now; run alignment
    '''

    # _,a_n,_,_=utils.afni_fileparts(volsin[0]) # surfvol input name
    a_n = utils.afni_fileparts(volsin[0])[1] # surfvol input root name
    ssalprefix = '%s%s' % (a_n, alignsuffix)

    unity = "1 0 0 0 0 1 0 0 0 0 1 0" # we all like unity, don't we?
    if config['identity']:
        fullmatrixfn = '"MATRIX(%s)"' % unity.replace(" ", ",")
    else:
        fullmatrixfn = '%s_mat.aff12.1D' % ssalprefix

        aloutfns = ['%s+orig.HEAD' % ssalprefix, fullmatrixfn] # expected output files if alignment worked

        if config['overwrite'] or not all([os.path.exists('%s/%s' % (refdir, f)) for f in aloutfns]):
            # use different inputs depending on whether expvol is EPI or ANAT
            twovolpat = ('-anat %s -epi %s -anat2epi -epi_base 0 -anat_has_skull no -epi_strip None' if config['isepi']
                       else '-dset1 %s -dset2 %s -dset1to2 -dset1_strip None -dset2_strip None')
            # use this pattern to generate a suffix
            twovolsuffix = twovolpat % (volsin[0], volsin[1])

            aea_opts = config['aea_opts']
            # align_epi_anat.py 
            cmd = 'cd "%s"; align_epi_anat.py -overwrite -suffix %s %s %s' % (refdir, alignsuffix, twovolsuffix, aea_opts)
            cmds.append(cmd)

        else:
            print "Alignment already done - skipping"

        # run these commands first, then check if everything worked properly
        utils.run_cmds(cmds, env)

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
    svalignedfn = '%s/%s+orig.HEAD' % (refdir, alprefix)

    newgrid = 1 # size of anatomical grid in mm. We'll have to resample, otherwise 3dWarp does
              # not respect the corners of the volume (as of April 2012)

    if overwrite or not os.path.exists(svalignedfn):
        surfvolfn = '%s/%s_SurfVol+orig' % (config['sumadir'], config['sid'])
        cmds.append('cd "%s";3dWarp -overwrite -newgrid %f -matvec_out2in `cat_matvec -MATRIX %s` -prefix ./%s %s' %
                    (refdir, newgrid, matrixfn, alprefix, surfvolfn))
    else:
        print '%s already exists - skipping Warp' % svalignedfn

    utils.run_cmds(cmds, env)
    cmds = []

    # nuke afni headers
    headernukefns = ['%s+orig.HEAD' % f for f in [ssalprefix, alprefix]]
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

    # run AddEdge so that volumes can be inspected visually for alignment
    if config['AddEdge']:
        basedset = volsin[1]
        [d, n, o, e] = afni_fileparts(basedset)
        if 'nii' in e:
            o = '+orig'
            if overwrite or not os.path.exists('%s/%s+orig.HEAD' % refdir, n):
                cmds.append('cd %s; 3dcopy -overwrite %s.nii %s%s' % (refdir, n, n, o))

        dset = '%s+orig.HEAD' % alprefix
        n_dset = afni_fileparts(dset)[1]

        addedge_fns = ['_ae.ExamineList.log']

        print n, n_dset, dset

        exts = ['HEAD', 'BRIK']
        addedge_rootfns = ['%s_%s+orig' % (n, postfix)
                            for postfix in ['e3', 'ec', n_dset + '_ec']]
        addedge_rootfns.extend(['%s_%s+orig' % (n_dset, postfix)
                            for postfix in ['e3', 'ec']])

        addedge_fns = ['%s.%s' % (fn, e) for fn in addedge_rootfns for e in exts]

        addegde_pathfns = map(lambda x:os.path.join(refdir, x), addedge_fns)

        addegde_exists = map(os.path.exists, addegde_pathfns)
        if overwrite or not all(addegde_exists):
            if overwrite:
                cmds.extend(map(lambda fn : 'rm "%s"' % fn, addegde_pathfns))
            cmds.append('cd %s; \@AddEdge %s%s %s' % (refdir, n, o, dset))
        else:
            print "AddEdge seems to have been run already"

    # because AFNI uses RAI orientation but Freesurfer LPI, make a new 
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
    sumafiles = os.listdir(config['sumadir'])

    # process all hemispheres and ld values
    for icold in icolds:
        for hemi in hemis:
            pat = '%s%sh.?*.asc' % (config['mi_icopat'] % icold, hemi)
            for sumafile in sumafiles:
                if fnmatch.fnmatch(sumafile, pat):
                    s = sumafile.split(".")
                    s[len(s) - 2] += "_al" # insert '_al' just before last dot
                    alsumafile = ".".join(s)

                    if config['overwrite'] or not os.path.exists('%s/%s' % (refdir, alsumafile)):
                        # now apply transformation
                        cmd = 'cd "%s";ConvertSurface -overwrite -i_fs %s/%s -o_fs ./%s -ixmat_1D %s' % \
                              (refdir, config['sumadir'], sumafile, alsumafile, matrixfn_LPI2RAI)
                        cmds.append(cmd)

    utils.run_cmds(cmds, env)


def run_makespec(config, env):
    '''Generates the SUMA specifcation files for all hemispheres and ld values'''
    refdir = config['refdir']
    icolds, hemis = _get_hemis_icolds(config)
    for icold in icolds:
        for hemi in hemis:

            # make spec file
            surfprefix = '%s%sh' % (config['mi_icopat'] % icold, hemi)
            specfn = '%s/%s.spec' % (refdir, surfprefix)
            if config['overwrite'] or not os.path.exists(specfn):
                suma_makespec(refdir, surfprefix, specfn)

            # make simple script to run afni and suma
            runsumafn = '%s/%sh_ico%d_runsuma.sh' % (refdir, hemi, icold)
            surfvol = '%(sid)s_SurfVol%(al2expsuffix)s+orig' % config

            if config['overwrite'] or not os.path.exists(runsumafn):
                suma_makerunsuma(runsumafn, specfn, surfvol)

def run_makespec_bothhemis(config, env):
    refdir = config['refdir']
    overwrite = config['overwrite']
    icolds, hemis = _get_hemis_icolds(config)

    if hemis != ['l', 'r']:
        raise ValueError("Cannot run without left and right hemisphere")

    for icold in icolds:
        specs = []
        for hemi in hemis:
            surfprefix = '%s%sh' % (config['mi_icopat'] % icold, hemi)
            specfn = '%s/%s.spec' % (refdir, surfprefix)
            specs.append(afni_suma_spec.read(specfn))

        specs = afni_suma_spec.hemi_pairs_add_views(specs[0], specs[1], 'inflated', refdir, overwrite=overwrite)
        spec_both = afni_suma_spec.merge_left_right(specs[0], specs[1])

        surfprefix = '%s%sh' % (config['mi_icopat'] % icold, 'b')
        specfn = '%s/%s.spec' % (refdir, surfprefix)
        spec_both.write(specfn, overwrite=overwrite)

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

    if config['verbose']:
        print 'Generated run suma file in %s' % fnout




def suma_makespec(indir, surfprefix, fnout=None):
    '''Generates a SUMA specification file that contains information about 
    the different surfaces'''
    postfix = '.asc'
    pat = '%s.?*%s' % (surfprefix, postfix)

    removepostfix = '_al'

    fns = os.listdir(indir)
    surfname2filename = dict()
    for fn in fns:
        if fnmatch.fnmatch(fn, pat):
            surfname = fn[len(surfprefix) + 1:(len(fn) - len(postfix))]

            if surfname.endswith(removepostfix):
                surfname = surfname[:-len(removepostfix)]
            surfname2filename[surfname] = fn

    # only include these surfaces
    usesurfs = ['smoothwm', 'intermediate', 'pial', 'semiinflated', 'tqinflated', 'inflated', 'sphere.reg']
    isanatomical = dict(zip(usesurfs, [True, True, True] + [False] * 4))


    # make the spec file
    lines = []
    lines.append('# Created %s' % str(datetime.datetime.now()))
    lines.append('Group = all')
    lines.append('')

    lines.extend('StateDef = %s' % f for f in usesurfs if f in surfname2filename)

    lines.append('')
    localdomainparent = surfname2filename.get('smoothwm', None)
    for surfname in usesurfs:
        if surfname in surfname2filename:
            ldp = ('SAME' if (not localdomainparent or localdomainparent == surfname2filename[surfname])
                         else localdomainparent)
            lines.extend(['NewSurface',
                          'SurfaceFormat = ASCII',
                          'SurfaceType = FreeSurfer',
                          'FreeSurferSurface = %s' % surfname2filename[surfname],
                          'LocalDomainParent = %s' % ldp,
                          'LocalCurvatureParent = %s' % ldp,
                          'SurfaceState = %s' % surfname,
                          'EmbedDimension = 3',
                          'Anatomical = %s' % ('Y' if isanatomical[surfname] else 'N'),
                         ''])
            if localdomainparent is None:
                localdomainparent = surfname2filename[surfname]

    if fnout:
        f = open(fnout, 'w')
        f.write('\n'.join(lines))
        f.close()
        if config['verbose']:
            print 'Generated SUMA spec file in %s' % fnout


    return lines

def average_fs_asc_surfs(fn1, fn2, fnout):
    '''averages two surfaces'''
    surf1 = surf_fs_asc.read(fn1)
    surf2 = surf_fs_asc.read(fn2)
    surfavg = surf1 * .5 + surf2 * .5
    surf_fs_asc.write(surfavg, fnout, overwrite=True)

def _get_hemis_icolds(config):
    '''returns the icolds (as a list of integers) and the hemispheres (as a list of single characters)'''
    icolds = [int(v) for v in config['ld'].split('+')] # linear divisions for MapIcosehedron
    hemis = config['hemi'].split('+') # list of hemis (usually ['l','r'])
    return (icolds, hemis)


def run_all(config, env):
    '''run commands from all steps specified in config'''
    cmds = []

    steps = config['steps'].split('+')
    step2func = {'toafni':run_toafni,
               'mapico':run_mapico,
               'moresurfs':run_moresurfs,
               'skullstrip':run_skullstrip,
               'align':run_alignment,
               'makespec':run_makespec,
               'makespecboth':run_makespec_bothhemis}

    if not all(s in step2func for s in steps):
        raise Exception("Illegal step in %s" % steps)

    for step in steps:
        if step in step2func:
            print "Running: %s" % step
            step2func[step](config, env)
        else:
            raise ValueError('Step not recognized: %r' % step)

    # this may be a security hazard - but we trust the input
    return cmds



def getoptions():
    yesno = ["yes", "no"]
    parser = argparse.ArgumentParser('Surface preprocessing and alignment for surface-based voxel selection using AFNI, SUMA and python\nNikolaas N. Oosterhof Jan 2012')
    parser.add_argument("-s", "--sid", required=False, help="subject id used in @SUMA_Make_Spec_FS ")
    parser.add_argument("-d", "--surfdir", required=True, help="Freesurfer surf/ directory")
    parser.add_argument("-a", "--anatvol", required=False, help="Anatomical that is assumed to be in alignment with the EPI data of interest")
    parser.add_argument("-e", "--epivol", required=False, help="EPI data of interest")
    parser.add_argument('-v', "--expvol", required=False, help="Experimental volume to which SurfVol is aligned")
    parser.add_argument('-E', "--isepi", required=False, choices=yesno, help="Is the experimental volume an EPI (yes) or anatomical (no)")
    parser.add_argument("-r", "--refdir", required=True, help="Output directory in which volumes and surfaces are in reference to ANAT")
    parser.add_argument("-p", "--steps", default='all', help='which processing steps. "all" is equivalent to "toafni+mapico+moresurfs+skullstrip+align+makespec"')
    parser.add_argument("-l", "--ld", default="4+8+16+32+64+128", help="MapIcosahedron linear devisions, e.g. 80, or 16+96 (for both 16 or 96)")
    parser.add_argument("-o", "--overwrite", action='store_true', default=False, help="Overwrite existing files")
    parser.add_argument('--hemi', default='l+r', choices=['l', 'r', 'l+r'], help='Hemispheres to process ([l+r]')
    parser.add_argument("--expvol_ss", default='yes', choices=yesno, help='Skull strip experimental volume ([yes],no)')
    parser.add_argument('--aea_opts', default='-cmass cmass+xyz -big_move', help="Options given to align_epi_anat, e.g. -big_move")
    parser.add_argument('-I', '--identity', action="store_true", default=False, help="Use identity transformation between SurfVol and anat/epivol (no alignment)")
    parser.add_argument('-A', '--AddEdge', default='yes', choices=yesno, help="Run AddEdge on aligned volumes")

    # for testing
    if True:
        args = None
    else:
        args = "-s s88 --surfdir /Users/nick/Downloads/fingerdata-0.2/fs/s88/surf -a /Users/nick/Downloads/fingerdata-0.2/glm/anat_al+orig -r /Users/nick/Downloads/fingerdata-0.2/refC"

    namespace = parser.parse_args(args)
    return vars(namespace)

if __name__ == '__Xmain__':
    d = '/Users/nick/Downloads/subj1/ref4'
    left = _get_surface_defs(d, 'ico32_lh')
    right = _get_surface_defs(d, 'ico32_rh')

    print left

    _add_multiview_surfaces(d, left, right)


if __name__ == '__main__':



    # get default configuration (for testing)
    # in the future, allow for setting these on command line
    config = getdefaults()
    options = getoptions()
    config.update(options) # overwrite default input arguments 

    # check config
    checkconfig(config)

    print "Using these options:\n"
    for v in options.keys():
        print '  %s = %r' % (v, config[v])


    # add auxiliry configuration settings that are *derived* from config 
    config = augmentconfig(config)

    # get path stuff; try to get matlab, freesurfer, afni in path
    env = getenv()

    # run commands based on config
    cmds = run_all(config, env)
    #cmds='cd /Users/nick/Downloads/fingerdata-0.2/refZ/||exit 1;align_epi_anat.py -overwrite -dset1 ./anat_al_ss+orig -dset2 ./s88_SurfVol_ss+orig -dset1to2 -giant_move -suffix _al2SV -Allineate_opts "-warp shr -VERB -weight_frac 1.0"  -epi_strip None -anat_has_skull no'
    utils.run_cmds(cmds, env)
    if False:
        d = '/Users/nick/Downloads/fingerdata-0.2/refA/'
        fn = d + 'ico96_lh.tqinflated_al.asc'
        average_fs_asc_surfs(fn, fn, 'foo')



