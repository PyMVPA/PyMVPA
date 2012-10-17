# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Utility functions for AFNI files

Created on Feb 19, 2012

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)
'''

import os, subprocess, time, datetime, collections

def as_list(v):
    '''makes this a singleton list if the input is not a list'''
    if type(v) not in [list, tuple]:
        v = [v]
    return v

def afni_fileparts(fn):
    '''File parts for afni filenames.
    
    Returns a tuple with these four parts.
    
    Also works for .nii files, in which case the third part is the empty
    
    Not tested for other file types
    
    Parameters
    ----------
    whole filename
      PATH/TO/FILE/NAME+orig.HEAD
      
    
    Returns
    -------
    fullpath: str
      PATH/TO/FILE
    rootname: str
      NAME
    orientation: str
      +orig
    extensions: str
      .HEAD
    
    '''

    tail, head = os.path.split(fn)

    s = head.split('+')
    name = s[0]
    orient = '+' + s[1] if len(s) == 2  else ''

    afniorients = ['+orig', '+tlrc', '+acpc']
    ext = None

    for a in afniorients:
        if orient.startswith(a):
            #ext=orient[len(a):]
            orient = a
            ext = ".HEAD"
            #if ext=='.':
            #    ext=''

    if ext is None:
        s = name.split(".")
        if len(s) > 1:
            ext = "." + ".".join(s[1:])
            name = s[0]
        else:
            ext = ''

    return tail, name, orient, ext

def afni_fileexists(fn):
    '''
    Parameters
    ----------
    fn : str
        AFNI filename (possibly without .HEAD or .BRIK extension)
    
    Returns
    -------
    bool
        True iff fn exists as AFNI file
    '''
    p, n, o, e = afni_fileparts(fn)

    if o:
        return os.path.isfile('%s/%s%s.HEAD' % (p, n, o))
    else:
        return (e in ['.nii', '.nii.gz']) and os.path.isfile(fn)

def run_cmds(cmds, env=None, dryrun=False):
    '''exectute a list of commands in the shell'''
    if env is None:
        env = os.environ

    # if cmds is just one command, make a singleton list    
    cmds = as_list(cmds)

    # run each command    
    for cmd in cmds:
        print("** Will execute the following commands:")
        for c in cmd.split(';'):
            print '** - %s' % c
        if not dryrun:
            print("**>> Starting now:")

            subprocess.call(cmd, env=os.environ, shell=True)

            print("**<< ... completed execution")

def which(f, env=None):
    '''Finds the full path to a file in the path
    
    Parameters
    ----------
    f: str
        Filename of executable
    env (optional): 
        Environment in which path is found.
        By default this is the environment in which python runs
        
    
    Returns
    str
        Full path of 'f' if 'f' is executable and in the path, 'f' itself 
        if 'f' is a path, None otherwise
    '''
    if env == None:
        env = os.environ

    def is_executable(fullpath):
        return os.path.exists(fullpath) and os.access(fullpath, os.X_OK)

    [p, n] = os.path.split(f)
    if p:
        return f
    else:
        for path in env['PATH'].split(os.pathsep):
            fullfn = os.path.join(path, n)
            if is_executable(fullfn):
                return fullfn
        return None
