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
import os.path as op

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

            subprocess.check_call(cmd, env=env, shell=True)

            print("**<< ... completed execution")

def cmd_capture_output(cmd, env=None):
    if env is None:
        env = os.environ
    return subprocess.check_output(cmd, env=env, shell=True)

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



def _package_afni_nibabel_for_standalone(outputdir, rootname='python'):
    '''
    helper function to put mvpa2.support.{afni,nibabel} into another
    directory (outputdir) where it can function as a stand-alone package
    '''

    outputdir_files = os.path.join(outputdir, rootname)
    for d in (outputdir, outputdir_files):
        if not os.path.exists(d):
            os.mkdir(d)

    fullpath = op.realpath(__file__)
    fullpath_parts = fullpath.split('/')
    if (len(fullpath_parts) < 4 or
            fullpath.split('/')[-4:-1] != ['mvpa2', 'support', 'afni']):
        raise ValueError('This script is not in mvpa2.support.afni. '
                         'Packaging for stand-alone is not supported')

    replacements = {'from mvpa2.base import warning':'def warning(x): print x'}

    rootdir = os.path.join(op.split(fullpath)[0], '..')
    parent_pkg = 'mvpa2.support'
    pkgs = ['afni', 'nibabel']
    srcdirs = [os.path.join(rootdir, pkg) for pkg in pkgs]


    input_path_fns = [os.path.join(d, f) for d in srcdirs
                                         for f in os.listdir(d)
                                         ]

    is_python_file = lambda fn: fn.endswith('.py') and not fn.endswith('__.py')
    input_path_fns = filter(is_python_file, input_path_fns)

    print input_path_fns

    outputfns = []
    for path_fn in input_path_fns:
        fn = os.path.split(path_fn)[1]

        with open(path_fn) as f:
            lines = f.read().split('\n')

        newlines = []
        for line in lines:
            newline = None

            for old, new in replacements.iteritems():
                line = line.replace(old, new)

            if 'import' in line:
                words = line.split()

                for pkg in pkgs:
                    full_pkg = parent_pkg + '.' + pkg
                    trgwords = ['from', full_pkg, 'import']
                    n = len(trgwords)

                    if len(words) >= n and words[:n] == trgwords:
                        # find how many trailing spaces
                        i = 0
                        while line.find(' ', i) == i:
                            i += 1
                        # get everything from import to end of line
                        # with enough spaces in front
                        newline = (' ' * i) + ' '.join(words[(n - 1):])
                        print line
                        print ' -> ', newline
                        break
                    else:
                        if pkg in words:
                            raise ValueError("Not supported in %s: %s" % (path_fn, line))

            if newline is None:
                newline = line

            newlines.append(newline)

        if fn.startswith('lib_'):
            repls = [('lib_', 'pymvpa2-'), ('.py', ''), ('_', '-')]
            srcbinfn = fn
            for src, trg in repls:
                srcbinfn = srcbinfn.replace(src, trg)

            parentfn = os.path.join(rootdir, '..', '..', 'bin', srcbinfn)
            print parentfn
            if os.path.exists(parentfn):
                with open(parentfn) as pf:
                    plines = pf.read().split('\n')
                    in_main = False
                    for line in plines:
                        if '__main__' in line:
                            in_main = True
                        if in_main:
                            newlines.append(line)
                newlines = [plines[0]] + newlines
            else:
                raise ValueError("not found: %s" % parentfn)

            print newlines

            trgfn = os.path.join(outputdir_files, fn.replace('lib_', ''))
        else:
            trgfn = op.join(outputdir_files, fn)

        with open(trgfn, 'w') as f:
            f.write('\n'.join(newlines))

        is_executable = newlines[0].startswith('#!')
        chmod_ = 0777 if is_executable else 0666
        os.chmod(trgfn, chmod_)

        print "Written file %s in %s" % (fn, outputdir_files)
        outputfns.append(os.path.split(trgfn)[1])




    readme = ('''
AFNI I/O and wrapper functions in python

Copyright 2010-2013 Nikolaas N. Oosterhof <nikolaas.oosterhof@unitn.it>

The software in the following files is covered under the MIT License
(included below):
''' +
    '\n'.join(map(lambda x:' - ' + x, outputfns)) +
    '''
Parts of this software is or will be included in PyMVPA.
For information see http://www.pymvpa.org.

-------------------------------------------------------------------------
The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.''')

    readmefn = op.join(outputdir_files, 'COPYING')
    with open(readmefn, 'w') as f:
        f.write(readme)

