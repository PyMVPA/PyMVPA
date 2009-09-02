# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provide system and PyMVPA information useful while reporting bugs
"""

__docformat__ = 'restructuredtext'

import time, sys, os, subprocess
import platform as pl
from tempfile import mkstemp
from StringIO import StringIO

import mvpa
from mvpa.base import externals, cfg

def _t2s(t):
    res = []
    for e in t:
        if isinstance(e, tuple):
            es = _t2s(e)
            if es != '':
                res += ['(%s)' % es]
        elif e != '':
            res += [e]
    return '/'.join(res)

__all__ = ['sysinfo']

def sysinfo(filename=None):
    """Report summary about PyMVPA and the system

    :Keywords:
      filename : None or string
        If provided, information will be stored in a file, not printed
        to the screen
    """

    if filename is None:
        out = StringIO()
    else:
        out = file(filename, 'w')

    out.write("Current date:   %s\n" % time.strftime("%Y-%m-%d %H:%M"))

    out.write("PyMVPA:\n")
    out.write(" Version:       %s\n" % mvpa.__version__)
    out.write(" Path:          %s\n" % mvpa.__file__)

    # Try to obtain git information if available
    out.write(" Version control (GIT):\n")
    try:
        gitpath = os.path.join(os.path.dirname(mvpa.__file__), os.path.pardir)
        gitpathgit = os.path.join(gitpath, '.git')
        if os.path.exists(gitpathgit):
            for scmd, cmd in [
                ('Status', ['status']),
                ('Reference', 'show-ref -h HEAD'.split(' ')),
                ('Difference from last release %s' % mvpa.__version__,
                 ['diff', '--shortstat', 'upstream/%s...' % mvpa.__version__])]:
                try:
                    (tmpd, tmpn) = mkstemp('mvpa', 'git')
                    retcode = subprocess.call(['git',
                                               '--git-dir=%s' % gitpathgit,
                                               '--work-tree=%s' % gitpath] + cmd,
                                              stdout=tmpd,
                                              stderr=subprocess.STDOUT)
                finally:
                    outlines = open(tmpn, 'r').readlines()
                    if len(outlines):
                        out.write('  %s:\n   %s' % (scmd, '   '.join(outlines)))
                    os.remove(tmpn)
                #except Exception, e:
                #    pass
        else:
            raise RuntimeError, "%s is not under GIT" % gitpath
    except Exception, e:
        out.write(' GIT information could not be obtained due "%s"\n' % e)

    out.write('SYSTEM:\n')
    out.write(' OS:            %s\n' %
              ' '.join([os.name,
                        pl.system(),
                        pl.release(),
                        pl.version()]).rstrip())
    out.write(' Distribution:  %s\n' %
              ' '.join([_t2s(pl.dist()),
                        _t2s(pl.mac_ver()),
                        _t2s(pl.win32_ver())]).rstrip())

    # Test and list all dependencies:
    sdeps = {True: [], False: []}
    for dep in sorted(externals._KNOWN):
        sdeps[externals.exists(dep, force=False)] += [dep]
    out.write('EXTERNALS:\n')
    out.write(' Present:       %s\n' % ', '.join(sdeps[True]))
    out.write(' Absent:        %s\n' % ', '.join(sdeps[False]))

    SV = ('.__version__', )              # standard versioning
    out.write(' Versions of critical externals:\n')
    for e, mname, fs in (
        ('ctypes', None, SV),
        ('matplotlib', None, SV),
        ('lxml', None, ('.etree.__version__',)),
        ('nifti', None, SV),
        ('numpy', None, SV),
        ('openopt', 'scikits.openopt', ('.openopt.__version__',)),
        ('pywt', None, SV),
        ('rpy', None, ('.rpy_version',)),
        ('scipy', None, SV),
        ('shogun', None, ('.Classifier.Version_get_version_release()',)),
        ):
        try:
            if not externals.exists(e):
                continue #sver = 'not present'
            else:
                if mname is None:
                    mname = e
                m = __import__(mname)
                svers = [eval('m%s' % (f,)) for f in fs]
                sver = ' '.join(svers)
        except Exception, exc:
            sver = 'failed to query due to "%s"' % str(exc)
        out.write('  %-12s: %s\n' % (e, sver))

    if externals.exists('matplotlib'):
        import matplotlib
        out.write(' Matplotlib backend: %s\n' % matplotlib.get_backend())

    out.write("RUNTIME:\n")
    out.write(" PyMVPA Environment Variables:\n")
    out.write('  '.join(['  %-20s: "%s"\n' % (str(k), str(v))
                        for k, v in os.environ.iteritems()
                        if (k.startswith('MVPA') or k.startswith('PYTHON'))]))

    out.write(" PyMVPA Runtime Configuration:\n")
    out.write('  ' + str(cfg).replace('\n', '\n  ').rstrip() + '\n')

    try:
        procstat = open('/proc/%d/status' % os.getpid()).readlines()
        out.write(' Process Information:\n')
        out.write('  ' + '  '.join(procstat))
    finally:
        pass

    if filename is None:
        return out.getvalue()

if __name__ == '__main__':
    print sysinfo()
