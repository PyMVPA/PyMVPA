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

import mvpa2
from mvpa2.base import externals, cfg
from mvpa2.base.dochelpers import borrowkwargs

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

__all__ = ['wtf', 'get_pymvpa_gitversion']


def get_pymvpa_gitversion():
    """PyMVPA version as reported by git.

    Returns
    -------
    None or str
      Version of PyMVPA according to git.
    """
    gitpath = os.path.join(os.path.dirname(mvpa2.__file__), os.path.pardir)
    gitpathgit = os.path.join(gitpath, '.git')
    if not os.path.exists(gitpathgit):
        return None
    ver = None
    try:
        (tmpd, tmpn) = mkstemp('mvpa', 'git')
        retcode = subprocess.call(['git',
                                   '--git-dir=%s' % gitpathgit,
                                   '--work-tree=%s' % gitpath,
                                   'describe', '--abbrev=4', 'HEAD'
                                   ],
                                  stdout=tmpd,
                                  stderr=subprocess.STDOUT)
        outline = open(tmpn, 'r').readlines()[0].strip()
        if outline.startswith('upstream/'):
            ver = outline.replace('upstream/', '')
    finally:
        os.remove(tmpn)
    return ver


class WTF(object):
    """Convenience class to contain information about PyMVPA and OS

    TODO: refactor to actually not contain just string representation
    but rather a dictionary (of dictionaries)
    """

    __knownitems__ = set(('process', 'runtime',
                          'externals', 'system', 'sources'))

    def __init__(self, include=None, exclude=None):
        """
        Parameters
        ----------
        include : list of str
          By default, all known (listed in `__knownitems__`) are reported.
          If only a limited set is needed to be reported -- specify it here.
        exclude : list of str
          If you want to exclude any item from being reported.
        """
        self._info = ''
        if include is None:
            report_items = self.__knownitems__.copy()
        else:
            # check first
            if not self.__knownitems__.issuperset(include):
                raise ValueError, \
                      "Items %s provided in exclude are not known to WTF." \
                      " Known are %s" % \
                      (str(set(include).difference(self.__knownitems__)),
                       self.__knownitems__)
            report_items = set(include)

        if exclude is not None:
            # check if all are known
            if not self.__knownitems__.issuperset(exclude):
                raise ValueError, \
                      "Items %s provided in exclude are not known to WTF." \
                      " Known are %s" % \
                      (str(set(exclude).difference(self.__knownitems__)),
                       self.__knownitems__)
            report_items = report_items.difference(exclude)
        self._report_items = report_items
        self._acquire()


    def _acquire_sources(self, out):
        out.write("PyMVPA:\n")
        out.write(" Version:       %s\n" % mvpa2.__version__)
        out.write(" Hash:          %s\n" % mvpa2.__hash__)
        out.write(" Path:          %s\n" % mvpa2.__file__)

        # Try to obtain git information if available
        out.write(" Version control (GIT):\n")
        try:
            gitpath = os.path.join(os.path.dirname(mvpa2.__file__), os.path.pardir)
            gitpathgit = os.path.join(gitpath, '.git')
            if os.path.exists(gitpathgit):
                for scmd, cmd in [
                    ('Status', ['status']),
                    ('Reference', 'show-ref -h HEAD'.split(' ')),
                    ('Difference from last release %s' % mvpa2.__version__,
                     ['diff', '--shortstat', 'upstream/%s...' % mvpa2.__version__])]:
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


    def _acquire_system(self, out):
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

    def _acquire_externals(self, out):
        # Test and list all dependencies:
        sdeps = {True: [], False: [], 'Error': []}
        for dep in sorted(externals._KNOWN):
            try:
                sdeps[externals.exists(dep, force=False)] += [dep]
            except:
                sdeps['Error'] += [dep]
        out.write('EXTERNALS:\n')
        out.write(' Present:       %s\n' % ', '.join(sdeps[True]))
        out.write(' Absent:        %s\n' % ', '.join(sdeps[False]))
        if len(sdeps['Error']):
            out.write(' Errors in determining: %s\n' % ', '.join(sdeps['Error']))

        SV = ('.__version__', )              # standard versioning
        out.write(' Versions of critical externals:\n')
        # First the ones known to externals,
        for k, v in sorted(externals.versions.iteritems()):
            out.write('  %-12s: %s\n' % (k, str(v)))
        try:
            if externals.exists('matplotlib'):
                import matplotlib
                out.write(' Matplotlib backend: %s\n'
                          % matplotlib.get_backend())
        except Exception, exc:
            out.write(' Failed to determine backend of matplotlib due to "%s"'
                      % str(exc))

    def _acquire_runtime(self, out):
        out.write("RUNTIME:\n")
        out.write(" PyMVPA Environment Variables:\n")
        out.write('  ' + '  '.join(
            ['%-20s: "%s"\n' % (str(k), str(v))
             for k, v in os.environ.iteritems()
             if (k.startswith('MVPA') or k.startswith('PYTHON'))]))

        out.write(" PyMVPA Runtime Configuration:\n")
        out.write('  ' + str(cfg).replace('\n', '\n  ').rstrip() + '\n')

    def _acquire_process(self, out):
        try:
            procstat = open('/proc/%d/status' % os.getpid()).readlines()
            out.write(' Process Information:\n')
            out.write('  ' + '  '.join(procstat))
        except:
            pass


    def _acquire(self):
        """
        TODO: refactor and redo ;)
        """
        out = StringIO()

        out.write("Current date:   %s\n" % time.strftime("%Y-%m-%d %H:%M"))

        # Little silly communicator/
        if 'sources' in self._report_items:
            self._acquire_sources(out)
        if 'system' in self._report_items:
            self._acquire_system(out)
        if 'externals' in self._report_items:
            self._acquire_externals(out)
        if 'runtime' in self._report_items:
            self._acquire_runtime(out)
        if 'process' in self._report_items:
            self._acquire_process(out)

        self._info = out.getvalue()


    def __repr__(self):
        if self._info is None:
            self._acquire()
        return self._info

    __str__ = __repr__


@borrowkwargs(WTF, '__init__')
def wtf(filename=None, **kwargs):
    """Report summary about PyMVPA and the system

    Parameters
    ----------
    filename : None or str
      If provided, information will be stored in a file, not printed
      to the screen
    **kwargs
      Passed to initialize `WTF` instance
    """

    info = WTF(**kwargs)
    if filename is not None:
        _ = open(filename, 'w').write(str(info))
    else:
        return info


if __name__ == '__main__':
    print wtf()
