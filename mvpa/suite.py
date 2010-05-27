# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""MultiVariate Pattern Analysis -- load helper

If you don't like to specify exact location of any particular
functionality within PyMVPA, please simply::

  from mvpa.suite import *

or

  import mvpa.suite

"""

__docformat__ = 'restructuredtext'


from mvpa import *

from mvpa.base import *
from mvpa.base.collections import *
from mvpa.base.config import *
from mvpa.base.dataset import *
from mvpa.base.externals import *
from mvpa.base.info import *
from mvpa.base.types import *
from mvpa.base.verbosity import *
from mvpa.base.param import *
from mvpa.base.state import *

if externals.exists('h5py'):
    from mvpa.base.hdf5 import *

if externals.exists('reportlab'):
    from mvpa.base.report import *
else:
    from mvpa.base.report_dummy import Report


from mvpa.algorithms.cvtranserror import *
from mvpa.algorithms.hyperalignment import *

from mvpa import clfs
from mvpa.clfs.distance import *
from mvpa.clfs.base import *
from mvpa.clfs.meta import *
from mvpa.clfs.knn import *
if externals.exists('lars'):
    from mvpa.clfs.lars import *
if externals.exists('elasticnet'):
    from mvpa.clfs.enet import *
if externals.exists('glmnet'):
    from mvpa.clfs.glmnet import *
from mvpa.clfs.smlr import *
from mvpa.clfs.blr import *
from mvpa.clfs.gnb import *
from mvpa.clfs.stats import *
from mvpa.clfs.similarity import *
if externals.exists('libsvm') or externals.exists('shogun'):
    from mvpa.clfs.svm import *
from mvpa.clfs.transerror import *
from mvpa.clfs.warehouse import *

from mvpa import kernels
from mvpa.kernels.base import *
from mvpa.kernels.np import *
if externals.exists('libsvm'):
    from mvpa.kernels.libsvm import *
if externals.exists('shogun'):
    from mvpa.kernels.sg import *

from mvpa import datasets
from mvpa.datasets import *
# just to make testsuite happy
from mvpa.datasets.base import *
from mvpa.datasets.miscfx import *
from mvpa.datasets.eep import *
from mvpa.datasets.eventrelated import *
# safe to import since multiple externals are handled inside
from mvpa.datasets.mri import *
# make NiftiImage available for people
if externals.exists('nifti'):
    from nifti import NiftiImage

from mvpa.datasets import splitters
from mvpa.datasets.splitters import *

from mvpa.generators.partition import *
from mvpa.generators.splitters import *

from mvpa import featsel
from mvpa.featsel.base import *
from mvpa.featsel.helpers import *
from mvpa.featsel.ifs import *
from mvpa.featsel.rfe import *

from mvpa import mappers
#from mvpa.mappers import *
from mvpa.mappers.base import *
from mvpa.mappers.slicing import *
from mvpa.mappers.flatten import *
from mvpa.mappers.prototype import *
from mvpa.mappers.projection import *
from mvpa.mappers.svd import *
from mvpa.mappers.procrustean import *
from mvpa.mappers.boxcar import *
from mvpa.mappers.fx import *
from mvpa.mappers.som import *
from mvpa.mappers.zscore import *
if externals.exists('scipy'):
    from mvpa.mappers.detrend import *
    from mvpa.mappers.filters import *
if externals.exists('mdp'):
    from mvpa.mappers.mdp_adaptor import *
if externals.exists('mdp ge 2.4'):
    from mvpa.mappers.lle import *

from mvpa import measures
from mvpa.measures.anova import *
from mvpa.measures.glm import *
from mvpa.measures.irelief import *
from mvpa.measures.base import *
from mvpa.measures.noiseperturbation import *
from mvpa.misc.neighborhood import *
from mvpa.measures.searchlight import *
from mvpa.measures.gnbsearchlight import *
from mvpa.measures.corrstability import *

from mvpa.support.copy import *
from mvpa.misc.fx import *
from mvpa.misc.attrmap import *
from mvpa.misc.errorfx import *
from mvpa.misc.cmdline import *
from mvpa.misc.data_generators import *
from mvpa.misc.exceptions import *
from mvpa.misc import *
from mvpa.misc.io import *
from mvpa.misc.io.base import *
from mvpa.misc.io.meg import *
if externals.exists('cPickle') and externals.exists('gzip'):
    from mvpa.misc.io.hamster import *
from mvpa.misc.fsl import *
from mvpa.misc.bv import *
from mvpa.misc.bv.base import *
from mvpa.misc.support import *
from mvpa.misc.transformers import *

if externals.exists("nifti"):
    from mvpa.misc.fsl.melodic import *

if externals.exists("pylab"):
    from mvpa.misc.plot import *
    from mvpa.misc.plot.erp import *
    if externals.exists(['griddata', 'scipy']):
        from mvpa.misc.plot.topo import *
    from mvpa.misc.plot.lightbox import plot_lightbox

if externals.exists("scipy"):
    from mvpa.support.stats import scipy
    from mvpa.measures.corrcoef import *
    from mvpa.measures.ds import *
    from mvpa.clfs.ridge import *
    from mvpa.clfs.plr import *
    from mvpa.misc.stats import *
    from mvpa.clfs.gpr import *
    from mvpa.support.nipy import *

if externals.exists("pywt"):
    from mvpa.mappers.wavelet import *

if externals.exists("pylab"):
    import pylab as pl

if externals.exists("lxml") and externals.exists("nifti"):
    from mvpa.atlases import *


if externals.exists("running ipython env"):
    from mvpa.support.ipython import *
    ipy_activate_pymvpa_goodies()

def suite_stats():
    """Return cruel dict of things which evil suite provides
    """

    glbls = globals()
    import types

    def _get_path(e):
        """Figure out basic path for the beast... probably there is already smth which could do that for me
        """
        if str(e).endswith('(built-in)>'):
            return "BUILTIN"
        if hasattr(e, '__file__'):
            return e.__file__
        elif hasattr(e, '__path__'):
            return e.__path__[0]
        elif hasattr(e, '__module__'):
            if isinstance(e.__module__, types.StringType):
                return e.__module__
            else:
                return _get_path(e.__module__)
        elif hasattr(e, '__class__'):
            return _get_path(e.__class__)
        else:
            raise RuntimeError, "Could not figure out path for %s" % e


    class EnvironmentStatistics(dict):
        def __init__(self, d):
            dict.__init__(self, foreign={})
            # compute cruel stats
            mvpa_str = '%smvpa' % os.path.sep
            for k, e in d.iteritems():
                found = False
                for ty, tk, check_path in (
                    (types.ListType, "lists", False),
                    (types.StringType, "strings", False),
                    (types.UnicodeType, "strings", False),
                    (types.FileType, "files", False),
                    (types.BuiltinFunctionType, None, True),
                    (types.BuiltinMethodType, None, True),
                    (types.ModuleType, "modules", True),
                    (types.ClassType, "classes", True),
                    (types.TypeType, "types", True),
                    (types.LambdaType, "functions", True),
                    (types.ObjectType, "objects", True),
                    ):
                    if isinstance(e, ty):
                        found = True
                        if tk is None:
                            break
                        if not tk in self:
                            self[tk] = {}
                        if check_path:
                            mpath = _get_path(e)
                            if mvpa_str in mpath or mpath.startswith('mvpa.'):
                                self[tk][k] = e
                            else:
                                self['foreign'][k] = e
                        else:
                            self[tk][k] = e
                        break
                if not found:
                    raise ValueError, \
                          "Could not figure out placement for %s %s" % (k, e)

        def __str__(self):
            s = ""
            for k in sorted(self.keys()):
                s += "\n%s [%d entries]:" % (k, len(self[k]))
                for i in sorted(self[k].keys()):
                    s += "\n  %s" % i
                    # Lets extract first line in doc
                    try:
                        doc = self[k][i].__doc__.strip()
                        try:
                            ind = doc.index('\n')
                        except:
                            ind = 1000
                        s+= ": " + doc[:min(ind, 80)]
                    except:
                        pass
            return s

    return EnvironmentStatistics(globals())
