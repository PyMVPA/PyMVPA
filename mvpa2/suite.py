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

  from mvpa2.suite import *

or

  import mvpa2.suite

"""

__docformat__ = 'restructuredtext'


from mvpa2 import *

if __debug__ and 'SUITE' in debug.active:
    __sdebug = lambda msg: debug('SUITE', "%s" % msg)
else:
    __sdebug = lambda *args: None
__sdebug.__doc__ = "Shortcut to output debug messages for suite imports"

__sdebug('base')
from mvpa2.base import *
from mvpa2.base.attributes import *
from mvpa2.base.collections import *
from mvpa2.base.constraints import *
from mvpa2.base.config import *
from mvpa2.base.dataset import *
from mvpa2.base.externals import *
from mvpa2.base.info import *
from mvpa2.base.types import *
from mvpa2.base.verbosity import *
from mvpa2.base.param import *
from mvpa2.base.state import *
from mvpa2.base.node import *
from mvpa2.base.learner import *
from mvpa2.base.progress import *

__sdebug('h5py')
if externals.exists('h5py'):
    from mvpa2.base.hdf5 import *

__sdebug('reportlab')
if externals.exists('reportlab'):
    from mvpa2.base.report import *
else:
    from mvpa2.base.report_dummy import Report

__sdebug('algorithms')
from mvpa2.algorithms.hyperalignment import *
if externals.exists('scipy'):
    # Some pieces do not demand scipy, but for now let's just do this way
    from mvpa2.algorithms.searchlight_hyperalignment import *
    from mvpa2.algorithms.group_clusterthr import *

__sdebug('clfs')
from mvpa2 import clfs
__sdebug('clfs distance')
from mvpa2.clfs.distance import *
__sdebug('clfs base')
from mvpa2.clfs.base import *
__sdebug('clfs meta')
from mvpa2.clfs.meta import *
__sdebug('clfs kNN')
from mvpa2.clfs.knn import *
__sdebug('clfs lars')
if externals.exists('lars'):
    from mvpa2.clfs.lars import *
__sdebug('clfs enet')
if externals.exists('elasticnet'):
    from mvpa2.clfs.enet import *
__sdebug('clfs glmnet')
if externals.exists('glmnet'):
    from mvpa2.clfs.glmnet import *
__sdebug('clfs skl')
if externals.exists('skl'):
    if externals.versions['skl'] >= '0.9':
        import sklearn as skl
    else:
        import scikits.learn as skl
    from mvpa2.clfs.skl import *
__sdebug('clfs smlr')
from mvpa2.clfs.smlr import *
from mvpa2.clfs.blr import *
from mvpa2.clfs.gnb import *
from mvpa2.clfs.stats import *
from mvpa2.clfs.similarity import *
if externals.exists('libsvm') or externals.exists('shogun'):
    __sdebug('clfs svm')
    from mvpa2.clfs.svm import *
from mvpa2.clfs.transerror import *
__sdebug('clfs warehouse')
from mvpa2.clfs.warehouse import *

__sdebug('kernels')
from mvpa2 import kernels
from mvpa2.kernels.base import *
from mvpa2.kernels.np import *
if externals.exists('libsvm'):
    from mvpa2.kernels.libsvm import *
if externals.exists('shogun'):
    from mvpa2.kernels.sg import *

__sdebug('datasets')
from mvpa2 import datasets
from mvpa2.datasets import *
# just to make testsuite happy
from mvpa2.datasets.base import *
from mvpa2.datasets.formats import *
from mvpa2.datasets.miscfx import *
from mvpa2.datasets.eep import *
from mvpa2.datasets.eventrelated import *
if externals.exists('nibabel') :
    from mvpa2.datasets.mri import *
    from mvpa2.datasets.gifti import map2gifti, gifti_dataset
from mvpa2.datasets.sources import *
from mvpa2.datasets.sources.native import *
from mvpa2.datasets.sources.bids import *
from mvpa2.datasets.sources.openfmri import *
from mvpa2.datasets import niml
from mvpa2.datasets.niml import from_niml, to_niml
from mvpa2.datasets import eeglab
from mvpa2.datasets.eeglab import eeglab_dataset
if externals.exists('scipy') :
    from mvpa2.datasets import cosmo
    from mvpa2.datasets.cosmo import map2cosmo, cosmo_dataset, \
                                     CosmoQueryEngine, CosmoSearchlight


__sdebug('generators')
from mvpa2.generators.base import *
from mvpa2.generators.partition import *
from mvpa2.generators.splitters import *
from mvpa2.generators.permutation import *
from mvpa2.generators.resampling import *

__sdebug('featsel')
from mvpa2 import featsel
from mvpa2.featsel.base import *
from mvpa2.featsel.helpers import *
from mvpa2.featsel.ifs import *
from mvpa2.featsel.rfe import *

__sdebug('mappers')
from mvpa2 import mappers
#from mvpa2.mappers import *
from mvpa2.mappers.base import *
from mvpa2.mappers.slicing import *
from mvpa2.mappers.flatten import *
from mvpa2.mappers.shape import *
from mvpa2.mappers.prototype import *
from mvpa2.mappers.projection import *
from mvpa2.mappers.staticprojection import *
from mvpa2.mappers.svd import *
from mvpa2.mappers.procrustean import *
from mvpa2.mappers.boxcar import *
from mvpa2.mappers.fx import *
from mvpa2.mappers.fxy import *
from mvpa2.mappers.som import *
from mvpa2.mappers.zscore import *
if externals.exists('scipy'):
    from mvpa2.mappers.detrend import *
    from mvpa2.mappers.filters import *
if externals.exists('mdp'):
    from mvpa2.mappers.mdp_adaptor import *
if externals.exists('mdp ge 2.4'):
    from mvpa2.mappers.lle import *
from mvpa2.mappers.glm import *
from mvpa2.mappers.skl_adaptor import *

__sdebug('measures')
from mvpa2 import measures
from mvpa2.measures.anova import *
if externals.exists('statsmodels'):
    from mvpa2.measures.statsmodels_adaptor import *
from mvpa2.measures.irelief import *
from mvpa2.measures.base import *
from mvpa2.measures.fx import *
from mvpa2.measures.noiseperturbation import *
from mvpa2.misc.neighborhood import *
from mvpa2.measures.searchlight import *
from mvpa2.measures.gnbsearchlight import *
from mvpa2.measures.nnsearchlight import *
from mvpa2.measures.corrstability import *
from mvpa2.measures.winner import *

__sdebug('misc')
from mvpa2.support.copy import *
from mvpa2.misc.fx import *
from mvpa2.misc.attrmap import *
from mvpa2.misc.errorfx import *
from mvpa2.misc.cmdline import *
from mvpa2.misc.data_generators import *
from mvpa2.misc.exceptions import *
from mvpa2.misc import *
from mvpa2.misc.io import *
from mvpa2.misc.io.base import *
from mvpa2.misc.io.meg import *
from mvpa2.misc.fsl import *
from mvpa2.misc.bv import *
from mvpa2.misc.bv.base import *
from mvpa2.misc.support import *
from mvpa2.misc.transformers import *
from mvpa2.misc.dcov import dCOV, dcorcoef

__sdebug("nibabel")
if externals.exists("nibabel"):
    from mvpa2.misc.fsl.melodic import *

if externals.exists("pylab"):
    from mvpa2.viz import *
    from mvpa2.misc.plot import *
    from mvpa2.misc.plot.erp import *
    if externals.exists(['griddata', 'scipy']):
        from mvpa2.misc.plot.topo import *
    from mvpa2.misc.plot.lightbox import plot_lightbox

    if externals.exists(['matplotlib', 'griddata']):
        from mvpa2.misc.plot.flat_surf import \
                FlatSurfacePlotter, curvature_from_any

__sdebug("scipy dependents")
if externals.exists("scipy"):
    from mvpa2.support.scipy.stats import scipy
    from mvpa2.measures.corrcoef import *
    from mvpa2.measures.rsa import *
    from mvpa2.clfs.ridge import *
    from mvpa2.clfs.plr import *
    from mvpa2.misc.stats import *
    from mvpa2.clfs.gpr import *
    from mvpa2.support.nipy import *

__sdebug("mappers wavelet")
if externals.exists("pywt"):
    from mvpa2.mappers.wavelet import *

__sdebug("pylab")
if externals.exists("pylab"):
    import pylab as pl

__sdebug("atlases")
if externals.exists("lxml") and externals.exists("nibabel"):
    from mvpa2.atlases import *

__sdebug("surface searchlight")
from mvpa2.misc.surfing.queryengine import SurfaceVerticesQueryEngine, \
                                           SurfaceVoxelsQueryEngine, \
                                           SurfaceQueryEngine, \
                                           disc_surface_queryengine

from mvpa2.misc.surfing import surf_voxel_selection, volgeom, \
                                volsurf, volume_mask_dict

from mvpa2.misc.surfing.volume_mask_dict import VolumeMaskDictionary

__sdebug("nibabel afni")
from mvpa2.support.nibabel import afni_niml_dset, afni_suma_1d, \
                                  afni_suma_spec, surf_fs_asc, surf, \
				                  surf_caret, \
                                  afni_niml_roi, afni_niml_annot
if externals.exists('nibabel'):
    from mvpa2.support.nibabel import surf_gifti


__sdebug("ipython goodies")
if externals.exists("running ipython env"):
    try:
        from mvpa2.support.ipython import *
        ipy_activate_pymvpa_goodies()
    except Exception, e:
        warning("Failed to activate custom IPython completions due to %s" % e)

def suite_stats(scope_dict=None):
    """Return cruel dict of things which evil suite provides
    """
    if scope_dict is None:
        scope_dict = {}

    scope_dict = scope_dict or globals()
    import types
    # Compatibility layer for Python3
    try:
        from io import FileIO as BuiltinFileType
    except ImportError:
        BuiltinFileType = types.FileType

    try:
        from types import ClassType as OldStyleClassType
    except ImportError:
        OldStyleClassType = type(None)

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
            if isinstance(e.__module__, str):
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
                    (list, "lists", False),
                    (str, "strings", False),
                    (unicode, "strings", False),
                    (BuiltinFileType, "files", False),
                    (types.BuiltinFunctionType, None, True),
                    (types.BuiltinMethodType, None, True),
                    (types.ModuleType, "modules", True),
                    (OldStyleClassType, "classes", True),
                    (type, "types", True),
                    (types.LambdaType, "functions", True),
                    (object, "objects", True),
                    ):
                    if isinstance(e, ty):
                        found = True
                        if tk is None:
                            break
                        if not tk in self:
                            self[tk] = {}
                        if check_path:
                            mpath = _get_path(e)
                            if mvpa_str in mpath or mpath.startswith('mvpa2.'):
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
                        s += ": " + doc[:min(ind, 80)]
                    except:
                        pass
            return s

    return EnvironmentStatistics(scope_dict)

__sdebug("THE END of mvpa2.suite imports")
