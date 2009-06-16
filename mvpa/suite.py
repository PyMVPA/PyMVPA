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
from mvpa.base.config import *
from mvpa.base.verbosity import *

from mvpa.algorithms.cvtranserror import *

from mvpa import clfs
from mvpa.clfs.distance import *
from mvpa.clfs.kernel import *
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
from mvpa.clfs.stats import *
if externals.exists('libsvm') or externals.exists('shogun'):
    from mvpa.clfs.svm import *
from mvpa.clfs.transerror import *
from mvpa.clfs.warehouse import *

from mvpa import datasets
from mvpa.datasets import *
# just to make testsuite happy
from mvpa.datasets.base import *
from mvpa.datasets.meta import *
from mvpa.datasets.masked import *
from mvpa.datasets.miscfx import *
from mvpa.datasets.channel import *
from mvpa.datasets.event import *
from mvpa.datasets.eep import *
if externals.exists('nifti'):
    from mvpa.datasets.nifti import *

from mvpa.datasets import splitters
from mvpa.datasets.splitters import *

from mvpa import featsel
from mvpa.featsel.base import *
from mvpa.featsel.helpers import *
from mvpa.featsel.ifs import *
from mvpa.featsel.rfe import *

from mvpa import mappers
#from mvpa.mappers import *
from mvpa.mappers.base import *
from mvpa.mappers.metric import *
from mvpa.mappers.mask import *
from mvpa.mappers.svd import *
from mvpa.mappers.procrustean import *
from mvpa.mappers.boxcar import *
from mvpa.mappers.samplegroup import *
from mvpa.mappers.som import *
from mvpa.mappers.array import *
if externals.exists('scipy'):
    from mvpa.mappers.zscore import ZScoreMapper
if externals.exists('mdp'):
    from mvpa.mappers.pca import *
    from mvpa.mappers.ica import *
if externals.exists('mdp >= 2.4'):
    from mvpa.mappers.lle import *

from mvpa import measures
from mvpa.measures.anova import *
from mvpa.measures.glm import *
from mvpa.measures.irelief import *
from mvpa.measures.base import *
from mvpa.measures.noiseperturbation import *
from mvpa.measures.searchlight import *
from mvpa.measures.splitmeasure import *
from mvpa.measures.corrstability import *

from mvpa.support.copy import *
from mvpa.misc.fx import *
from mvpa.misc.errorfx import *
from mvpa.misc.cmdline import *
from mvpa.misc.data_generators import *
from mvpa.misc.exceptions import *
from mvpa.misc import *
from mvpa.misc.io import *
from mvpa.misc.io.eepbin import *
from mvpa.misc.io.meg import *
if externals.exists('cPickle') and externals.exists('gzip'):
    from mvpa.misc.io.hamster import *
from mvpa.misc.fsl import *
from mvpa.misc.bv import *
from mvpa.misc.bv.base import *
from mvpa.misc.param import *
from mvpa.misc.state import *
from mvpa.misc.support import *
from mvpa.misc.transformers import *

if externals.exists("nifti"):
    from mvpa.misc.fsl.melodic import *

if externals.exists("pylab"):
    from mvpa.misc.plot import *
    from mvpa.misc.plot.erp import *
    if externals.exists(['griddata', 'scipy']):
        from mvpa.misc.plot.topo import *
    if externals.exists('nifti'):
        from mvpa.misc.plot.mri import plotMRI

if externals.exists("scipy"):
    from mvpa.measures.corrcoef import *
    from mvpa.measures.ds import *
    from mvpa.clfs.ridge import *
    from mvpa.clfs.plr import *
    from mvpa.misc.stats import *
    from mvpa.clfs.gpr import *

if externals.exists("pywt"):
    from mvpa.mappers.wavelet import *

if externals.exists("pylab"):
    import pylab as P

if externals.exists("lxml") and externals.exists("nifti"):
    from mvpa.atlases import *
