#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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


from mvpa.algorithms.cvtranserror import *
from mvpa.base import *
from mvpa.clfs.base import *
from mvpa.clfs.knn import *
if externals.exists('lars'):
    from mvpa.clfs.lars import *
from mvpa.clfs.plr import *
from mvpa.clfs.ridge import *
from mvpa.clfs.smlr import *
from mvpa.clfs.gpr import *
from mvpa.clfs.stats import *
if externals.exists('libsvm') or externals.exists('shogun'):
    from mvpa.clfs.svm import *
from mvpa.clfs.transerror import *
from mvpa.clfs.warehouse import *
from mvpa.datasets.base import *
from mvpa.datasets import *
from mvpa.datasets.maskeddataset import *
from mvpa.datasets.metric import *
from mvpa.datasets.miscfx import *
if externals.exists('nifti'):
    from mvpa.datasets.niftidataset import *
from mvpa.datasets.splitter import *
from mvpa.featsel.base import *
from mvpa.featsel.helpers import *
from mvpa.featsel.ifs import *
from mvpa.featsel.rfe import *
from mvpa.mappers import *
from mvpa.measures.anova import *
from mvpa.measures.corrcoef import *
from mvpa.measures.base import *
from mvpa.measures.noiseperturbation import *
from mvpa.measures.searchlight import *
from mvpa.measures.splitmeasure import *
from mvpa.misc.errorfx import *
from mvpa.misc.cmdline import *
from mvpa.misc.copy import *
from mvpa.misc.data_generators import *
from mvpa.misc.exceptions import *
from mvpa.misc import *
from mvpa.misc.iohelpers import *
from mvpa.misc.fsl import *
from mvpa.misc.bv import *
from mvpa.misc.param import *
from mvpa.misc.state import *
from mvpa.misc.stats import *
from mvpa.misc.support import *
from mvpa.misc.transformers import *
from mvpa.misc.verbosity import *

if externals.exists("pylab"):
	import pylab as P
