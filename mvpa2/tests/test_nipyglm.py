# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for the NiPy GLM mapper (requiring NiPy)."""

import numpy as np

from mvpa2.testing.tools import *

skip_if_no_external('nipy')
skip_if_no_external('scipy')

from scipy import signal
from mvpa2.datasets import Dataset
from mvpa2.mappers.nipy_glm import NiPyGLMMapper
from mvpa2.misc.fx import double_gamma_hrf, single_gamma_hrf

def get_bold():
    # TODO add second model
    hrf_x = np.linspace(0,25,250)
    hrf = double_gamma_hrf(hrf_x) - single_gamma_hrf(hrf_x, 0.8, 1, 0.05)

    samples = 1200
    exp_time = np.linspace(0, 120, samples)

    fast_er_onsets = np.array([50, 240, 340, 590, 640, 940, 960])
    fast_er = np.zeros(samples)
    fast_er[fast_er_onsets] = 1

    model_hr = np.convolve(fast_er, hrf)[:samples]

    tr = 2.0
    model_lr = signal.resample(model_hr, int(samples / tr / 10), window='ham')

    ## moderate noise level
    baseline = 800
    wsignal = baseline + 8.0 \
              * model_lr + np.random.randn(int(samples / tr / 10)) * 4.0
    nsignal = baseline \
              + np.random.randn(int(samples / tr / 10)) * 4.0

    ds = Dataset(samples=np.array([wsignal, nsignal]).T,
                 sa={'model': model_lr})

    return ds

def test_glm_mapper():
    bold = get_bold()
    assert_equal(bold.nfeatures, 2)
    assert('model' in bold.sa)
    reg_names = ['model']
    pest = NiPyGLMMapper(reg_names)(bold)
    assert_equal(pest.shape, (len(reg_names), bold.nfeatures))
    assert_array_equal(pest.sa.regressor_names, reg_names)
    pest = NiPyGLMMapper(reg_names, add_constant=True)(bold)
    assert_equal(pest.shape, (len(reg_names) + 1, bold.nfeatures))
    # nothing at all
    noglm = NiPyGLMMapper([])
    assert_raises(ValueError, noglm.__call__, bold)
    # no reg from ds at all
    pest = NiPyGLMMapper([], add_constant=True)(bold)
    assert_equal(pest.shape, (1, bold.nfeatures))
    assert_array_equal(pest.sa.regressor_names, ['constant'])
    # only reg from mapper
    pest = NiPyGLMMapper([],
                         add_regs=(('trend',
                                   (np.linspace(-1,1,len(bold)))),))(bold)
    assert_equal(pest.shape, (1, bold.nfeatures))
    assert_array_equal(pest.sa.regressor_names, ['trend'])
    # full monty
    pest = NiPyGLMMapper(['model'],
                         add_regs=(('trend',
                                   (np.linspace(-1,1,len(bold)))),),
                         add_constant=True,
                         space='conditions',
                         return_design=True,
                         return_model=True,
                         glmfit_kwargs=dict(model='ols'))(bold)
    assert_equal(pest.shape, (len(reg_names) + 2, bold.nfeatures))
    assert_array_equal(pest.sa.conditions, ['model', 'trend', 'constant'])
    assert('model' in pest.a)
    assert('regressors' in pest.sa)
    assert_array_equal(pest.sa.regressors[0], bold.sa.model)
    assert_array_equal(pest.sa.regressors[-1], np.ones(len(bold)))


