#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*- 
#ex: set sts=4 ts=4 sw=4 noet:
"""

 COPYRIGHT: Yaroslav Halchenko 2013

 LICENSE: MIT

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
  THE SOFTWARE.
"""

__author__ = 'Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2013 Yaroslav Halchenko'
__license__ = 'MIT'

from glm_features import *

from mvpa2.misc.data_generators import simple_hrf_dataset
from mvpa2.misc.fx import double_gamma_hrf

from mvpa2.testing import *

def test_regroup():
    evs = {'a': [1, 2],
           'b': [3, 4],
           'c': [5]}
    assert_equal(regroup_conditions(evs, {'g1': ['a', 'c']}),
                 {'b': [3, 4], 'g1': [1, 2, 5]})
    # no inplace modifications
    assert_equal(sorted(evs.keys()), ['a', 'b', 'c'])
    assert_equal(regroup_conditions(evs, {'g1': ['a']}),
                 {'b': [3, 4], 'g1': [1, 2], 'c': [5]})
    assert_raises(KeyError, regroup_conditions, evs, {'g1': ['x']})


def test_bunch_to_evs():
    from nipype.interfaces.base import Bunch

    b = Bunch(conditions=['cond1', 'cond2'],
              onsets=[[20, 120], [80, 160]],
              durations=[[0], [0]])
    evs, regrs = bunch_to_evs(b)
    assert_equal(regrs, None)
    assert_equal(evs, {'cond1': {'onsets': [20, 120], 'durations': [0]},
                       'cond2': {'onsets': [80, 160], 'durations': [0]}})

    b = Bunch(conditions=['cond1', 'cond2'],
              onsets=[[20, 120], [80, 160]],
              durations=[[0, 0], [0, 2]],
              regressor_names=['r1', 'r2'],
              regressors=[[0, 1, 2],
                          [0, 2 ,1]])
    evs, regrs = bunch_to_evs(b)
    assert_equal(regrs, {'r1': [0, 1, 2], 'r2': [0, 2, 1]})
    assert_equal(evs, {'cond1': {'onsets': [20, 120], 'durations': [0, 0]},
                       'cond2': {'onsets': [80, 160], 'durations': [0, 2]}})

from mvpa2.misc.support import Event
def generate_events(onset, **kwargs):
    opts = {'onset': onset}
    for k,x in kwargs.iteritems():
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            x = [x]*len(onset)
        opts[k] = x
    return [Event(**dict([(k, opts[k][i]) for k in opts.keys()]))
            for i in xrange(len(onset))]

@reseed_rng()
def test_hrf_estimate():
    # a very simple test for now -- single condition, high SNR, not
    # that much of overlap, matching HRF
    onsets1 = np.arange(0, 120, 6)
    intensities1 = np.random.uniform(1, 3, len(onsets1))
    events = generate_events(onsets1, intensity=intensities1, target='L1')
    # jitter them a bit
    #onsets1 += np.random.uniform(0, 8, size=onsets1.shape) - 4
    #onsets1 = np.clip(onsets1, 0, 1000)
    hrf_gen, hrf_est = double_gamma_hrf, double_gamma_hrf
    tr = 2.
    # even 0.5 is sufficient to make it converge to some "interesting" results
    # where even if I betas0 are provided, matching original intensities and
    # hrf_gen is the used canonical -- estimated betas are quite far away.
    # Filed an issue: https://github.com/fabianp/hrf_estimation/issues/4
    noise_level = 0.1
    fir_length = 20

    data = simple_hrf_dataset(events, hrf_gen=hrf_gen,
                              tr=tr, noise_level=noise_level, baseline=0)

    # 10 would be 20sec at tr=2.
    he = HRFEstimator({'cond1': {'onsets': onsets1}}, tr,
                       hrf_gen=hrf_est,
                       fir_length=fir_length,
                       # nuisance_sas = ['noise'],
                       # betas0=intensities1,
                       enable_ca=['all'])
    hrfsds = he(data)
    betas = he.ca.betas
    """
    import pylab as pl; pl.scatter(intensities1, betas.samples[:, 0]); pl.show()
    """
    # how well this reconstructs voxel1 with the signal?
    data_rec = simple_hrf_dataset(
        generate_events(onsets1, intensity=betas.samples[:, 0]),
        hrf_gen=double_gamma_hrf, tr=tr, noise_level=0, baseline=0)
    
    """
    import pylab as pl; pl.plot(data.samples[:, 0], label='noisy data'); pl.plot((data.samples - data.sa.noise)[:, 0], label='clean data'); pl.plot(data_rec.samples[:,0], label='reconstructed'); pl.legend(); pl.show()
    """
    cc_rec = np.corrcoef(((data.samples - data.sa.noise)[:, 0],
                           data_rec.samples[:, 0]))[0, 1]
    assert_greater(cc_rec, 0.8)

    assert_equal(len(hrfsds), fir_length)
    assert_almost_equal(hrfsds.sa.time_coords[1]-hrfsds.sa.time_coords[0], tr)

    # Basic tests
    assert_equal(he.ca.betas.shape, (len(onsets1), data.nfeatures))
    assert_equal(he.ca.design.shape, (len(data), fir_length*len(onsets1)))

    assert_true(hrfsds.fa.signal_level[0])
    assert_false(hrfsds.fa.signal_level[1])

    canonical = hrf_gen(hrfsds.sa.time_coords)
    cc = np.corrcoef(np.hstack((hrfsds, canonical[:, None])), rowvar=0)
    # voxel0 is informative one and its estimate would become a bit noisier
    # version of canonical HRF but still quite high
    assert_true(0.7 < cc[0, 2] < 1)
    # for bogus feature it should correlate more with canonical than v0
    assert_greater(cc[1, 2], cc[1, 0])
    # voxel1 is not informative and no HRF could be deduced so it would stay
    # at canonical and with high cc
    assert_greater(cc[1, 2], 0.8)

    cc_betas = np.corrcoef(np.hstack((betas.samples, intensities1[:, None])),
                           rowvar=0)
    # there should be no correlation between betas of informative
    # voxel and noisy one
    assert_greater(0.4, cc_betas[0, 1])
    # neight to original
    assert_greater(0.4, cc_betas[1, 2])
    # but estimates for a good voxel should have reasonably high correlation
    assert_greater(cc_betas[0, 2], 0.6)

    # provide nuisance_sas pointing to originally added noise
    he.nuisance_sas = ['noise']
    hrfsds_ = he(data)
    betas_ = he.ca.betas
    cc_ = np.corrcoef(np.hstack((hrfsds_, canonical[:, None])), rowvar=0)
    # Fidelity should be higher if we provide original noise as
    # nuisances but we will leave some 1e-4 margin for being wrong due
    # to numeric precision etc
    assert_array_less(cc, cc_ + 1e-4)
    # results should be even better match then before
    cc_betas_ = np.corrcoef(np.hstack((betas.samples, intensities1[:, None])),
                            rowvar=0)


    #print np.linalg.norm(hrfsds.samples[:, 1] - canonical, 'fro')
    # voxel1 has no information
    #    import pydb; import pylab as pl;  pydb.debugger()
    i = 1

    """
            #pl.imshow(design2)
        import pylab as pl; pl.plot(data.samples[:, 0], label='noisy data'); pl.plot((data.samples - data.sa.noise)[:, 0], label='clean data'); pl.plot(data_rec.samples[:,0], label='reconstructed'); pl.legend(); pl.show()

        #pl.show()
        print V
        pl.figure();
        pl.plot(dataset.samples[:, 0]); pl.plot(dataset.samples[:, 1]); pl.legend(('v1', 'v2'))
        pl.figure();
        pl.plot(hrfsds[:, 0]); pl.plot(hrfsds[:, 1]); pl.plot(canonical); pl.legend(('v1', 'v2', 'canonical')); pl.show()

        import pydb; pydb.debugger()
        i = 1
        return design
        # in case of later 
"""
