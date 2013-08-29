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
def check_hrf_estimate(noise_level, cheating_start, jitter):
    # a very simple test for now -- single condition, high SNR, not
    # that much of overlap, matching HRF
    onsets1 = np.arange(0, 120, 6)
    intensities1 = np.random.uniform(1, 3, len(onsets1))
    # jitter design a bit
    if jitter:
        onsets1 += np.random.uniform(0, 6, size=onsets1.shape) - 3
        onsets1 = np.clip(onsets1, 0, 1000)
    # finally generate the events
    events = generate_events(onsets1, intensity=intensities1, target='L1')

    hrf_gen, hrf_est = double_gamma_hrf, double_gamma_hrf
    tr = 2.
    tres = 1.  # t resolution for HRF etc
    # even 0.5 is sufficient to make it converge to some "interesting" results
    # where even if I betas0 are provided, matching original intensities and
    # hrf_gen is the used canonical -- estimated betas are quite far away.
    # Filed an issue: https://github.com/fabianp/hrf_estimation/issues/4
    fir_length = 20
    baseline = 100

    data = simple_hrf_dataset(events, hrf_gen=hrf_gen,
                              fir_length=int(fir_length*tr/tres),
                              tr=tr, noise_level=noise_level, baseline=baseline)

    # 10 would be 20sec at tr=2.
    rank_one_kwargs = dict(
        #v0=intensities1,
        #alpha=0.00100,
    )
    if cheating_start:
        rank_one_kwargs['v0'] = intensities1

    he = HRFEstimator({'cond1': {'onsets': onsets1}},
                       tr,
                       hrf_gen=hrf_est,
                       fir_length=fir_length,
                       rank_one_kwargs=rank_one_kwargs,
                       enable_ca=['all'])
    hrfsds = he(data)
    betas = he.ca.betas

    assert_array_equal(betas.sa['groups'].unique, ['cond1'])
    assert_array_equal(betas.sa['indexes'].unique, np.arange(len(onsets1)))

    """
    import pylab as pl; pl.scatter(intensities1, betas.samples[:, 0]); pl.show()
    """
    
    # baseline should be estimated more or less correct
    assert_array_lequal(np.abs(he.ca.nuisances.samples - baseline),
                        0.1 + noise_level * 2)
    # how well this reconstructs voxel1 with the signal?
    data_rec = simple_hrf_dataset(
        generate_events(onsets1, intensity=betas.samples[:, 0]),
        hrf_gen=hrfsds.samples[:, 0],
        fir_length=fir_length,
        tr=tr,
        tres=tr,                          # those should both correspond to TR for this reconstruction since HRF is already computed at TR
        noise_level=0,
        baseline=he.ca.nuisances.samples[0,0])

    data_clean = data.samples - data.sa.noise
    # for possible retrospection
    designed_data = he.ca.designed_data if he.ca.is_set('designed_data') else None

    def plot_results():
        """Helper for a possible introspection of the results
        """
        import pylab as pl;
        pl.figure(); pl.plot(data.samples[:, 0], label='noisy data'); pl.plot((data.samples - data.sa.noise)[:, 0], label='clean data'); pl.plot(data_rec.samples[:,0], label='reconstructed');
        if designed_data is not None:
            pl.plot(designed_data[:] + baseline, label='designed + %d' % baseline);
        pl.legend();
        pl.figure(); time_x = np.arange(0, fir_length*tr, tr);  pl.plot(time_x, hrf_gen(time_x), label='original %.2f' % np.linalg.norm(hrf_gen(time_x))); pl.plot(time_x, hrfsds.samples[:, 0], label='estimated %.2f' % np.linalg.norm(hrfsds.samples[:, 0])); pl.legend();
        pl.figure(); pl.scatter(intensities1, betas[:, 0]); pl.xlabel('original'); pl.ylabel('estimated');
        pl.show()

    if he.ca.is_set('designed_data'):
        # must be nearly identical
        assert_greater(np.corrcoef((data_clean[:, 0], designed_data))[0, 1], 0.98 - 0.1*int(jitter))

    cc_rec = np.corrcoef((data_clean[:, 0], data_rec.samples[:, 0]))[0, 1]
    assert_greater(cc_rec, 0.9-noise_level/2)

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
    assert_true(0.8 - noise_level < cc[0, 2] <= 1)
    if noise_level < 0.2:
        # for bogus feature it should correlate more with canonical than v0
        assert_greater(cc[1, 2], cc[1, 0])

        # voxel1 is not informative and no HRF could be deduced so it would stay
        # at canonical and with high cc if noise level was low
        assert_greater(cc[1, 2], 0.8)

    cc_betas = np.corrcoef(np.hstack((betas.samples, intensities1[:, None])),
                           rowvar=0)
    if not cheating_start:
        # there should be no correlation between betas of informative
        # voxel and noisy one
        assert_greater(0.4, cc_betas[0, 1])
        # neither to original
        assert_greater(0.4, cc_betas[1, 2])
    # but estimates for a good voxel should have reasonably high correlation
    # yoh: blunt heuristic here for the comparison
    assert_greater(cc_betas[0, 2], 0.9-noise_level * 1.8)

    # provide nuisance_sas pointing to originally added noise
    he.nuisance_sas = ['noise']
    hrfsds_ = he(data)
    betas_ = he.ca.betas
    cc_ = np.corrcoef(np.hstack((hrfsds_, canonical[:, None])), rowvar=0)
    # Fidelity should be higher if we provide original noise as
    # nuisances but we will leave some 1e-4 margin for being wrong due
    # to numeric precision etc
    assert_array_less(cc, cc_ + 1e-4)
    # results should be really close to the underlying data -- we provided everything!
    cc_betas_ = np.corrcoef(np.hstack((betas_.samples, intensities1[:, None])),
                            rowvar=0)
    assert_greater(cc_betas_[0, 2], 0.99)

    #print np.linalg.norm(hrfsds.samples[:, 1] - canonical, 'fro')
    # voxel1 has no information
    # import pydb; import pylab as pl;  pydb.debugger()
    # i = 1

def test_hrf_estimate():
    for nl in [0, 0.5, 0.8]:
        for cheating_start in (True, False):
            for jitter in (False, ): # True):
                yield check_hrf_estimate, nl, cheating_start, jitter
