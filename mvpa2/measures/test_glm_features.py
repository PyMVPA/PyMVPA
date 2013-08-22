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

from nose.tools import *
from glm_features import *

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

