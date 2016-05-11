# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for the BIDS I/O support'''

from mvpa2.testing import eq_

from StringIO import StringIO

from mvpa2.datasets.sources.bids import load_events


def test_load_events():
    evtsv = "onset\tduration"
    eq_(load_events(StringIO(evtsv)), [])
    ra = load_events(StringIO(evtsv), as_recarr=True)
    eq_(len(ra), 0)
    eq_(ra.dtype.names, ('onset', 'duration'))
    # now with content to do type checks
    evtsv = "onset\tduration\ttrial_type\n2\t1.3\tboring\n3.5\t4\texciting"
    ra = load_events(StringIO(evtsv))
    eq_(ra,
        [{'onset': 2.0, 'duration': 1.3, 'trial_type': 'boring'},
         {'onset': 3.5, 'duration': 4.0, 'trial_type': 'exciting'}])
