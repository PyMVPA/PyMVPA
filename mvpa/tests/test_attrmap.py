# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from numpy.testing import assert_array_equal

from mvpa.misc.attrmap import AttributeMap


def test_attrmap():
    map_default = {'eins': 0, 'zwei': 2, 'sieben': 1}
    map_custom = {'eins': 11, 'zwei': 22, 'sieben': 33}
    literal = ['eins', 'zwei', 'sieben', 'eins', 'sieben', 'eins']
    num_default = [0, 2, 1, 0, 1, 0]
    num_custom = [11, 22, 33, 11, 33, 11]

    # no custom mapping given
    am = AttributeMap()
    assert_array_equal(am.to_numeric(literal), num_default)
    assert_array_equal(am.to_literal(num_default), literal)

    # with custom mapping
    am = AttributeMap(map=map_custom)
    assert_array_equal(am.to_numeric(literal), num_custom)
    assert_array_equal(am.to_literal(num_custom), literal)

#def testLabelsMapping(self):
#    """Test mapping of the labels from strings to numericals
#    """
#    od = {'apple':0, 'orange':1}
#    samples = [[3], [2], [3]]
#    labels_l = ['apple', 'orange', 'apple']
#
#    # test broadcasting of the label
#    ds = Dataset(samples=samples, labels='orange')
#    self.failUnless(N.all(ds.labels == ['orange']*3))
#
#    # Test basic mapping of litteral labels
#    for ds in [Dataset(samples=samples, labels=labels_l, labels_map=od),
#               # Figure out mapping
#               Dataset(samples=samples, labels=labels_l, labels_map=True)]:
#        self.failUnless(N.all(ds.labels == [0, 1, 0]))
#        self.failUnless(ds.labels_map == od)
#        ds_ = ds[1]
#        self.failUnless(ds_.labels_map == od,
#            msg='selectSamples should provide full mapping preserved')
#
#    # We should complaint about insufficient mapping
#    self.failUnlessRaises(ValueError, Dataset, samples=samples,
#        labels=labels_l, labels_map = {'apple':0})
#
#    # Conformance to older behavior -- if labels are given in
#    # strings, no mapping occur by default
#    ds2 = Dataset(samples=samples, labels=labels_l)
#    self.failUnlessEqual(ds2.labels_map, None)
#
#    # We should label numerical labels if it was requested:
#    od3 = {1:100, 2:101, 3:100}
#    ds3 = Dataset(samples=samples, labels=[1, 2, 3],
#                  labels_map=od3)
#    self.failUnlessEqual(ds3.labels_map, od3)
#    self.failUnless(N.all(ds3.labels == [100, 101, 100]))
#
#    ds3_ = ds3[1]
#    self.failUnlessEqual(ds3.labels_map, od3)
#
#    ds4 = Dataset(samples=samples, labels=labels_l)
#
#    # Lets check setting the labels map
#    ds = Dataset(samples=samples, labels=labels_l, labels_map=od)
#
#    self.failUnlessRaises(ValueError, ds.setLabelsMap,
#                          {'orange': 1, 'nonorange': 3})
#    new_map = {'tasty':0, 'crappy':1}
#    ds.labels_map = new_map.copy()
#    self.failUnlessEqual(ds.labels_map, new_map)


