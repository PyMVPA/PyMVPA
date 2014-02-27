# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for commonly used data source formats.

"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals

import sys
import numpy as np

if __debug__:
    from mvpa2.base import debug

from mvpa2.datasets.base import Dataset
from mvpa2.base import warning
from mvpa2.misc.attrmap import AttributeMap

# LightSVM :

def to_lightsvm_format(dataset, out, targets_attr='targets',
                       domain=None, am=None):
    """Export dataset into LightSVM format

    Parameters
    ----------
    dataset : Dataset
    out
      Anything understanding .write(string), such as `File`
    targets_attr : string, optional
      Name of the samples attribute to be output
    domain : {None, 'regression', 'binary', 'multiclass'}, optional
      What domain dataset belongs to.  If `None`, it would be deduced
      depending on the datatype ('regression' if float, classification
      in case of int or string, with 'binary'/'multiclass' depending on
      the number of unique targets)
    am : `AttributeMap` or None, optional
      Which mapping to use for storing the non-conformant targets. If
      None was provided, new one would be automagically generated
      depending on the given/deduced domain.

    Returns
    -------
    am

    LightSVM format is an ASCII representation with a single sample per
    each line::

      output featureIndex:featureValue ... featureIndex:featureValue

    where ``output`` is specific for a given domain:

    regression
      float number
    binary
      integer labels from {-1, 1}
    multiclass
      integer labels from {1..ds.targets_attr.nunique}

    """
    targets_a = dataset.sa[targets_attr]
    targets = targets_a.value

    # XXX this all below
    #  * might become cleaner
    #  * might be RF to become more generic to be used may be elsewhere as well

    if domain is None:
        if targets.dtype.kind in ['S', 'U', 'i']:
            if len(targets_a.unique) == 2:
                domain = 'binary'
            else:
                domain = 'multiclass'
        else:
            domain = 'regression'

    if domain in ['multiclass', 'binary']:
        # check if labels are appropriate and provide mapping if necessary
        utargets = targets_a.unique
        if domain == 'binary' and set(utargets) != set([-1, 1]):
            # need mapping
            if len(utargets) != 2:
                raise ValueError, \
                      "We need 2 unique targets in %s of %s. Got targets " \
                      "from set %s" % (targets_attr, dataset, utargets)
            if am is None:
                am = AttributeMap(dict(zip(utargets, [-1, 1])))
            elif set(am.keys()) != set([-1, 1]):
                raise ValueError, \
                      "Provided %s doesn't map into binary " \
                      "labels -1,+1" % (am,)
        elif domain == 'multiclass' \
                 and set(utargets) != set(range(1, len(utargets)+1)):
            if am is None:
                am = AttributeMap(dict(zip(utargets,
                                           range(1, len(utargets) + 1))))
            elif set(am.keys()) != set([-1, 1]):
                raise ValueError, \
                      "Provided %s doesn't map into multiclass " \
                      "range 1..N" % (am, )

    if am is not None:
        # map the targets
        targets = am.to_numeric(targets)

    for t, s in zip(targets, dataset.samples):
        out.write(('%g %s\n'
                   % (t,
                      ' '.join(
                          '%i:%.8g' % (i, v)
                          for i,v in zip(range(1, dataset.nfeatures+1), s)))).encode('ascii'))

    out.flush()                # push it out
    return am


def from_lightsvm_format(in_, targets_attr='targets', am=None):
    """Loads dataset from a file in lightsvm format

    Parameters
    ----------
    in_
      Anything we could iterate over and obtain strings, such as `File`
    targets_attr : string, optional
      Name of the samples attribute to be used to store targets/labels
    am : `AttributeMap` or None, optional
      Which mapping to use for mapping labels back into possibly a
      literal representation.

    Returns
    -------
    dataset

    See Also
    --------
    to_lightsvm_format : conversion to lightsvm format
    """
    targets = []
    samples = []
    for l in in_:
        # we need to parse the line
        entries = l.split()
        targets += entries[:1]
        id_features = [e.split(':') for e in entries[1:]]
        f_ids = np.array([x[0] for x in id_features], dtype=int)
        f_vals = [float(x[1]) for x in id_features]
        if np.any(f_ids != np.arange(1, len(f_ids)+1)):
            raise NotImplementedError, \
                  "For now supporting only input of non-sparse " \
                  "lightsvm-formatted files. got line with feature " \
                  "ids %s " % f_ids
        samples.append(f_vals)

    # lets try to make targets of int, float, string until first
    # matching type ;)
    for t in (int, float, None):
        try:
            targets = np.array(targets, dtype=t)
            break
        except ValueError:
            pass

    if am is not None:
        targets = am.to_literal(targets)
    samples = np.array(samples)
    sa = {}
    sa[targets_attr] = targets
    ds = Dataset(samples, sa=sa)

    return ds

# CRF++ : http://crfpp.sourceforge.net/



