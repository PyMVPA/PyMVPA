#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""

MNIST handwritten digits
========================

.. index:: MDP


MDP-style classification
------------------------

"""

import cPickle
import mdp
import gzip
import numpy as N

class DigitsIterator:
    def __init__(self, digits, labels):
        self.digits = digits
        self.labels = labels
    def __iter__(self):
        frac = 10
        ll = len(self.labels)
        for i in xrange(frac):
            yield self.digits[i*ll/frac:(i+1)*ll/frac], \
                  self.labels[i*ll/frac:(i+1)*ll/frac]

data = cPickle.load(gzip.open('mnist.pickle.gz'))
for k in ['traindata', 'testdata']:
    data[k] = data[k].reshape(-1, 28 * 28)

fdaflow = (mdp.nodes.WhiteningNode(output_dim=10, dtype='d') +
           mdp.nodes.PolynomialExpansionNode(2) +
           mdp.nodes.FDANode(output_dim=9) +
           mdp.nodes.GaussianClassifierNode())

fdaflow.verbose = True

fdaflow.train([[data['traindata']],
               None,
               DigitsIterator(data['traindata'],
                              data['trainlabels']),
               DigitsIterator(data['traindata'],
                              data['trainlabels'])
               ])

feature_space = fdaflow[:-1](data['testdata'])
guess = fdaflow[-1].classify(feature_space)
err = 1 - N.mean(guess == data['testlabels'])
print 'Test error:', err

"""

Doing it the PyMVPA way
-----------------------

"""

import pylab as P
from mvpa.suite import *

"""

Following  MNIST_ dataset is not distributed along with
PyMVPA due to its size.  Please download it into directory from which
you are running this example first.

.. _MNIST: http://www.pymvpa.org/files/data/mnist.pickle.gz

"""

data = cPickle.load(gzip.open('mnist.pickle.gz'))
ds = Dataset.from_basic(
        data['traindata'],
        labels=data['trainlabels'],
        mapper=FlattenMapper((28, 28)))
testds = Dataset.from_basic(
        data['testdata'],
        labels=data['testlabels'],
        mapper=ds.a.mapper)

ds.init_origids('samples')
testds.init_origids('samples')

examples = [0, 25024, 50000, 59000]

P.figure(figsize=(6, 6))

for i, id_ in enumerate(examples):
    ax = P.subplot(2, 2, i+1)
    ax.axison = False
    P.imshow(data['traindata'][id_].T, cmap=P.cm.gist_yarg,
             interpolation='nearest', aspect='equal')

P.subplots_adjust(left=0, right=1, bottom=0, top=1,
                  wspace=0.05, hspace=0.05)
P.show()


fdaflow = (mdp.nodes.WhiteningNode(output_dim=10, dtype='d') +
           mdp.nodes.PolynomialExpansionNode(2) +
           mdp.nodes.FDANode(output_dim=9))
fdaflow.verbose = True

mapper = MDPFlowMapper(fdaflow,
                       ([], [], [DatasetAttributeExtractor('sa', 'labels')]))

terr = TransferError(MappedClassifier(SMLR(), mapper),
                     enable_states=['confusion',
                                    'samples_error'])
err = terr(testds, ds)
print 'Test error:', err
try:
    from enthought.mayavi.mlab import points3d
    P3D = True
except ImportError:
    print 'Sorry, no 3D plots!'
    P3D = False

fmts = ['bo', 'ro', 'ko', 'mo']
pts = []
for i, ex in enumerate(examples):
    pts.append(mapper.forward(ds.samples[ex:ex+100])[:, :3])

if P3D:
    for p in pts:
        points3d(p[:, 0], p[:, 1], p[:, 2])

#if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
#    P.show()
