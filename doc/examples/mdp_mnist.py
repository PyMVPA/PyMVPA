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
import numpy as np

class DigitsIterator:
    def __init__(self, digits, labels):
        self.digits = digits
        self.targets = labels
    def __iter__(self):
        frac = 10
        ll = len(self.targets)
        for i in xrange(frac):
            yield self.digits[i*ll/frac:(i+1)*ll/frac], \
                  self.targets[i*ll/frac:(i+1)*ll/frac]

data = cPickle.load(gzip.open('mnist.pickle.gz'))
for k in ['traindata', 'testdata']:
    data[k] = data[k].reshape(-1, 28 * 28)

fdaclf = (mdp.nodes.WhiteningNode(output_dim=10, dtype='d') +
          mdp.nodes.PolynomialExpansionNode(2) +
          mdp.nodes.FDANode(output_dim=9) +
          mdp.nodes.GaussianClassifierNode())

fdaclf.verbose = True

fdaclf.train([[data['traindata']],
               None,
               DigitsIterator(data['traindata'],
                              data['trainlabels']),
               DigitsIterator(data['traindata'],
                              data['trainlabels'])
               ])

feature_space = fdaclf[:-1](data['testdata'])
guess = fdaclf[-1].label(feature_space)
err = 1 - np.mean(guess == data['testlabels'])
print 'Test error:', err

"""

Doing it the PyMVPA way
-----------------------

"""

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from mvpa.suite import *

"""

Following  MNIST_ dataset is not distributed along with
PyMVPA due to its size.  Please download it into directory from which
you are running this example first.
Data visualization depends on the 3D matplotlib features, which are
available only from version 0.99

.. _MNIST: http://www.pymvpa.org/files/data/mnist.pickle.gz

"""

data = cPickle.load(gzip.open('mnist.pickle.gz'))
train = dataset_wizard(
        data['traindata'],
        targets=data['trainlabels'],
        chunks='train')
test = dataset_wizard(
        data['testdata'],
        targets=data['testlabels'],
        chunks='test')
# merge the datasets into on
ds = vstack((train, test))
ds.init_origids('samples')

#examples = [0, 25024, 50000, 59000]
examples = [3001 + 5940 * i for i in range(10)]
#examples = [0, 9000, 18000]


pl.figure(figsize=(2, 5))

for i, id_ in enumerate(examples):
    ax = pl.subplot(2, 5, i+1)
    ax.axison = False
    pl.imshow(data['traindata'][id_].T, cmap=pl.cm.gist_yarg,
             interpolation='nearest', aspect='equal')

pl.subplots_adjust(left=0, right=1, bottom=0, top=1,
                  wspace=0.05, hspace=0.05)
pl.draw()


fdaflow = (mdp.nodes.WhiteningNode(output_dim=10, dtype='d') +
           mdp.nodes.PolynomialExpansionNode(2) +
           mdp.nodes.FDANode(output_dim=9))
fdaflow.verbose = True

mapper = MDPFlowMapper(fdaflow,
                       ([], [], [DatasetAttributeExtractor('sa', 'targets')]))

tm = TransferMeasure(MappedClassifier(SMLR(), mapper),
                     Splitter('chunks', attr_values=['train', 'test']),
                     enable_ca=['stats', 'samples_error'])
tm(ds)
print 'Test error:', 1 - tm.ca.stats.stats['ACC']

if externals.exists('matplotlib') \
   and externals.versions['matplotlib'] >= '0.99':
    pts = []
    for i, ex in enumerate(examples):
        pts.append(mapper.forward(ds.samples[ex:ex+200])[:, :3])

    fig = pl.figure()

    ax = Axes3D(fig)
    colors = ('r','g','b','k','c','m','y','burlywood','chartreuse','gray')
    clouds = []
    for i, p in enumerate(pts):
        print i
        clouds.append(ax.plot(p[:, 0], p[:, 1], p[:, 2], 'o', c=colors[i],
                              label=str(i), alpha=0.6))

    ax.legend([str(i) for i in range(10)])
    pl.draw()

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()
