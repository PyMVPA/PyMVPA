# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Use scikit-learn transformer as mappers.

This module provides an adaptor to use sklearn transformers as PyMVPA mappers.
"""

__docformat__ = 'restructuredtext'

from mvpa2.support.copy import deepcopy

from mvpa2.base.learner import Learner
from mvpa2.mappers.base import Mapper

__all__ = ['SKLTransformer']

class SKLTransformer(Mapper):
    """Adaptor to use arbitrary sklearn transformer as a mapper.

    This basic adaptor support forward mapping only. It is clever enough
    to call ``fit_transform()`` instead of a serial ``fit()`` and
    ``transform()`` combo when an untrained instance is called with a dataset.

    >>> from sklearn.manifold import MDS
    >>> from mvpa2.misc.data_generators import normal_feature_dataset
    >>> ds = normal_feature_dataset(perlabel=10, nlabels=2)
    >>> print ds.shape
    (20, 4)
    >>> mds = SKLTransformer(MDS())
    >>> mapped = mds(ds)
    >>> print mapped.shape
    (20, 2)
    """
    def __init__(self, transformer, **kwargs):
        """
        Parameters
        ----------
        transformer : sklearn.transformer instance
        space : str or None, optional
          If not None, a sample attribute of the given name will be extracted
          from the training dataset and passed to the sklearn transformer's
          ``fit()`` method as ``y`` argument.

        """
        # NOTE: trailing spaces in above docstring must not be pruned
        # for correct parsing

        Mapper.__init__(self, auto_train=False, **kwargs)
        self._transformer = None
        self._pristine_transformer = transformer

    def __call__(self, ds):
        # overwrite __call__ to prevent the rigorous check of the learner was
        # trained before use and auto-train, because sklearn has optimized ways
        # for doing that, i.e. fit_transform()
        return super(Learner, self).__call__(ds)

    def _untrain(self):
        self._transformer = None

    def _get_y(self, ds):
        space = self.get_space()
        if space:
            y = ds.sa[space].value
        else:
            y = None
        return y

    def _get_transformer(self):
        if self._transformer is None:
            self._transformer = deepcopy(self._pristine_transformer)
        return self._transformer

    def _train(self, ds):
        tf = self._get_transformer()
        return tf.fit(ds.samples, self._get_y(ds))

    def _forward_dataset(self, ds):
        tf = self._get_transformer()
        if not self.is_trained:
            # sklearn support fit and transform at the same time, which might
            # be a lot faster, but we only do that, if the mapper is not
            # trained already
            out = tf.fit_transform(ds.samples, self._get_y(ds))
            self._set_trained()
        else:
            # some SKL classes do not swallow a superfluous `y` argument
            # we could be clever and use 'inspect' to query the function
            # signature, but we'll use a sledge hammer instead
            try:
                out = tf.transform(ds.samples, self._get_y(ds))
            except TypeError:
                out = tf.transform(ds.samples)
        return out
