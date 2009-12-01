# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N
import mdp

from mvpa.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa.misc.support import isInVolume


class DatasetAttributeExtractor(object):
    def __init__(self, col, key):
        self._col = col
        self._key = key

    def __call__(self, ds):
        return ds.__dict__[self._col][self._key]

DAE = DatasetAttributeExtractor


class MDPNodeMapper(Mapper):
    def __init__(self, node, nodeargs=None, inspace=None):
        """
        Parameters
        ----------
        node : mdp.Node instance
        nodeargs : dict
          'train' ((), {})
          'stoptrain' ((), {})
          'exec' ((), {})
          'inv' ((), {})
        """
        # Tiziano will check if there can be/is a public way to do it
        if not len(node._train_seq) == 1:
            raise ValueError("MDPNodeMapper does not support MDP nodes with "
                             "multiple training phases.")
        Mapper.__init__(self, inspace=inspace)
        self.node = node
        self.nodeargs = nodeargs


    def _expand_args(self, phase, ds=None):
        args = []
        kwargs = {}
        if not self.nodeargs is None and phase in self.nodeargs:
            sargs, skwargs = self.nodeargs[phase]
            for a in sargs:
                if isinstance(a, DatasetAttributeExtractor):
                    if ds is None:
                        raise RuntimeError('MDPNodeMapper does not (yet) '
                                           'support argument extraction from dataset on '
                                           'forward()')
                    args.append(a(ds))
                else:
                    args.append(a)
            for k in skwargs:
                if isinstance(skwargs[k], DatasetAttributeExtractor):
                    if ds is None:
                        raise RuntimeError('MDPNodeMapper does not (yet) '
                                           'support argument extraction from dataset on '
                                           'forward()')
                    kwargs[k] = skwargs[k](ds)
                else:
                    kwargs[k] = skwargs[k]
        return args, kwargs


    def _train(self, ds):
        if not self.node.is_trainable():
                return
        args, kwargs = self._expand_args('train', ds)
        self.node.train(ds.samples, *args, **kwargs)


    def _forward_data(self, data):
        # XXX maybe move stop_training to the end of _train, but would
        # prohibit incremental training
        if self.node.is_training():
            args, kwargs = self._expand_args('stoptrain', data)
            self.node.stop_training(*args, **kwargs)

        args, kwargs = self._expand_args('exec', data)
        return self.node.execute(data, *args, **kwargs)


    def _reverse_data(self, data):
        args, kwargs = self._expand_args('inv', data)
        return self.node.inverse(data, *args, **kwargs)


    def is_valid_outid(self, id):
        """Checks for a valid output id for this (trained) mapper).

        If the mapper is not trained any id is invalid.
        """
        # untrained -- all is invalid
        outdim = self.get_outsize()
        if outdim is None:
            kwargs[k] = skwargs[k](ds)
        else:
            kwargs[k] = skwargs[k]
        return args, kwargs


    def is_valid_inid(self, id):
        """Checks for a valid output id for this (trained) mapper).

        If the mapper is not trained any id is invalid.
        """
        # untrained -- all is invalid
        indim = self.get_insize()
        if indim is None:
            return False
        return id >= 0 and id < indim


    def get_insize(self):
        """Return the (flattened) size of input space vectors."""
        return self.node.input_dim


    def get_outsize(self):
        """Return the size of output space vectors."""
        return self.node.output_dim


    def get_outids(self, in_ids=None, **kwargs):
        """Determine the output ids from a list of input space id/coordinates.

        Parameters
        ----------
        in_ids : list
          List of input ids whos output ids shall be determined.
        **kwargs: anything
          Further qualification of coordinates in particular spaces. Spaces are
          identified by the respected keyword and the values expresses an
          additional criterion. If the mapper has any information about the
          given space it uses this information to further restrict the set of
          output ids. Information about unkown spaces is returned as is.

        Returns
        -------
        (list, dict)
          The list that contains all corresponding output ids. The default
          implementation returns an empty list -- meaning there is no
          one-to-one, or one-to-many correspondance of input and output feature
          spaces. The dictionary contains all space-related information that
          have not been processed by the mapper (i.e. the spaces they referred
          to are unknown to the mapper. By default all additional keyword
          arguments are returned as is.
        """
        ourspace = self.get_inspace()
        # first contrain the set of in_ids if a known space is given
        if not ourspace is None and ourspace in kwargs:
            # XXX don't do anything for now -- we claim that we cannot
            # track features through the MDP node
            # remove the space contraint, since it has been processed
            del kwargs[ourspace]

        return ([], kwargs)
