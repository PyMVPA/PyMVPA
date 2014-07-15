# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Implementation of a common processing object (node)."""

__docformat__ = 'restructuredtext'

import time
import numpy as np
from mvpa2.support import copy

from mvpa2.base.dochelpers import _str, _repr, _repr_attrs
from mvpa2.base.state import ClassWithCollections, ConditionalAttribute

from mvpa2.base.collections import SampleAttributesCollection, \
     FeatureAttributesCollection, DatasetAttributesCollection

if __debug__:
    from mvpa2.base import debug

class Node(ClassWithCollections):
    """Common processing object.

    A `Node` is an object the processes datasets. It can be called with a
    `Dataset` and returns another dataset with the results. In addition, a node
    can also be used as a generator. Upon calling ``generate()`` with a datasets
    it yields (potentially) multiple result datasets.

    Node have a notion of ``space``. The meaning of this space may vary heavily
    across sub-classes. In general, this is a trigger that tells the node to
    compute and store information about the input data that is "interesting" in
    the context of the corresponding processing in the output dataset.
    """

    calling_time = ConditionalAttribute(enabled=True,
        doc="Time (in seconds) it took to call the node")

    raw_results = ConditionalAttribute(enabled=False,
        doc="Computed results before invoking postproc. " +
            "Stored only if postproc is not None.")

    # Work-around for "happily-broken-by-design" HDF5 storage
    # serialization: upon reconstruction, no __init__ is called
    # so no private attributes, introduced since the moment when
    # original structure was serialized, would get populated.
    # "More proper" solution would be finally to implement a full
    # chain of __reduce__ and __setstate__s for derived classes (and use
    # Parameters more).  For now we would simply define those with default
    # values also at class level.
    __pass_attr = None

    def __init__(self, space=None, pass_attr=None, postproc=None, **kwargs):
        """
        Parameters
        ----------
        space : str, optional
          Name of the 'processing space'. The actual meaning of this argument
          heavily depends on the sub-class implementation. In general, this is
          a trigger that tells the node to compute and store information about
          the input data that is "interesting" in the context of the
          corresponding processing in the output dataset.
        pass_attr : str, list of str|tuple, optional
          Additional attributes to pass on to an output dataset. Attributes can
          be taken from all three attribute collections of an input dataset
          (sa, fa, a -- see :meth:`Dataset.get_attr`), or from the collection
          of conditional attributes (ca) of a node instance. Corresponding
          collection name prefixes should be used to identify attributes, e.g.
          'ca.null_prob' for the conditional attribute 'null_prob', or
          'fa.stats' for the feature attribute stats. In addition to a plain
          attribute identifier it is possible to use a tuple to trigger more
          complex operations. The first tuple element is the attribute
          identifier, as described before. The second element is the name of the
          target attribute collection (sa, fa, or a). The third element is the
          axis number of a multidimensional array that shall be swapped with the
          current first axis. The fourth element is a new name that shall be
          used for an attribute in the output dataset.
          Example: ('ca.null_prob', 'fa', 1, 'pvalues') will take the
          conditional attribute 'null_prob' and store it as a feature attribute
          'pvalues', while swapping the first and second axes. Simplified
          instructions can be given by leaving out consecutive tuple elements
          starting from the end.
        postproc : Node instance, optional
          Node to perform post-processing of results. This node is applied
          in `__call__()` to perform a final processing step on the to be
          result dataset. If None, nothing is done.
        """
        ClassWithCollections.__init__(self, **kwargs)
        if __debug__:
            debug("NO",
                  "Init node '%s' (space: '%s', postproc: '%s')",
                  (self.__class__.__name__, space, str(postproc)))
        self.set_space(space)
        self.set_postproc(postproc)
        if isinstance(pass_attr, basestring):
            pass_attr = (pass_attr,)
        self.__pass_attr = pass_attr


    def __call__(self, ds):
        """
        The default implementation calls ``_precall()``, ``_call()``, and
        finally returns the output of ``_postcall()``.

        Parameters
        ----------
        ds: Dataset
          Input dataset.

        Returns
        -------
        Dataset
        """
        t0 = time.time()                # record the time when call initiated

        self._precall(ds)
        result = self._call(ds)
        result = self._postcall(ds, result)

        self.ca.calling_time = time.time() - t0 # set the calling_time
        return result


    def _precall(self, ds):
        """Preprocessing of data

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original input dataset.

        Returns
        -------
        Dataset
        """
        return ds


    def _call(self, ds):
        raise NotImplementedError


    def _postcall(self, ds, result):
        """Postprocessing of results.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original input dataset.
        result: Dataset
          Preliminary result dataset (as produced by ``_call()``).

        Returns
        -------
        Dataset
        """
        result = self._pass_attr(ds, result)
        result = self._apply_postproc(ds, result)
        return result

    def _pass_attr(self, ds, result):
        """Pass a configured set of attributes on to the output dataset"""
        pass_attr = self.__pass_attr
        if pass_attr is not None:
            ca = self.ca
            ca_keys = self.ca.keys()
            for a in pass_attr:
                maxis = 0
                rcol = None
                attr_newname = None
                if isinstance(a, tuple):
                    if len(a) > 1:
                        # target collection is second element
                        colswitch = {'sa': result.sa, 'fa': result.fa, 'a': result.a}
                        rcol = colswitch[a[1]]
                    if len(a) > 2:
                        # major axis is third element
                        maxis = a[2]
                    if len(a) > 3:
                        # new attr name if fourth element
                        attr_newname = a[3]
                    # the attribute name is the first element
                    a = a[0]
                # It might come from .ca of this instance
                if a.startswith('ca.'):
                    a = a[3:]
                if a in ca_keys:
                    if rcol is None:
                        # We will assign it to .sa for now
                        rcol = result.sa
                    attr = ca[a]
                else:
                    # look in the ds
                    # find it in the original ds
                    attr, col = ds.get_attr(a)
                    if rcol is None:
                        # ONLY if there was no explicit output collection set
                        # deduce corresponding collection in results
                        # Since isinstance would take longer (eg 200 us vs 4)
                        # for now just use 'is' on the __class__
                        col_class = col.__class__
                        if col_class is SampleAttributesCollection:
                            rcol = result.sa
                        elif col_class is FeatureAttributesCollection:
                            rcol = result.fa
                        elif col_class is DatasetAttributesCollection:
                            rcol = result.a
                        else:
                            raise ValueError("Cannot determine origin of %s collection"
                                             % col)
                if attr_newname is None:
                    # go with previous name if no other is given
                    attr_newname = attr.name
                if maxis == 0:
                    # all good
                    value = attr.value
                else:
                    # move selected axis to the front
                    value = np.swapaxes(attr.value, 0, maxis)
                # "shallow copy" into the result
                # this way we also invoke checks for the correct length etc
                rcol[attr_newname] = value
        return result

    def _apply_postproc(self, ds, result):
        """Apply any post-processing to an output dataset"""
        if not self.__postproc is None:
            if __debug__:
                debug("NO",
                      "Applying post-processing node %s", (self.__postproc,))
            self.ca.raw_results = result
            result = self.__postproc(result)
        return result

    def generate(self, ds):
        """Yield processing results.

        This methods causes the node to behave like a generator. By default it
        simply yields a single result of its processing -- identical to the
        output of calling the node with a dataset. Subclasses might implement
        generators that yield multiple results.

        Parameters
        ----------
        ds: Dataset
          Input dataset

        Returns
        -------
        generator
          the generator yields the result of the processing.
        """
        yield self(ds)


    def get_space(self):
        """Query the processing space name of this node."""
        return self.__space


    def set_space(self, name):
        """Set the processing space name of this node."""
        self.__space = name


    def get_postproc(self):
        """Returns the post-processing node or None."""
        return self.__postproc


    def set_postproc(self, node):
        """Assigns a post-processing node

        Set to `None` to disable postprocessing.
        """
        self.__postproc = node


    def __str__(self):
        return _str(self)


    def __repr__(self, prefixes=[]):
        return super(Node, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['space', 'pass_attr', 'postproc']))

    space = property(get_space, set_space,
                     doc="Processing space name of this node")

    pass_attr = property(lambda self: self.__pass_attr,
                         doc="Which attributes of the dataset or self.ca "
                         "to pass into result dataset upon call")

    postproc = property(get_postproc, set_postproc,
                        doc="Node to perform post-processing of results")


class CompoundNode(Node):
    """List of nodes. 

    A CompoundNode behaves similar to a list container: Nodes can be appended,
    and the chain can be sliced like a list, etc ...
    
    Subclasses such as ChainNode and CombinedNode implement the _call
    method in different ways.
    """
    def __init__(self, nodes, **kwargs):
        """
        Parameters
        ----------
        nodes: list
          Node instances.
        """
        if not len(nodes):
            raise ValueError("%s needs at least one embedded node."
                             % self.__class__.__name__)

        self._nodes = nodes
        Node.__init__(self, **kwargs)


    def __copy__(self):
        # XXX how do we safely and exhaustively copy a node?
        return self.__class__([copy.copy(n) for n in self])


    def _call(self, ds):
        raise NotImplementedError("This is an abstract class.")


    def generate(self, ds, startnode=0):
        """
        Parameters
        ----------
        ds: Dataset
          To be processed dataset
        startnode: int
          First node in the chain that shall be considered. This argument is
          mostly useful for internal optimization.
        """
        first_node = self[startnode]
        if __debug__:
            debug('MAP', "%s: input (%s) -> generator (%i/%i): '%s'",
                  (self.__class__.__name__, ds.shape,
                   startnode + 1, len(self), first_node))
        # let the first node generator as many datasets as it wants
        for gds in first_node.generate(ds):
            if startnode == len(self) - 1:
                # if this is already the last node yield the result
                yield gds
            else:
                # otherwise feed them through the rest of the chain
                for rgds in self.generate(gds, startnode=startnode + 1):
                    yield rgds


    #
    # Behave as a container
    #
    def append(self, node):
        """Append a node to the chain."""
        # XXX and if a node is a ChainMapper itself -- should we just
        # may be loop and add all the entries?
        self._nodes.append(node)


    def __len__(self):
        return len(self._nodes)


    def __iter__(self):
        for n in self._nodes:
            yield n


    def __reversed__(self):
        return reversed(self._nodes)


    def __getitem__(self, key):
        # if just one is requested return just one, otherwise return a
        # NodeChain again
        if isinstance(key, int):
            return self._nodes[key]
        else:
            # operate on shallow copy of self
            sliced = copy.copy(self)
            sliced._nodes = self._nodes[key]
            return sliced


    def __repr__(self, prefixes=[]):
        return super(CompoundNode, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['nodes']))


    def __str__(self):
        return _str(self, '-'.join([str(n) for n in self]))

    nodes = property(fget=lambda self:self._nodes)


class ChainNode(CompoundNode):
    """
    This class allows to concatenate a list of nodes into a processing chain.
    When called with a dataset, it is sequentially fed through nodes in the
    chain. A ChainNode may also be used as a generator. In this case, all
    nodes in the chain are treated as generators too, and the ChainNode
    behaves as a single big generator that recursively calls all embedded
    generators and yield the results.
    """
    def __init__(self, nodes, **kwargs):
        """
        Parameters
        ----------
        nodes: list
          Node instances.
        """
        CompoundNode.__init__(self, nodes=nodes, **kwargs)

    def _call(self, ds):
        mp = ds
        for i, n in enumerate(self):
            if __debug__:
                debug('MAP', "%s: input (%s) -> node (%i/%i): '%s'",
                      (self.__class__.__name__,
                       hasattr(mp, 'shape') and mp.shape or '???',
                       i + 1, len(self),
                       n))
            mp = n(mp)
        if __debug__:
            debug('MAP', "%s: output (%s)", (self.__class__.__name__, mp.shape))
        return mp


class CombinedNode(CompoundNode):
    """Node to pass a dataset on to a set of nodes and combine there output.

    Output combination or aggregation is currently done by hstacking or
    vstacking the resulting datasets.
    """

    def __init__(self, nodes, combine_axis, a=None, **kwargs):
        """
        Parameters
        ----------
        mappers : list
        combine_axis : ['h', 'v']
        a: {'unique','drop_nonunique','uniques','all'} or True or False or None (default: None)
            Indicates which dataset attributes from datasets are stored 
            in merged_dataset. If an int k, then the dataset attributes from 
            datasets[k] are taken. If 'unique' then it is assumed that any
            attribute common to more than one dataset in datasets is unique;
            if not an exception is raised. If 'drop_nonunique' then as 'unique',
            except that exceptions are not raised. If 'uniques' then, for each 
            attribute,  any unique value across the datasets is stored in a tuple 
            in merged_datasets. If 'all' then each attribute present in any 
            dataset across datasets is stored as a tuple in merged_datasets; 
            missing values are replaced by None. If None (the default) then no 
            attributes are stored in merged_dataset. True is equivalent to
            'drop_nonunique'. False is equivalent to None.
        """
        CompoundNode.__init__(self, nodes=nodes, **kwargs)
        self._combine_axis = combine_axis
        self._a = a

    def __copy__(self):
        return self.__class__([copy.copy(n) for n in self],
                                         copy.copy(self._combine_axis),
                                         copy.copy(self._a))


    def _call(self, ds):
        out = [node(ds) for node in self]
        from mvpa2.datasets import hstack, vstack
        stacker = {'h': hstack, 'v': vstack}
        stacked = stacker[self._combine_axis](out, self._a)
        return stacked

    def __repr__(self, prefixes=[]):
        return super(CombinedNode, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['combine_axis', 'a']))


