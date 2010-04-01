# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""HDF5-based file IO for PyMVPA objects.

Based on the `h5py` package, this module provides two functions (`obj2hdf()`
and `hdf2obj()`, as well as the convenience functions `h5save()` and
`h5load()`) to store (in principle) arbitrary Python objects into HDF5 groups,
and using HDF5 as input, convert them back into Python object instances.

Similar to `pickle` a Python object is disassembled into its pieces, but instead
of serializing it into a byte-stream it is stored in chunks which type can be
natively stored in HDF5. That means basically everything that can be stored in
a NumPy array.

If an object is not readily storable, its `__reduce__()` method is called to
disassemble it into basic pieces.  The default implementation of
`object.__reduce__()` is typically sufficient. Hence, for any new-style Python
class there is, in general, no need to implement `__reduce__()`. However, custom
implementations might allow for leaner HDF5 representations and leaner files.
Basic types, such as `list`, and `dict`, which `__reduce__()` method does not do
help with disassembling are also handled.

.. warning::

  Although, in principle, storage and reconstruction of arbitrary object types
  is possible, it might not be implemented yet. The current focus lies on
  storage of PyMVPA datasets and their attributes (e.g. Mappers).  Especially,
  objects with recursive references will cause problems with the current
  implementation.
"""

__docformat__ = 'restructuredtext'

import types
import numpy as np
import h5py
from mvpa.base.types import asobjarray

if __debug__:
    from mvpa.base import debug

# Comment: H5Py defines H5Error
class HDF5ConversionError(Exception):
    """Generic exception to be thrown while doing conversions to/from HDF5
    """
    pass

#
# TODO: check for recursions!!!
#
def hdf2obj(hdf):
    """Convert an HDF5 group definition into an object instance.

    Obviously, this function assumes the conventions implemented in the
    `obj2hdf()` function. Those conventions will eventually be documented in
    the module docstring, whenever they are sufficiently stable.

    Parameters
    ----------
    hdf : HDF5 group instance
      HDF5 group instance. this could also be an HDF5 file instance.

    Notes
    -----
    Although, this function uses a way to reconstruct object instances that is
    similar to unpickling, it should be *relatively* safe to open HDF files
    from untrusted sources. Only basic datatypes are stored in HDF files, and
    there is no foreign code that is executed during reconstructing. For that
    reason, any type that shall be reconstructed needs to be importable
    (importing is done be fully-qualified module names).

    Returns
    -------
    object instance
    """
    # already at the level of real data
    if isinstance(hdf, h5py.Dataset):
        if __debug__:
            debug('HDF5', "Load HDF5 dataset '%s'." % hdf.name)
        if not len(hdf.shape):
            # extract the scalar from the 0D array
            return hdf[()]
        else:
            # read array-dataset into an array
            value = np.empty(hdf.shape, hdf.dtype)
            hdf.read_direct(value)
            return value
    else:
        # check if we have a class instance definition here
        if not ('class' in hdf.attrs or 'recon' in hdf.attrs):
            raise LookupError("Found hdf group without class instance "
                    "information (group: %s). Cannot convert it into an "
                    "object (attributes: '%s')."
                    % (hdf.name, hdf.attrs.keys()))

        if __debug__:
            debug('HDF5', "Parsing HDF5 group (attributes: '%s')."
                          % (hdf.attrs.keys()))
        if 'recon' in hdf.attrs:
            # we found something that has some special idea about how it wants
            # to be reconstructed
            # look for arguments for that reconstructor
            recon = hdf.attrs['recon']
            mod = hdf.attrs['module']
            if mod == '__builtin__':
                raise NotImplementedError(
                        "Built-in reconstructors are not supported (yet). "
                        "Got: '%s'." % recon)

            # turn names into definitions
            mod = __import__(mod, fromlist=[recon])
            recon = mod.__dict__[recon]

            if 'rcargs' in hdf:
                recon_args = _hdf_tupleitems_to_obj(hdf['rcargs'])
            else:
                recon_args = ()

            if __debug__:
                debug('HDF5', "Reconstructing object with '%s' (%i arguments)."
                              % (recon, len(recon_args)))
            # reconstruct
            obj = recon(*recon_args)

            # TODO Handle potentially avialable state settings
            return obj

        cls = hdf.attrs['class']
        mod = hdf.attrs['module']
        if not mod == '__builtin__':
            # some custom class is desired
            # import the module and the class
            if __debug__:
                debug('HDF5', "Importing '%s' from '%s'." % (cls, mod))
            mod = __import__(mod, fromlist=[cls])

            if cls in ('function', 'type'):
                oname = hdf.attrs['name']
                # special case of non-built-in functions
                if __debug__:
                    debug('HDF5', "Loaded %s '%s' from '%s'."
                                  % (cls, oname, mod))
                return mod.__dict__[oname]

            # get the class definition from the module dict
            cls = mod.__dict__[cls]

            if __debug__:
                debug('HDF5', "Reconstructing class '%s' instance."
                              % cls)
            # create the object
            if issubclass(cls, dict):
                # use specialized __new__ if necessary or beneficial
                obj = dict.__new__(cls)
            else:
                obj = object.__new__(cls)

            if 'state' in hdf:
                # insert the state of the object
                if __debug__:
                    debug('HDF5', "Populating instance state.")
                state = _hdf_dictitems_to_obj(hdf['state'])
                obj.__dict__.update(state)
                if __debug__:
                    debug('HDF5', "Updated %i state items." % len(state))

            # do we process a container?
            if 'items' in hdf:
                if issubclass(cls, dict):
                    # charge a dict itself
                    if __debug__:
                        debug('HDF5', "Populating dictionary object.")
                    obj.update(_hdf_dictitems_to_obj(hdf['items']))
                    if __debug__:
                        debug('HDF5', "Loaded %i items." % len(obj))
                else:
                    raise NotImplementedError(
                            "Unhandled conatiner typ (got: '%s')." % cls)

            return obj

        else:
            if __debug__:
                debug('HDF5', "Reconstruction built-in object '%s'." % cls)
            # built in type (there should be only 'list', 'dict' and 'None'
            # that would not be in a Dataset
            if cls == 'NoneType':
                return None
            elif cls == 'tuple':
                return _hdf_tupleitems_to_obj(hdf['items'])
            elif cls == 'list':
                l = _hdf_listitems_to_obj(hdf['items'])
                if 'is_objarray' in hdf.attrs:
                    # need to handle special case of arrays of objects
                    return asobjarray(l)
                else:
                    return l
            elif cls == 'dict':
                return _hdf_dictitems_to_obj(hdf['items'])
            elif cls == 'function':
                raise RuntimeError("Unhandled reconstruction of built-in "
                        "function (at '%s')." % hdf.name)
            else:
                raise RuntimeError("Found hdf group with a builtin type "
                        "that is not handled by the parser (group: %s). This "
                        "is a conceptual bug in the parser. Please report."
                        % hdf.name)


def _hdf_dictitems_to_obj(hdf, skip=None):
    if skip is None:
        skip = []
    if hdf.attrs.get('__keys_in_tuple__', 0):
        items = _hdf_listitems_to_obj(hdf)
        items = [i for i in items if not i[0] in skip]
        return dict(items)
    else:
        # legacy files had keys as group names
        return dict([(item, hdf2obj(hdf[item]))
                        for item in hdf
                            if not item in skip])


def _hdf_listitems_to_obj(hdf):
    return [hdf2obj(hdf[str(i)]) for i in xrange(len(hdf))]


def _hdf_tupleitems_to_obj(hdf):
    return tuple(_hdf_listitems_to_obj(hdf))

#
# TODO: check for recursions!!!
#
def obj2hdf(hdf, obj, name=None, **kwargs):
    """Store an object instance in an HDF5 group.

    A given object instance is (recursively) disassembled into pieces that are
    storable in HDF5. In general, any pickable object should be storable, but
    since the parser is not complete, it might not be possible (yet).

    .. warning::

      Currently, the parser does not track recursions. If an object contains
      recursive references all bets are off. Here be dragons...

    Parameters
    ----------
    hdf : HDF5 group instance
      HDF5 group instance. this could also be an HDF5 file instance.
    obj : object instance
      Object instance that shall be stored.
    name : str or None
      Name of the object. In case of a complex object that cannot be stored
      natively without disassembling them, this is going to be a new group,
      Otherwise the name of the dataset. If None, no new group is created.
    **kwargs
      All additional arguments will be passed to `h5py.Group.create_dataset()`
    """
    if isinstance(obj, np.ndarray) and obj.dtype == np.object \
       and not len(obj.shape):
        # we store 0d object arrays just by content and set a flag
        obj = np.asscalar(obj)
        hdf.attrs.create('is_objarray', True)

    # if it is something that can go directly into HDF5, put it there
    # right away
    if np.isscalar(obj) \
       or (isinstance(obj, np.ndarray) and not obj.dtype == np.object):
        if name is None:
            # HDF5 cannot handle datasets without a name
            name = '__unnamed__'
        if __debug__:
            debug('HDF5', "Storing '%s' in HDF5 dataset '%s'"
                          % (type(obj), name))

        hdf.create_dataset(name, None, None, obj, **kwargs)
        return

    if __debug__:
        debug('HDF5', "Convert '%s' into HDF5 group with name '%s'."
                      % (type(obj), name))

    if not name is None:
        # complex objects
        if __debug__:
            debug('HDF5', "Create HDF5 group '%s'" % (name))
        grp = hdf.create_group(str(name))
    else:
        grp = hdf

    # special case of array of type object -- we turn them into lists and
    # process as usual, but set a flag to trigger appropriate reconstruction
    if isinstance(obj, np.ndarray) and obj.dtype == np.object:
        if __debug__:
            debug('HDF5', "Convert array of objects into a list.")
        obj = list(obj)
        grp.attrs.create('is_objarray', True)

    # try disassembling the object
    try:
        pieces = obj.__reduce__()
    except TypeError:
        # probably a container
        pieces = None
        if __debug__:
            debug('HDF5', "'%s' could not be __reduce__()'d. A container?."
                          % (type(obj)))

    # common container handling, either __reduce__ was not possible
    # or it was the default implementation
    if pieces is None or pieces[0].__name__ == '_reconstructor':
        # store class info (fully-qualified)
        grp.attrs.create('class', obj.__class__.__name__)
        if hasattr(obj, '__module__'):
            grp.attrs.create('module', obj.__module__)
        else:
            grp.attrs.create('module', obj.__class__.__module__)
        if hasattr(obj, '__name__'):
            # for functions/types we need a name for reconstruction
            oname = obj.__name__
            if oname == '<lambda>':
                raise HDF5ConversionError(
                    "Can't obj2hdf lambda functions. Got %r" % (obj,))
            grp.attrs.create('name', oname)
        if isinstance(obj, list) or isinstance(obj, tuple):
            if __debug__: debug('HDF5', "Special case: Store a list/tuple.")
            items = grp.create_group('items')
            for i, item in enumerate(obj):
                obj2hdf(items, item, name=str(i), **kwargs)
        elif isinstance(obj, dict):
            if __debug__:
                debug('HDF5', "Special case: Store a dictionary.")
            items = grp.create_group('items')
            for i, key in enumerate(obj):
                # keys might be complex object, so they cannot serve as a
                # name in this case
                obj2hdf(items, (key, obj[key]), name=str(i), **kwargs)
                # leave a tag that the keys are stored within the item
                # tuple, to make it possible to support legacy files
                items.attrs.create('__keys_in_tuple__', 1)
        # pull all remaining data from the default __reduce__
        if not pieces is None and len(pieces) > 2:
            stategrp = grp.create_group('state')
            # there is something in the state
            state = pieces[2]
            if __debug__:
                debug('HDF5', "Store object state (%i items)." % len(state))
            # loop over all attributes and store them
            for attr in state:
                obj2hdf(stategrp, state[attr], attr, **kwargs)
        # for the default __reduce__ there is nothin else to do
        return
    else:
        if __debug__:
            debug('HDF5', "Custom __reduce__: (%i constructor arguments)."
                          % len(pieces[1]))
        # XXX handle custom reduce
        grp.attrs.create('recon', pieces[0].__name__)
        grp.attrs.create('module', pieces[0].__module__)
        args = grp.create_group('rcargs')
        for i, arg in enumerate(pieces[1]):
            obj2hdf(args, arg, str(i), **kwargs)
        return


def h5save(filename, data, name=None, mode='w', **kwargs):
    """Stores arbitray data in an HDF5 file.

    This is a convenience wrapper around `obj2hdf()`. Please see its
    documentation for more details -- especially the warnings!!

    Parameters
    ----------
    filename : str
      Name of the file the data shall be stored in.
    data : arbitrary
      Instance of an object that shall be stored in the file.
    name : str or None
      Name of the object. In case of a complex object that cannot be stored
      natively without disassembling them, this is going to be a new group,
      otherwise the name of the dataset. If None, no new group is created.
    mode : {'r', 'r+', 'w', 'w-', 'a'}
      IO mode of the HDF5 file. See `h5py.File` documentation for more
      information.
    **kwargs
      All additional arguments will be passed to `h5py.Group.create_dataset`.
      This could, for example, be `compression='gzip'`.
    """
    hdf = h5py.File(filename, mode)
    try:
        obj2hdf(hdf, data, name, **kwargs)
    finally:
        hdf.close()


def h5load(filename, name=None):
    """Loads the content of an HDF5 file that has been stored by `h5save()`.

    This is a convenience wrapper around `hdf2obj()`. Please see its
    documentation for more details.

    Parameters
    ----------
    filename : str
      Name of the file to open and load its content.
    name : str
      Name of a specific object to load from the file.

    Returns
    -------
    instance
      An object of whatever has been stored in the file.
    """
    hdf = h5py.File(filename, 'r')
    try:
        if not name is None:
            if not name in hdf:
                raise ValueError("No object of name '%s' in file '%s'."
                                 % (name, filename))
            obj = hdf2obj(hdf[name])
        else:
            if not len(hdf) and not len(hdf.attrs):
                # there is nothing
                obj = None
            else:
                try:
                    obj = hdf2obj(hdf)
                    # XXX above operation might be really expensive
                    # and finally fail with something very deep
                    # under. TODO: RF to carry some cheap 'sensoring'
                    # first and then simply proceed with deep
                    # recursion or logic below
                except LookupError, e:
                    if __debug__:
                        debug('HDF5', "Failed to lookup object at top level of "
                              "'%s' due to %s." % (hdf, e))

                    # no object into at the top-level, but maybe in the next one
                    # this would happen for plain mat files with arrays
                    if len(hdf) == 1 and '__unnamed__' in hdf:
                        # just a single with special naem -> special case:
                        # return as is
                        obj = hdf2obj(hdf['__unnamed__'])
                        if 'is_objarray' in hdf.attrs:
                            # handle 0d obj arrays
                            obj = np.array(obj, dtype=np.object)
                    else:
                        # otherwise build dict with content
                        obj = {}
                        for k in hdf:
                            obj[k] = hdf2obj(hdf[k])
    finally:
        hdf.close()
    return obj
