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
Basic types, such as `list`, and `dict`, whose `__reduce__()` method does not do
help with disassembling are also handled.

.. warning::

  Although, in principle, storage and reconstruction of arbitrary object types
  is possible, it might not be implemented yet. The current focus lies on
  storage of PyMVPA datasets and their attributes (e.g. Mappers).
"""

__docformat__ = 'restructuredtext'

import types
import numpy as np
import h5py

import os
import os.path as osp

import mvpa2
from mvpa2.base import externals
from mvpa2.base.types import asobjarray

if __debug__:
    from mvpa2.base import debug

# don't ask -- makes history re-education a breeze
universal_classname_remapper = {
    ('mvpa2.mappers.base', 'FeatureSliceMapper'):
        ('mvpa2.featsel.base', 'StaticFeatureSelection'),
}
# Comment: H5Py defines H5Error
class HDF5ConversionError(Exception):
    """Generic exception to be thrown while doing conversions to/from HDF5
    """
    pass

def hdf2obj(hdf, memo=None):
    """Convert an HDF5 group definition into an object instance.

    Obviously, this function assumes the conventions implemented in the
    `obj2hdf()` function. Those conventions will eventually be documented in
    the module docstring, whenever they are sufficiently stable.

    Parameters
    ----------
    hdf : HDF5 group instance
      HDF5 group instance. this could also be an HDF5 file instance.
    memo : dict
      Dictionary tracking reconstructed objects to prevent recursions (analog to
      deepcopy).

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
    if memo is None:
        # init object tracker
        memo = {}
    # note, older file formats did not store objrefs
    if 'objref' in hdf.attrs:
        objref = hdf.attrs['objref']
    else:
        objref = None

    # if this HDF group has an objref that points to an already recontructed
    # object, simple return this object again
    if not objref is None and objref in memo:
        obj = memo[objref]
        if __debug__:
            debug('HDF5', "Use tracked object %s (%i)" % (type(obj), objref))
        return obj

    #
    # Actual data
    #
    if isinstance(hdf, h5py.Dataset):
        if __debug__:
            debug('HDF5', "Load from HDF5 dataset [%s]" % hdf.name)
        if 'is_scalar' in hdf.attrs:
            # extract the scalar from the 0D array
            obj = hdf[()]
            # and coerce it back into the native Python type if necessary
            if issubclass(type(obj), np.generic):
                obj = np.asscalar(obj)
        elif 'is_numpy_scalar' in hdf.attrs:
            # extract the scalar from the 0D array as is
            obj = hdf[()]
        else:
            # read array-dataset into an array
            obj = np.empty(hdf.shape, hdf.dtype)
            if obj.size:
                hdf.read_direct(obj)
    else:
        # check if we have a class instance definition here
        if not ('class' in hdf.attrs or 'recon' in hdf.attrs):
            raise LookupError("Found hdf group without class instance "
                    "information (group: %s). Cannot convert it into an "
                    "object (content: '%s', attributes: '%s')."
                    % (hdf.name, hdf.keys(), hdf.attrs.keys()))

        mod_name = hdf.attrs['module']

        if 'recon' in hdf.attrs:
            # Custom objects custom reconstructor
            obj = _recon_customobj_customrecon(hdf, memo)
        elif mod_name != '__builtin__':
            # Custom objects default reconstructor
            cls_name = hdf.attrs['class']
            if cls_name in ('function', 'type', 'builtin_function_or_method'):
                # Functions and types
                obj = _recon_functype(hdf)
            else:
                # Other custom objects
                obj = _recon_customobj_defaultrecon(hdf, memo)
        else:
            # Built-in objects
            cls_name = hdf.attrs['class']
            if __debug__:
                debug('HDF5', "Reconstructing built-in object '%s'." % cls_name)
            # built in type (there should be only 'list', 'dict' and 'None'
            # that would not be in a Dataset
            if cls_name == 'NoneType':
                obj = None
            elif cls_name == 'tuple':
                obj = _hdf_tupleitems_to_obj(hdf, memo)
            elif cls_name == 'list':
                # could be used also for storing object ndarrays
                if 'is_objarray' in hdf.attrs:
                    obj = _hdf_list_to_objarray(hdf, memo)
                else:
                    obj = _hdf_list_to_obj(hdf, memo)
            elif cls_name == 'dict':
                obj = _hdf_dict_to_obj(hdf, memo)
            elif cls_name == 'type':
                obj = eval(hdf.attrs['name'])
            elif cls_name == 'function':
                raise RuntimeError("Unhandled reconstruction of built-in "
                        "function (at '%s')." % hdf.name)
            else:
                raise RuntimeError("Found hdf group with a builtin type "
                        "that is not handled by the parser (group: %s). This "
                        "is a conceptual bug in the parser. Please report."
                        % hdf.name)
    #
    # Final post-processing
    #

    # track if desired
    if objref:
        if __debug__:
            debug('HDF5', "Placing %s objref '%s' to memo", (obj, objref))
        memo[objref] = obj
    if __debug__:
        debug('HDF5', "Done loading %s [%s]"
                      % (type(obj), hdf.name))
    return obj


def _recon_functype(hdf):
    """Reconstruct a function or type from HDF"""
    cls_name = hdf.attrs['class']
    mod_name = hdf.attrs['module']
    ft_name = hdf.attrs['name']
    if __debug__:
        debug('HDF5', "Load '%s.%s.%s' [%s]"
                      % (mod_name, cls_name, ft_name, hdf.name))
    mod, obj = _import_from_thin_air(mod_name, ft_name, cls_name=cls_name)
    return obj

def _get_subclass_entry(cls, clss, exc_msg="", exc=NotImplementedError):
    """In a list of tuples (cls, ...) return the entry for the first
    occurrence of the class of which `cls` is a subclass of.
    Otherwise raise `exc` with the given message"""

    for clstuple in clss:
        if issubclass(cls, clstuple[0]):
            return clstuple
    raise exc(exc_msg % locals())

def _update_obj_state_from_hdf(obj, hdf, memo):
    if 'state' in hdf:
        # insert the state of the object
        if __debug__:
            debug('HDF5', "Populating instance state.")
        if hasattr(obj, '__setstate__'):
            state = hdf2obj(hdf['state'], memo)
            obj.__setstate__(state)
        else:
            state = _hdf_dict_to_obj(hdf['state'], memo)
            if state:
                obj.__dict__.update(state)
        if __debug__:
            debug('HDF5', "Updated %i state items." % len(state))

def _recon_customobj_customrecon(hdf, memo):
    """Reconstruct a custom object from HDF using a custom recontructor"""
    # we found something that has some special idea about how it wants
    # to be reconstructed
    mod_name = hdf.attrs['module']
    recon_name = hdf.attrs['recon']
    if __debug__:
        debug('HDF5', "Load from custom reconstructor '%s.%s' [%s]"
                      % (mod_name, recon_name, hdf.name))
    # turn names into definitions
    mod, recon = _import_from_thin_air(mod_name, recon_name)

    obj = None
    if 'rcargs' in hdf:
        recon_args_hdf = hdf['rcargs']
        if __debug__:
            debug('HDF5', "Load reconstructor args in [%s]"
                          % recon_args_hdf.name)
        if 'objref' in hdf.attrs:
            # XXX TODO YYY ZZZ WHATEVER
            # yoh: the problem is that inside this beast might be references
            # to current, not yet constructed object, and if we follow
            # Python docs we should call recon with *recon_args, thus we
            # cannot initiate the beast witout them.  But if recon is a class
            # with __new__ we could may be first __new__ and then only __init__
            # with recon_args?
            if '__new__' in dir(recon):
                try:
                    # TODO: what if multiple inheritance?
                    obj = recon.__bases__[0].__new__(recon)
                except:
                    # try direct __new__
                    try:
                        obj = recon.__new__()
                    except:
                        # give up and hope for the best
                        obj = None
                if obj is not None:
                    memo[hdf.attrs['objref']] = obj
        recon_args = _hdf_tupleitems_to_obj(recon_args_hdf, memo)
    else:
        recon_args = ()

    # reconstruct
    if obj is None:
        obj = recon(*recon_args)
    else:
        # Let only to init it
        obj.__init__(*recon_args)
    # insert any stored object state
    _update_obj_state_from_hdf(obj, hdf, memo)
    return obj


def _import_from_thin_air(mod_name, importee, cls_name=None):
    if cls_name is None:
        cls_name = importee
    try:
        mod = __import__(mod_name, fromlist=[importee])
    except ImportError, e:
        if mod_name.startswith('mvpa') and not mod_name.startswith('mvpa2'):
            # try to be gentle on data that got stored with PyMVPA 0.5 or 0.6
            mod_name = mod_name.replace('mvpa', 'mvpa2', 1)
            mod = __import__(mod_name, fromlist=[cls_name])
        else:
            raise e
    try:
        imp = mod.__dict__[importee]
    except KeyError:
        mod_name, importee = universal_classname_remapper[(mod_name, importee)]
        mod = __import__(mod_name, fromlist=[cls_name])
        imp = mod.__dict__[importee]
    return mod, imp


def _recon_customobj_defaultrecon(hdf, memo):
    """Reconstruct a custom object from HDF using the default recontructor"""
    cls_name = hdf.attrs['class']
    mod_name = hdf.attrs['module']
    if __debug__:
        debug('HDF5', "Load class instance '%s.%s' instance [%s]"
                      % (mod_name, cls_name, hdf.name))
    mod, cls = _import_from_thin_air(mod_name, cls_name)

    # create the object
    # use specialized __new__ if necessary or beneficial
    pcls, = _get_subclass_entry(cls, ((dict,), (list,), (object,)),
                                "Do not know how to create instance of %(cls)s")
    obj = pcls.__new__(cls)
    # insert any stored object state
    _update_obj_state_from_hdf(obj, hdf, memo)

    # do we process a container?
    if 'items' in hdf:
        # charge the items -- handling depends on the parent class
        pcls, umeth, cfunc = _get_subclass_entry(
            cls,
            ((dict, 'update', _hdf_dict_to_obj),
             (list, 'extend', _hdf_list_to_obj)),
            "Unhandled container type (got: '%(cls)s').")
        if __debug__:
            debug('HDF5', "Populating %s object." % pcls)
        getattr(obj, umeth)(cfunc(hdf, memo))
        if __debug__:
            debug('HDF5', "Loaded %i items." % len(obj))

    return obj


def _hdf_dict_to_obj(hdf, memo, skip=None):
    if skip is None:
        skip = []
    # legacy compat code
    if not 'items' in hdf:
        items_container = hdf
    # end of legacy compat code
    else:
        items_container = hdf['items']

    if items_container.attrs.get('__keys_in_tuple__', 0):
        # pre-create the object so it could be correctly
        # objref'ed/used in memo
        d = dict()
        items = _hdf_list_to_obj(hdf, memo, target_container=d)
        # some time back we had attribute names stored as arrays
        for k, v in items:
            if k in skip:
                continue
            try:
                d[k] = v
            except TypeError:
                # fucked up dataset -- trying our best
                if isinstance(k, np.ndarray):
                    d[np.asscalar(k)] = v
                else:
                    # no idea, really
                    raise
        return d
    else:
        # legacy files had keys as group names
        return dict([(item, hdf2obj(items_container[item], memo=memo))
                        for item in items_container
                            if not item in skip])

def _hdf_list_to_objarray(hdf, memo):
    if not ('shape' in hdf.attrs):
        if __debug__:
            debug('HDF5', "Enountered objarray stored without shape (due to a bug "
                "in post 2.1 release).  Some nested structures etc might not be "
                "loaded incorrectly")
        # yoh: we have possibly a problematic case due to my fix earlier
        # resolve to old logic:  nested referencing might not work :-/
        obj = _hdf_list_to_obj(hdf, memo)
        # need to handle special case of arrays of objects
        if np.isscalar(obj):
            obj = np.array(obj, dtype=np.object)
        else:
            obj = asobjarray(obj)
    else:
        shape = tuple(hdf.attrs['shape'])
        # reserve space first
        if len(shape):
            obj = np.empty(np.prod(shape), dtype=object)
        else:
            # scalar
            obj = np.array(None, dtype=object)
        # now load the items from the list, noting existence of this
        # container
        obj_items = _hdf_list_to_obj(hdf, memo, target_container=obj)
        # assign to the object array
        for i, v in enumerate(obj_items):
            obj[i] = v
        if len(shape) and shape != obj.shape:
            obj = obj.reshape(shape)
    return obj

def _hdf_list_to_obj(hdf, memo, target_container=None):
    """Convert an HDF item sequence into a list

    Lists are used for storing also dicts.  To properly reference
    the actual items in memo, target_container could be specified
    to point to the actual data structure to be referenced, which
    later would get populated with list's items.
    """
    # new-style files have explicit length
    if 'length' in hdf.attrs:
        length = hdf.attrs['length']
        if __debug__:
            debug('HDF5', "Found explicit sequence length setting (%i)"
                          % length)
        hdf_items = hdf['items']
    elif 'items' in hdf:
        # not so legacy file, at least has an items container
        length = len(hdf['items'])
        if __debug__:
            debug('HDF5', "No explicit sequence length setting (guess: %i)"
                          % length)
        hdf_items = hdf['items']
    # legacy compat code
    else:
        length = len(hdf)
        if __debug__:
            debug('HDF5', "Ancient file, guessing sequence length (%i)"
                          % length)
        # really legacy file, not even items container
        hdf_items = hdf
    # end of legacy compat code

    # prepare item list
    items = [None] * length
    # need to put items list in memo before starting to parse to allow to detect
    # self-inclusion of this list in itself
    if 'objref' in hdf.attrs:
        objref = hdf.attrs['objref']
        if target_container is None:
            if __debug__:
                debug('HDF5', "Track sequence with %i elements under objref '%s'"
                              % (length, objref))
            memo[objref] = items
        else:
            if __debug__:
                debug('HDF5', "Track provided target_container under objref '%s'",
                      objref)
            memo[objref] = target_container
    # for all expected items
    for i in xrange(length):
        if __debug__:
            debug('HDF5', "Item %i" % i)
        str_i = str(i)
        obj = None
        objref = None
        # we need a separate flag, see below
        got_obj = False
        # do we have an item attribute for this item (which is the objref)
        if str_i in hdf_items.attrs:
            objref = hdf_items.attrs[str_i]
        # do we have an actual value for this item
        if str_i in hdf_items:
            obj = hdf2obj(hdf_items[str_i], memo=memo)
            # we need to signal that we got something, since it could as well
            # be None
            got_obj = True
        if not got_obj:
            # no actual value for item
            if objref is None:
                raise LookupError("Cannot find list item '%s'" % str_i)
            else:
                # no value but reference -> value should be in memo
                if objref in memo:
                    if __debug__:
                        debug('HDF5', "Use tracked object (%i)"
                                      % objref)
                    items[i] = memo[objref]
                else:
                    raise LookupError("No value for objref '%i'" % objref)
        else:
            # we have a value for this item
            items[i] = obj
            # store value for ref if present
            if not objref is None:
                memo[objref] = obj

    return items


def _hdf_tupleitems_to_obj(hdf, memo):
    """Same as _hdf_list_to_obj, but converts to tuple upon return"""
    return tuple(_hdf_list_to_obj(hdf, memo))


def _seqitems_to_hdf(obj, hdf, memo, noid=False, **kwargs):
    """Store a sequence as HDF item list"""
    hdf.attrs.create('length', len(obj))
    items = hdf.create_group('items')
    for i, item in enumerate(obj):
        if __debug__:
            debug('HDF5', "Item %i" % i)
        obj2hdf(items, item, name=str(i), memo=memo, noid=noid, **kwargs)


def obj2hdf(hdf, obj, name=None, memo=None, noid=False, **kwargs):
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
    memo : dict
      Dictionary tracking stored objects to prevent recursions (analog to
      deepcopy).
    noid : bool
      If True, the to be processed object has no usable id. Set if storing
      objects that were created temporarily, e.g. during type conversions.
    **kwargs
      All additional arguments will be passed to `h5py.Group.create_dataset()`
    """
    if memo is None:
        # initialize empty recursion tracker
        memo = {}

    #
    # Catch recursions: just stored references to already known objects
    #
    if noid:
        # noid: tracking this particular object is not intended
        obj_id = 0
    else:
        obj_id = id(obj)
    if not noid and obj_id in memo:
        # already in here somewhere, nothing else but reference needed
        # this can also happen inside containers, so 'name' should not be None
        hdf.attrs.create(name, obj_id)
        if __debug__:
            debug('HDF5', "Store '%s' by objref: %i" % (type(obj), obj_id))
        # done
        return

    #
    # Ugly special case of arrays of objects
    #
    is_objarray = False                # assume the bright side ;-)
    is_ndarray = isinstance(obj, np.ndarray)
    if is_ndarray:
        if obj.dtype == np.object:
            shape = obj.shape
            if not len(obj.shape):
                # even worse: 0d array
                # we store 0d object arrays just by content
                if __debug__:
                    debug('HDF5', "0d array(object) -> object")
                obj = obj[()]
            else:
                # proper arrays can become lists
                if __debug__:
                    debug('HDF5', "array(objects) -> list(objects)")
                obj = list(obj.flatten())
                # make sure we don't ref this temporary list object
                # noid = True
                # yoh: obj_id is of the original obj here so should
                # be stored
            # flag that we messed with the original type
            is_objarray = True
            # and re-estimate the content's nd-array-ness
            is_ndarray = isinstance(obj, np.ndarray)

    # if it is something that can go directly into HDF5, put it there
    # right away
    is_scalar = np.isscalar(obj)
    if is_scalar or is_ndarray:
        is_numpy_scalar = issubclass(type(obj), np.generic)
        if name is None:
            # HDF5 cannot handle datasets without a name
            name = '__unnamed__'
        if __debug__:
            debug('HDF5', "Store '%s' (ref: %i) in [%s/%s]"
                          % (type(obj), obj_id, hdf.name, name))
        # the real action is here
        if 'compression' in kwargs \
               and (is_scalar or (is_ndarray and not len(obj.shape))):
            # recent (>= 2.0.0) h5py is strict not allowing
            # compression to be set for scalar types or anything with
            # shape==() ... TODO: check about is_objarrays ;-)
            kwargs = dict([(k, v) for (k, v) in kwargs.iteritems()
                           if k != 'compression'])
        hdf.create_dataset(name, None, None, obj, **kwargs)
        if not noid and not is_scalar:
            # objref for scalar items would be overkill
            hdf[name].attrs.create('objref', obj_id)
            # store object reference to be able to detect duplicates
            if __debug__:
                debug('HDF5', "Record objref in memo-dict (%i)" % obj_id)
            memo[obj_id] = obj

        ## yoh: was not sure why we have to assign here as well as below to grp
        ##      so commented out and seems to work just fine ;)
        ## yoh: because it never reaches grp! (see return below)
        if is_objarray:
            # we need to confess the true origin
            hdf[name].attrs.create('is_objarray', True)
            # it was of more than 1 dimension or it was a scalar
            if not len(shape) and externals.versions['hdf5'] < '1.8.7':
                if __debug__:
                    debug('HDF5', "Versions of hdf5 before 1.8.7 have problems with empty arrays")
            else:
                hdf[name].attrs.create('shape', shape)

        # handle scalars giving numpy scalars different flag
        if is_numpy_scalar:
            hdf[name].attrs.create('is_numpy_scalar', True)
        elif is_scalar:
            hdf[name].attrs.create('is_scalar', True)
        return

    #
    # Below handles stuff that cannot be natively stored in HDF5
    #
    if not name is None:
        if __debug__:
            debug('HDF5', "Store '%s' (ref: %i) in [%s/%s]"
                          % (type(obj), obj_id, hdf.name, name))
        grp = hdf.create_group(str(name))
    else:
        # XXX wouldn't it be more coherent to always have non-native objects in
        # a separate group
        if __debug__:
            debug('HDF5', "Store '%s' (ref: %i) in [%s]"
                          % (type(obj), obj_id, hdf.name))
        grp = hdf

    #
    # Store important flags and references in the group meta data
    #
    if not noid and not obj is None:
        # no refs for basic types
        grp.attrs.create('objref', obj_id)
        # we also note that we processed this object
        memo[obj_id] = obj

    if is_objarray:
        # we need to confess the true origin
        grp.attrs.create('is_objarray', True)
        grp.attrs.create('shape', shape)

    # standard containers need special treatment
    if not hasattr(obj, '__reduce__'):
        raise HDF5ConversionError("Cannot store class without __reduce__ "
                                  "implementation (%s)" % type(obj))
    # try disassembling the object
    try:
        pieces = obj.__reduce__()
        if __debug__:
            debug('HDF5', "Reduced '%s' (ref: %i) in [%s]"
                          % (type(obj), obj_id, hdf.name))
    except TypeError as te:
        # needs special treatment
        pieces = None
        if __debug__:
            debug('HDF5', "Failed to reduce '%s' (ref: %i) in [%s]: %s" # (%s)"
                          % (type(obj), obj_id, hdf.name, te)) #, obj))

    # common container handling, either __reduce__ was not possible
    # or it was the default implementation
    if pieces is None or pieces[0].__name__ == '_reconstructor':
        # figure out the source module
        if hasattr(obj, '__module__'):
            src_module = obj.__module__
        else:
            src_module = obj.__class__.__module__

        cls_name = obj.__class__.__name__
        # special case: metaclass types NOT instance of a class with metaclass
        if hasattr(obj, '__metaclass__') and hasattr(obj, '__base__'):
            cls_name = 'type'

        if src_module != '__builtin__':
            if hasattr(obj, '__name__'):
                if not obj.__name__ in dir(__import__(src_module,
                                                      fromlist=[obj.__name__])):
                    raise HDF5ConversionError("Cannot store locally defined "
                                              "function '%s'" % cls_name)
            else:
                if not cls_name in dir(__import__(src_module,
                                                  fromlist=[cls_name])):
                    raise HDF5ConversionError("Cannot store locally defined "
                                              "class '%s'" % cls_name)
        # store class info (fully-qualified)
        grp.attrs.create('class', cls_name)
        grp.attrs.create('module', src_module)

        if hasattr(obj, '__name__'):
            # for functions/types we need a name for reconstruction
            oname = obj.__name__
            if oname == '<lambda>':
                raise HDF5ConversionError(
                    "Can't obj2hdf lambda functions. Got %r" % (obj,))
            grp.attrs.create('name', oname)
        if isinstance(obj, list) or isinstance(obj, tuple):
            _seqitems_to_hdf(obj, grp, memo, **kwargs)
        elif isinstance(obj, dict):
            if __debug__:
                debug('HDF5', "Store dict as zipped list")
            # need to set noid since outer tuple containers are temporary
            _seqitems_to_hdf(zip(obj.keys(), obj.values()), grp, memo,
                             noid=True, **kwargs)
            grp['items'].attrs.create('__keys_in_tuple__', 1)

    else:
        if __debug__:
            debug('HDF5', "Use custom __reduce__ for storage: (%i arguments)."
                          % len(pieces[1]))
        grp.attrs.create('recon', pieces[0].__name__)
        grp.attrs.create('module', pieces[0].__module__)
        args = grp.create_group('rcargs')
        _seqitems_to_hdf(pieces[1], args, memo, **kwargs)

    # pull all remaining data from __reduce__
    if not pieces is None and len(pieces) > 2:
        # there is something in the state
        state = pieces[2]
        if __debug__:
            if state is not None:
                debug('HDF5', "Store object state (%i items)." % len(state))
            else:
                debug('HDF5', "Storing object with None state")
        # need to set noid since state dict is unique to an object
        obj2hdf(grp, state, name='state', memo=memo, noid=True,
                **kwargs)


def h5save(filename, data, name=None, mode='w', mkdir=True, **kwargs):
    """Stores arbitrary data in an HDF5 file.

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
    mkdir : bool, optional
      Create target directory if it does not exist yet.
    **kwargs
      All additional arguments will be passed to `h5py.Group.create_dataset`.
      This could, for example, be `compression='gzip'`.
    """
    if mkdir:
        target_dir = osp.dirname(filename)
        if target_dir and not osp.exists(target_dir):
            os.makedirs(target_dir)
    hdf = h5py.File(filename, mode)
    hdf.attrs.create('__pymvpa_hdf5_version__', '2')
    hdf.attrs.create('__pymvpa_version__', mvpa2.__version__)
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
                # stored objects can only by special groups or datasets
                if isinstance(hdf, h5py.Dataset) \
                   or ('class' in hdf.attrs or 'recon' in hdf.attrs):
                    # this is an object stored at the toplevel
                    obj = hdf2obj(hdf)
                else:
                    # no object into at the top-level, but maybe in the next one
                    # this would happen for plain mat files with arrays
                    if len(hdf) == 1 and '__unnamed__' in hdf:
                        # just a single with special name -> special case:
                        # return as is
                        obj = hdf2obj(hdf['__unnamed__'])
                    else:
                        # otherwise build dict with content
                        obj = {}
                        for k in hdf:
                            obj[k] = hdf2obj(hdf[k])
    finally:
        hdf.close()
    return obj
