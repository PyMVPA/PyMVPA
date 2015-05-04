# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Module with some special objects to be used as magic attributes with
dedicated containers aka. `Collections`.
"""

__docformat__ = 'restructuredtext'

import copy, re
import numpy as np

from mvpa2.base.dochelpers import _str, borrowdoc
from mvpa2.base.types import is_sequence_type

if __debug__:
    # we could live without, but it would be nicer with it
    try:
        from mvpa2.base import debug
        __mvpadebug__ = True
    except ImportError:
        __mvpadebug__ = False


_object_getattribute = dict.__getattribute__
_object_setattr = dict.__setattr__
_object_setitem = dict.__setitem__

# To validate fresh
_dict_api = set(dict.__dict__)

class Collectable(object):
    """Collection element.

    A named single item container that allows for type, or property checks of
    an assigned value, and also offers utility functionality.
    """
    def __init__(self, value=None, name=None, doc=None):
        """
        Parameters
        ----------
        value : arbitrary (see derived implementations)
          The actual value of this attribute.
        name : str
          Name of the collectable under which it should be available in its
          respective collection.
        doc : str
          Documentation about the purpose of this collectable.
        """
        if doc is not None:
            # to prevent newlines in the docstring
            try:
                doc = re.sub('[\n ]+', ' ', doc)
            except TypeError:
                # catch some old datasets stored in HDF5
                doc = re.sub('[\n ]+', ' ', np.asscalar(doc))

        self.__doc__ = doc
        self.__name = name
        self._value = None
        if not value is None:
            self._set(value)
        if __debug__ and __mvpadebug__:
            debug("COL", "Initialized %r", (self,))


    def __copy__(self):
        # preserve attribute type
        copied = self.__class__(name=self.name, doc=self.__doc__)
        # just get a view of the old data!
        copied.value = copy.copy(self.value)
        return copied

    ## def __deepcopy__(self, memo=None):
    ##     # preserve attribute type
    ##     copied = self.__class__(name=self.name, doc=self.__doc__)
    ##     # get a deepcopy of the old data!
    ##     copied._value = copy.deepcopy(self._value, memo)
    ##     return copied

    def _get(self):
        return self._value


    def _set(self, val):
        if __debug__ and __mvpadebug__:
            # Since this call is quite often, don't convert
            # values to strings here, rely on passing them
            # withing msgargs
            debug("COL", "Setting %s to %s ", (self, val))
        self._value = val


    def __str__(self):
        res = "%s" % (self.name)
        return res


    def __reduce__(self):
        return (self.__class__,
                    (self._value, self.name, self.__doc__))


    def __repr__(self):
        value = self.value
        return "%s(name=%s, doc=%s, value=%s)" % (self.__class__.__name__,
                                                  repr(self.name),
                                                  repr(self.__doc__),
                                                  repr(value))


    def _get_name(self):
        return self.__name


    def _set_name(self, name):
        """Set the name of parameter

        Notes
        -----
        Should not be called for an attribute which is already assigned
        to a collection
        """
        if name is not None:
            if isinstance(name, basestring):
                if name[0] == '_':
                    raise ValueError, \
                          "Collectable attribute name must not start " \
                          "with _. Got %s" % name
            else:
                raise ValueError, \
                      "Collectable attribute name must be a string. " \
                      "Got %s" % `name`
        self.__name = name


    # Instead of going for VProperty lets make use of virtual method
    def _get_virtual(self):
        return self._get()


    def _set_virtual(self, value):
        return self._set(value)


    value = property(_get_virtual, _set_virtual)
    name = property(_get_name, _set_name)


class SequenceCollectable(Collectable):
    """Collectable to handle sequences.

    It takes care about caching and recomputing unique values, as well as
    optional checking if assigned sequences have a desired length.
    """
    def __init__(self, value=None, name=None, doc="Sequence attribute",
                 length=None):
        """
        Parameters
        ----------
        value : arbitrary (see derived implementations)
          The actual value of this attribute.
        name : str
          Name of the attribute under which it should be available in its
          respective collection.
        doc : str
          Documentation about the purpose of this attribute.
        length : int
          If not None, enforce any array assigned as value of this collectable
          to be of this `length`. If an array does not match this requirement
          it is not modified, but a ValueError is raised.
        """
        # first configure the value checking, to enable it for the base class
        # init
        # XXX should we disallow empty Collectables??
        if not value is None and not hasattr(value, '__len__'):
            raise ValueError("%s only takes sequences as value."
                             % self.__class__.__name__)
        self._target_length = length
        Collectable.__init__(self, value=value, name=name, doc=doc)
        self._reset_unique()


    def __reduce__(self):
        return (self.__class__,
                    (self.value, self.name, self.__doc__, self._target_length))


    def __repr__(self):
        value = self.value
        return "%s(name=%s, doc=%s, value=%s, length=%s)" \
                    % (self.__class__.__name__,
                       repr(self.name),
                       repr(self.__doc__),
                       repr(value),
                       repr(self._target_length))


    def __len__(self):
        return self.value.__len__()


    def __getitem__(self, key):
        return self.value.__getitem__(key)


    def _set(self, val):
        # check if the new value has the desired length -- if length checking is
        # desired at all
        if not self._target_length is None \
           and len(val) != self._target_length:
            raise ValueError("Value length [%i] does not match the required "
                             "length [%i] of attribute '%s'."
                             % (len(val),
                                self._target_length,
                                str(self.name)))
        self._reset_unique()
        Collectable._set(self, val)


    def _reset_unique(self):
        self._unique_values = None


    @property
    def unique(self):
        """Return unique values
        """
        if self.value is None:
            return None
        if self._unique_values is None:
            try:
                self._unique_values = np.unique(self.value)
            except TypeError:
                # We are probably on Python 3 and value contains None's
                # or any other different type breaking the comparison
                # so operate through set()
                # See http://projects.scipy.org/numpy/ticket/2188

                # Get a 1-D array
                #  list around set is required for Python3
                value_unique = list(set(np.asanyarray(self.value).ravel()))
                try:
                    self._unique_values = np.array(value_unique)
                except ValueError:
                    # without forced dtype=object it might have failed due to
                    # something related to
                    # http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=679948
                    # which was fixed recently...
                    self._unique_values = np.array(value_unique, dtype=object)
        return self._unique_values


    def set_length_check(self, value):
        """Set a target length of the value in this collectable.

        Parameters
        ----------
        value : int
          If not None, enforce any array assigned as value of this collectable
          to be of this `length`. If an array does not match this requirement
          it is not modified, but a ValueError is raised.
        """
        self._target_length = value



class ArrayCollectable(SequenceCollectable):
    """Collectable embedding an array.

    When shallow-copied it includes a view of the array in the copy.
    """
    def __copy__(self):
        # preserve attribute type
        copied = self.__class__(name=self.name, doc=self.__doc__,
                                length=self._target_length)
        # just get a view of the old data!
        copied.value = self.value.view()
        return copied


    def _set(self, val):
        if not hasattr(val, 'view'):
            if is_sequence_type(val):
                try:
                    val = np.asanyarray(val)
                except ValueError, e:
                    if "setting an array element with a sequence" in str(e):
                        val = np.asanyarray(val, dtype=object)
                    else:
                        raise
            else:
                raise ValueError("%s only takes ndarrays (or array-likes "
                                 "providing view(), or sequence that can "
                                 "be converted into arrays (got '%s')."
                                 % (self.__class__.__name__,
                                    str(type(val))))
        SequenceCollectable._set(self, val)


class SampleAttribute(ArrayCollectable):
    """Per sample attribute in a dataset"""
    pass

class FeatureAttribute(ArrayCollectable):
    """Per feature attribute in a dataset"""
    pass

class DatasetAttribute(ArrayCollectable):
    """Dataset attribute"""
    pass



class Collection(dict):
    """Container of some Collectables.
    """
    def __init__(self, items=None):
        """
        Parameters
        ----------
        items : all types accepted by update()
        """
        dict.__init__(self)
        if not items is None:
            self.update(items)

    def copy(self, deep=True, a=None, memo=None):
        """Create a copy of a collection.

        By default this is going to return a deep copy of the
        collection, hence no data would be shared between the original
        dataset and its copy.

        Parameters
        ----------
        deep : boolean, optional
          If False, a shallow copy of the collection is return instead. The copy
          contains only views of the values.
        a : list or None
          List of attributes to include in the copy of the dataset. If
          `None` all attributes are considered. If an empty list is
          given, all attributes are stripped from the copy.
        memo : dict
          Developers only: This argument is only useful if copy() is called
          inside the __deepcopy__() method and refers to the dict-argument
          `memo` in the Python documentation.
        """

        # create the new collections of the right type derived classes
        # might like to assure correct setting of additional
        # attributes such as self._attr_length
        anew = self.__class__()

        # filter the attributes if necessary
        if a is None:
            aorig = self
        else:
            aorig = dict([(k, v) for k, v in self.iteritems() if k in a])

        # XXX copyvalues defaults to None which provides capability to
        #     just bind values (not even 'copy').  Might it need be
        #     desirable here?
        anew.update(aorig, copyvalues=deep and 'deep' or 'shallow',
                    memo=memo)

        if __debug__ and __mvpadebug__ and 'COL' in debug.active:
            debug("COL", "Copied %s into %s using args deep=%r a=%r",
                  (self, anew, deep, a))
            #if 'state2' in str(self):
            #    import pydb; pydb.debugger()
        return anew

    # XXX If enabled, then overrides dict.__reduce* leading to conditional
    #     attributes loosing their documentations in copying etc.
    #
    #def __copy__(self):
    #    return self.copy(deep=False)
    #
    #
    #def __deepcopy__(self, memo=None):
    #    return self.copy(deep=True, memo=memo)


    def __setitem__(self, key, value):
        """Add a new Collectable to the collection

        Parameters
        ----------
        key : str
          The name of the collectable under which it is available in the
          collection. This name is also stored in the item itself
        value : anything
          The actual item the should become part of the collection. If this is
          not an instance of `Collectable` or a subclass the value is
          automatically wrapped into it.
        """
        # Check if given key is not trying to override anything in
        # dict interface
        if key in _dict_api:
            raise ValueError, \
                  "Cannot add a collectable %r to collection %s since an " \
                  "attribute or a method with such a name is already present " \
                  "in dict interface.  Choose some other name." % (key, self)
        if not isinstance(value, Collectable):
            value = Collectable(value, name=key)
        else:
            if not value.name: # None assigned -- just use the Collectable
                value.name = key
            elif value.name == key: # assigned the same -- use the Collectable
                pass
            else: # use the copy and assign new name
                # see https://github.com/PyMVPA/PyMVPA/issues/149
                # for the original issue.  __copy__ directly to avoid copy.copy
                # doing the same + few more checks
                value = value.__copy__()
                # overwrite the Collectable's name with the given one
                value.name = key

        _object_setitem(self, key, value)


    def update(self, source, copyvalues=None, memo=None):
        """
        Parameters
        ----------
        source : list, Collection, dict
        copyvalues : None, shallow, deep
          If None, values will simply be bound to the collection items'
          values thus sharing the same instance. 'shallow' and 'deep' copies use
          'copy' and 'deepcopy' correspondingly.
        memo : dict
          Developers only: This argument is only useful if copy() is called
          inside the __deepcopy__() method and refers to the dict-argument
          `memo` in the Python documentation.
        """
        if isinstance(source, list):
            for a in source:
                if isinstance(a, tuple):
                    #list of tuples, e.g. from dict.items()
                    name = a[0]
                    value = a[1]
                else:
                    # list of collectables
                    name = a.name
                    value = a

                if copyvalues is None:
                    self[name] = value
                elif copyvalues == 'shallow':
                    self[name] = copy.copy(value)
                elif copyvalues == 'deep':
                    self[name] = copy.deepcopy(value, memo)
                else:
                    raise ValueError("Unknown value ('%s') for copy argument."
                                     % copy)
        elif isinstance(source, dict):
            for k, v in source.iteritems():
                # expand the docs
                if isinstance(v, tuple):
                    value = v[0]
                    doc = v[1]
                else:
                    value = v
                    doc = None
                # add the attribute with optional docs
                if copyvalues is None:
                    self[k] = v
                elif copyvalues == 'shallow':
                    self[k] = copy.copy(v)
                elif copyvalues == 'deep':
                    self[k] = copy.deepcopy(v, memo)
                else:
                    raise ValueError("Unknown value ('%s') for copy argument."
                                     % copy)
                # store documentation
                self[k].__doc__ = doc
        else:
            raise ValueError("Collection.update() cannot handle '%s'."
                             % str(type(source)))


    def __getattribute__(self, key):
        try:
            return self[key].value
        except KeyError:
            return _object_getattribute(self, key)


    def __setattr__(self, key, value):
        try:
            self[key].value = value
        except KeyError:
            _object_setattr(self, key, value)
        except Exception, e:
            # catch any other exception in order to provide a useful error message
            errmsg = "parameter '%s' cannot accept value `%r` (%s)" % (key, value, str(e))
            try:
                cdoc = self[key].constraints.long_description()
                if cdoc[0] == '(' and cdoc[-1] == ')':
                    cdoc = cdoc[1:-1]
                errmsg += " [%s]" % cdoc
            except:
                pass
            raise ValueError(errmsg)


    # TODO: unify with the rest of __repr__ handling
    def __repr__(self):
        return "%s(items=%r)" \
                  % (self.__class__.__name__, self.values())


    def __str__(self):
        return _str(self, ','.join([str(k) for k in sorted(self.keys())]))



class UniformLengthCollection(Collection):
    """Container for attributes with the same length.
    """
    def __init__(self, items=None, length=None):
        """
        Parameters
        ----------
        length : int
          When adding items to the collection, they are checked if the have this
          length.
        """
        # cannot call set_length(), since base class __getattribute__ goes wild
        # before its __init__ is called.
        self._uniform_length = length
        Collection.__init__(self, items)


    def __reduce__(self):
        return (self.__class__,
                (self.items(), self._uniform_length))

    @borrowdoc(Collection)
    def copy(self, *args, **kwargs):
        # Create a generic copy of the collection
        anew = super(UniformLengthCollection, self).copy(*args, **kwargs)

        # if it had any attributes assigned, those should have set
        # attr_length already, otherwise lets assure that we copy the
        # correct one into the new instance
        if self.attr_length is not None and anew.attr_length is None:
            anew.set_length_check(self.attr_length)
        return anew


    def set_length_check(self, value):
        """
        Parameters
        ----------
        value : int
          When adding new items to the collection, they are checked if the have
          this length.
        """
        self._uniform_length = value
        for v in self.values():
            v.set_length_check(value)


    def __setitem__(self, key, value):
        """Add a new IndexedCollectable to the collection

        Parameters
        ----------
        item : IndexedCollectable
          or of derived class. Must have 'name' assigned.
        """
        # local binding
        ulength = self._uniform_length

        # XXX should we check whether it is some other Collectable?
        if not isinstance(value, ArrayCollectable):
            # if it is only a single element iterable, attempt broadcasting
            if is_sequence_type(value) and len(value) == 1 \
                    and not ulength is None:
                if ulength > 1:
                    # cannot use np.repeat, because it destroys dimensionality
                    value = [value[0]] * ulength
            value = ArrayCollectable(value)
        if ulength is None:
            ulength = len(value)
        elif not len(value.value) == ulength:
            raise ValueError("Collectable '%s' with length [%i] does not match "
                             "the required length [%i] of collection '%s'."
                             % (key,
                                len(value.value),
                                ulength,
                                str(self)))
        # tell the attribute to maintain the desired length
        value.set_length_check(ulength)
        Collection.__setitem__(self, key, value)

    @staticmethod
    def _compare_to_value(a, v, strict=True):
        """Helper to find elements within attribute matching the value

        value might be a multidimensional beast

        Parameters
        ----------
        strict: bool, optional
          If True, it would throw ValueError exception if provided value
          is not present, or incompatible.  If False, it would allow to proceed
          returning an empty mask (all values False)
        """
        r = a == v

        if isinstance(r, bool):
            # comparison collapsed to a single thing, must be False
            assert(r is False)
            raise ValueError("%r is not comparable to items among %s"
                             % (v, a))

        if a.ndim > 1:
            # we are dealing with multi-dimensional attributes.
            # then value we are looking for must be just 1 dimension less,
            # otherwise numpy would broadcast the value and match which is not
            # desired
            vshape = np.asanyarray(v).shape
            if vshape != a.shape[1:]:
                raise ValueError("Value %r you are looking for is of %s "
                                 "shape, whenever collection contains entries "
                                 "of shape %s" % (v, vshape, a.shape[1:]))

        # collapse all other dimensions.  numpy would have broadcasted
        # the leading dimension
        while r.ndim > 1:
            r = np.all(r, axis=1)
        if not np.any(r):
            if strict:
                raise ValueError("None of the items matched %r among %s"
                                 % (v, a))
            else:
                return np.zeros(len(a), dtype=bool)
        return r

    def match(self, d, strict=True):
        """Given a dictionary describing selection, return mask for matching items

        Given a dictionary with keys known to the collection, search for item
        attributes which would satisfy the selection.  E.g.

        >>> col = UniformLengthCollection({'roi': ['a', 'b', 'c', 'a']})
        >>> print col.match({'roi': ['a']})
        [ True False False  True]
        >>> print col.match({'roi': ['c', 'a']})
        [ True False  True  True]

        Multiple keys could be specified with desired matching values.
        Intersection of matchings is returned across different keys:

        >>> col = UniformLengthCollection({'roi': ['a', 'b', 'c', 'a'],
        ...                                'vox': [[1,0], [0,1], [1,0], [0, 1]]})
        >>> print col.match({'roi': ['c', 'a'], 'vox': [[0,1]]})
        [False False False  True]

        Parameters
        ----------
        d: dict
          Dict describing the selection.  Keys must be known to the collection
        strict: bool, optional
          If True, absent matching to any specified selection key/value pair
          would result in ValueError exception.  If False, it would allow to
          not have matches, but if only a single value for a key is given or none
          of the values match -- you will end up with empty selection.
        """
        mask = np.ones(self.attr_length, dtype=bool)

        for k, target_values in d.iteritems():
            if not k in self.keys():
                raise ValueError("%s is not known to %s" % (k, self))
            value = self[k].value
            target_values_mask = reduce(np.logical_or,
                                        [self._compare_to_value(value, target_value, strict=strict)
                                         for target_value in target_values])
            mask = np.logical_and(mask, target_values_mask)
        return mask

    attr_length = property(fget=lambda self:self._uniform_length,
                           doc="Uniform length of all attributes in a collection")



class SampleAttributesCollection(UniformLengthCollection):
    """Container for attributes of samples (i.e. labels, chunks...)
    """
    pass


class FeatureAttributesCollection(UniformLengthCollection):
    """Container for attributes of features
    """
    pass


class DatasetAttributesCollection(Collection):
    """Container for attributes of datasets (i.e. mappers, ...)
    """
    pass
