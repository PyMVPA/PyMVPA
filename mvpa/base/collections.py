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

from copy import copy


if __debug__:
    # we could live without, but it would be nicer with it
    try:
        from mvpa.base import debug
        __mvpadebug__ = True
    except ImportError:
        __mvpadebug__ = False


class Collectable(object):
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
        self.__doc__ = doc
        self.__name = name
        self._value = None
        if not value is None:
            self._set(value)
        if __mvpadebug__:
            debug("COL", "Initialized new collectable: %s" % `self`)


    def __copy__(self):
        # preserve attribute type
        copied = self.__class__(name=self.name, doc=self.__doc__)
        # just get a view of the old data!
        copied.value = copy(self.value)
        return copied


    def _get(self):
        return self._value


    def _set(self, val):
        if __mvpadebug__:
            # Since this call is quite often, don't convert
            # values to strings here, rely on passing them
            # withing msgargs
            debug("COL",
                  "Setting %(self)s to %(val)s ",
                  msgargs={'self':self, 'val':val})
        self._value = val


    def __str__(self):
        res = "%s" % (self.name)
        return res


    def __repr__(self):
        value = self.value
        return "%s(name=%s, doc=%s, value=%s)" % (self.__class__.__name__,
                                                  repr(self.name),
                                                  repr(self.__doc__),
                                                  repr(value))


    def _getName(self):
        return self.__name


    def _setName(self, name):
        """Set the name of parameter

        .. note::
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
    def _getVirtual(self):
        return self._get()


    def _setVirtual(self, value):
        return self._set(value)


    value = property(_getVirtual, _setVirtual)
    name = property(_getName, _setName)



class ArrayCollectable(Collectable):
    """Container which embeds an array.

    It also takes care about caching and recomputing unique values.
    """
    def __init__(self, value=None, name=None, doc="Attribute with array",
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
        self._target_length = length
        Collectable.__init__(self, name=name, doc=doc, value=value)
        self._resetUnique()
        if __mvpadebug__:
            debug("UATTR",
                  "Initialized new AttributeWithArray %s " % name + `self`)


    def __copy__(self):
        # preserve attribute type
        copied = self.__class__(name=self.name, doc=self.__doc__,
                                length=self._target_length)
        # just get a view of the old data!
        copied.value = self.value.view()
        return copied


    def __repr__(self):
        value = self.value
        return "%s(name=%s, doc=%s, value=%s, length=%s)" \
                    % (self.__class__.__name__,
                       repr(self.name),
                       repr(self.__doc__),
                       repr(value),
                       repr(self.__target_length))


    def _resetUnique(self):
        self._uniqueValues = None


    def _set(self, val):
        # check if the new value has the desired length -- if length checking is
        # desired at all
        if not self._target_length is None \
           and len(val) != self.__target_length:
            raise ValueError("Value length [%i] does not match the required "
                             "length [%i] of attribute '%s'."
                             % (len(val),
                                self.__target_length,
                                str(self.name)))

        self._resetUnique()
        CollectableAttribute._set(self, val)


    def set_length_check(self, value):
        """
        Parameters
        ----------
        value : int
          If not None, enforce any array assigned as value of this collectable
          to be of this `length`. If an array does not match this requirement
          it is not modified, but a ValueError is raised.
        """
        self._target_length = value


    def _getUniqueValues(self):
        if self.value is None:
            return None
        if self._uniqueValues is None:
            # XXX we might better use Set, but yoh recalls that
            #     N.unique was more efficient. May be we should check
            #     on the the class and use Set only if we are not
            #     dealing with ndarray (or lists/tuples)
            self._uniqueValues = N.unique(self.value)
        return self._uniqueValues


    unique = property(fget=_getUniqueValues)
