# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classes to control and store state information.

It was devised to provide conditional storage 
"""

_DEV_DOC = """
TODO:
+ Use %r instead of %s in repr for ClassWithCollections
  Replaced few %s's... might be fixed
+ Check why __doc__ is not set in kernels
  Seems to be there now...
- puke if *args and **kwargs are provided and exceed necessary number
+ for Parameter add ability to make it 'final'/read-only
+ repr(instance.params) contains only default value -- not current or
  set in the constructor... not good
  Now if value is not default -- would be present
? check/possible assure order of parameters/states to be as listed in the
  constructor
  There is _instance_index (could be set with 'index' parameter in
  Parameter). ATM it is used in documentation to list them in original
  order.
"""

# XXX: MH: The use of `index` as variable name confuses me. IMHO `index` refers
#          to a position in a container (i.e. list access). However, in this
#          file it is mostly used in the context of a `key` for dictionary
#          access. Can we refactor that?
# YOH: good point -- doing so... also we need to document somewhere that
#      names of Collectables are actually the keys in Collections
__docformat__ = 'restructuredtext'

import operator
import mvpa.support.copy as copy
from sets import Set
from textwrap import TextWrapper

# Although not used here -- included into interface
from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc.attributes import IndexedCollectable, StateVariable
from mvpa.base.dochelpers import enhancedDocString

from mvpa.base import externals
# XXX local rename is due but later on
from mvpa.base.collections import Collection as BaseCollection

if __debug__:
    from mvpa.base import debug
    # XXX
    # To debug references on top level -- useful to keep around for now,
    # don't remove until refactoring is complete
    import sys
    _debug_references = 'ATTRREFER' in debug.active
    _debug_shits = []                   # remember all to don't complaint twice
    import traceback

_in_ipython = externals.exists('running ipython env')
# Separators around definitions, needed for ReST, but bogus for
# interactive sessions
_def_sep = ('`', '')[int(_in_ipython)]

_object_getattribute = object.__getattribute__
_object_setattr = object.__setattr__

###################################################################
# Collections
#
# TODO:
#  - refactor: use base.collections and unify this to that
#  - minimize interface


class Collection(BaseCollection):
    """Container of some IndexedCollectables.

    :Groups:
     - `Access Implementors`: `listing`, `names`
     - `Mutators`: `__init__`
     - `R/O Properties`: `listing`, `names`, `items`

     XXX Seems to be not used and duplicating functionality: `listing`
     (thus `listing` property)
    """

    def __init__(self, items=None, owner=None, name=None):
        """Initialize the Collection

        Parameters
        ----------
        items : dict of IndexedCollectable's
          items to initialize with
        owner : object
          an object to which collection belongs
        name : str
          name of the collection (as seen in the owner, e.g. 'states')
        """
        # first set all stuff to nothing and later on charge it
        # this is important, since some of the stuff below relies in the
        # defaults
        self.__name = None
        self.__owner = None

        super(Collection, self).__init__(items)

        if not owner is None:
            self._set_owner(owner)
        if not name is None:
            self._set_name(name)



    def _set_name(self, name):
        self.__name = name

    def __str__(self):
        num = len(self)
        if __debug__ and "ST" in debug.active:
            maxnumber = 1000            # I guess all
        else:
            maxnumber = 4
        if self.__name is not None:
            res = self.__name
        else:
            res = ""
        res += "{"
        for i in xrange(min(num, maxnumber)):
            if i > 0:
                res += " "
            res += "%s" % str(self.values()[i])
        if len(self) > maxnumber:
            res += "..."
        res += "}"
        if __debug__:
            if "ST" in debug.active:
                res += " owner:%s#%s" % (self.owner.__class__.__name__,
                                         id(self.owner))
        return res


    def _cls_repr(self):
        """Collection specific part of __repr__ for a class containing
        it, ie a part of __repr__ for the owner object

        Returns
        -------
        list
          list of items to be appended within __repr__ after a .join()
        """
        # XXX For now we do not expect any pure non-specialized
        # collection , thus just override in derived classes
        raise NotImplementedError, "Class %s should override _cls_repr" \
              % self.__class__.__name__

    def _is_initializable(self, key):
        """Checks if key could be assigned within collection via
        _initialize

        Returns
        -------
        bool
          value for a given `key`

        It is to facilitate dynamic assignment of collections' items
        within derived classes' __init__ depending on the present
        collections in the class.
        """
        # XXX Each collection has to provide what indexes it allows
        #     to be set within constructor. Custom handling of some
        #     arguments (like (dis|en)able_states) is to be performed
        #     in _initialize
        # raise NotImplementedError, \
        #      "Class %s should override _is_initializable" \
        #      % self.__class__.__name__

        # YYY lets just check if it is in the keys
        return key in self.keys()


    def _initialize(self, key, value):
        """Initialize `key` (no check performed) with `value`
        """
        # by default we just set corresponding value
        self[key]._set(value, init=True)


    def __repr__(self):
        # do not include the owner arg, since that will most likely
        # cause recursions when the collection it self is included
        # into the repr() of the ClassWithCollections instance
        return "%s(items=%s, name=%s)" \
                  % (self.__class__.__name__,
                     repr(self.values()),
                     repr(self.name))

        items_s = ""
        sep = ""
        for item in self:
            try:
                itemvalue = "%r" % (self[item].value,)
                if len(itemvalue)>50:
                    itemvalue = itemvalue[:10] + '...' + itemvalue[-10:]
                items_s += "%s'%s':%s" % (sep, item, itemvalue)
                sep = ', '
            except:
                pass
        if items_s != "":
            s += "items={%s}" % items_s
        if self.owner is not None:
            s += "%sowner=%r" % (sep, self.owner)
        s += ")"
        return s


    def is_set(self, key=None):
        """If item (or any in the present or listed) was set

        Parameters
        ----------
        key : None or str or list of str
          What items to check if they were set in the collection
        """
        if not (key is None):
            if isinstance(key, basestring):
                return self[key].is_set
            else:
                items = key           # assume that we got some list
        else:
            items = self         # go through all the items

        for key in items:
            if self[key].is_set:
                return True
        return False


    def which_set(self):
        """Return list of keys which were set"""
        result = []
        # go through all members and if any is_set -- return True
        for key, v in self.iteritems():
            if v.is_set:
                result.append(key)
        return result


    # XXX RF to be removed if ownership feature is removed
    def __setitem__(self, key, value):
        super(Collection, self).__setitem__(key, value)
        if not self.owner is None:
            self._update_owner(name)


    # XXX RF to become pop?
    def remove(self, key):
        """Remove item from the collection
        """
        if not key in self:
            raise ValueError, "Key %s isn't known to collection %s" \
                  % (key, self)
        self._update_owner(key, register=False)
        _ = self.pop(key)


    def _action(self, key, func, missingok=False, **kwargs):
        """Run specific func either on a single item or on all of them

        Parameters
        ----------
        key : str
          Name of the state variable
        func
          Function (not bound) to call given an item, and **kwargs
        missingok : bool
          If True - do not complain about wrong key
        """
        if isinstance(key, basestring):
            if key.upper() == 'ALL':
                for key_ in self:
                    self._action(key_, func, missingok=missingok, **kwargs)
            else:
                try:
                    func(self[key], **kwargs)
                except:
                    if missingok:
                        return
                    raise
        elif operator.isSequenceType(key):
            for item in key:
                self._action(item, func, missingok=missingok, **kwargs)
        else:
            raise ValueError, \
                  "Don't know how to handle  variable given by %s" % key


    def reset(self, key=None):
        """Reset the state variable defined by `key`"""

        if not key is None:
            keys = [ key ]
        else:
            keys = self.names

        if len(self.items):
            for key in keys:
                # XXX Check if that works as desired
                self._action(key, self.values()[0].__class__.reset,
                             missingok=False)


    def _set_owner(self, owner):
        if not isinstance(owner, ClassWithCollections):
            raise ValueError, \
                  "Owner of the StateCollection must be ClassWithCollections object"
        if __debug__:
            try:    strowner = str(owner)
            except: strowner = "UNDEF: <%s#%s>" % (owner.__class__, id(owner))
            debug("ST", "Setting owner for %s to be %s" % (self, strowner))
        if not self.__owner is None:
            # Remove attributes which were registered to that owner previousely
            self._update_owner(register=False)
        self.__owner = owner
        if not self.__owner is None:
            self._update_owner(register=True)


    def _update_owner(self, key=None, register=True):
        """Define an entry within owner's __dict__
         so ipython could easily complete it

         Parameters
         ----------
         key : str or list of str
           Name of the attribute. If None -- all known get registered
         register : bool
           Register if True or unregister if False

         XXX Needs refactoring since we duplicate the logic of expansion of
         key value
        """
        # Yarik standing behind me, forcing me to do this -- I have no clue....
        if not (__debug__ and _debug_references):
            return
        if not key is None:
            if not key in self:
                raise ValueError, \
                      "Attribute %s is not known to %s" % (key, self)
            keys = [ key ]
        else:
            keys = self.names

        ownerdict = self.owner.__dict__
        selfdict = self.__dict__
        owner_known = ownerdict['_known_attribs']
        for key_ in keys:
            if register:
                if key_ in ownerdict:
                    raise RuntimeError, \
                          "Cannot register attribute %s within %s " % \
                          (key_, self.owner) + "since it has one already"
                ownerdict[key_] = self[key_]
                if key_ in selfdict:
                    raise RuntimeError, \
                          "Cannot register attribute %s within %s " % \
                          (key_, self) + "since it has one already"
                selfdict[key_] = self[key_]
                owner_known[key_] = self.__name
            else:
                if key_ in ownerdict:
                    # yoh doesn't think that we need to complain if False
                    ownerdict.pop(key_)
                    owner_known.pop(key_)
                if key_ in selfdict:
                    selfdict.pop(key_)

    # XXX RF: not used anywhere / myself -- hence not worth it?
    @property
    def listing(self):
        """Return a list of registered states along with the documentation"""

        # lets assure consistent litsting order
        items = self.items()
        items.sort()
        return [ "%s%s%s: %s" % (_def_sep, str(x[1]), _def_sep, x[1].__doc__)
                 for x in items ]

    # XXX RF: do we need those names?  use keys()
    @property
    def names(self):
        """Return ids for all registered state variables"""
        return self.keys()


    # Properties
    owner = property(fget=lambda x:x.__owner, fset=_set_owner)
    name = property(fget=lambda x:x.__name, fset=_set_name)


class ParameterCollection(Collection):
    """Container of Parameters for a stateful object.
    """

#    def __init__(self, items=None, owner=None, name=None):
#        """Initialize the state variables of a derived class
#
#        Parameters
#        ----------
#        items : dict
#          dictionary of states
#        """
#        Collection.__init__(self, items, owner, name)
#

    def _cls_repr(self):
        """Part of __repr__ for the owner object
        """
        prefixes = []
        for k in self.names:
            # list only params with not default values
            if self[k].is_default:
                continue
            prefixes.append("%s=%s" % (k, self[k].value))
        return prefixes


    def reset_value(self, key, missingok=False):
        """Reset all parameters to default values"""
        from param import Parameter
        self._action(key, Parameter.reset_value, missingok=False)


class StateCollection(Collection):
    """Container of StateVariables for a stateful object.

    :Groups:
     - `Public Access Functions`: `has_key`, `is_enabled`, `is_active`
     - `Access Implementors`: `listing`, `names`, `_get_enabled`
     - `Mutators`: `__init__`, `enable`, `disable`, `_set_enabled`
     - `R/O Properties`: `listing`, `names`, `items`
     - `R/W Properties`: `enabled`
    """

    def __init__(self, items=None, owner=None):
        """Initialize the state variables of a derived class

        Parameters
        ----------
        items : dict
          dictionary of states
        owner : ClassWithCollections
          object which owns the collection
        name : str
          literal description. Usually just attribute name for the
          collection, e.g. 'states'
        """
        Collection.__init__(self, items=items, owner=owner)

        self.__storedTemporarily = []
        """List to contain sets of enabled states which were enabled
        temporarily.
        """

    #
    # XXX TODO: figure out if there is a way to define proper
    #           __copy__'s for a hierarchy of classes. Probably we had
    #           to define __getinitargs__, etc... read more...
    #
    #def __copy__(self):

    def _cls_repr(self):
        """Part of __repr__ for the owner object
        """
        prefixes = []
        for name, invert in ( ('enable', False), ('disable', True) ):
            states = self._get_enabled(nondefault=False,
                                       invert=invert)
            if len(states):
                prefixes.append("%s_states=%s" % (name, str(states)))
        return prefixes


    def _is_initializable(self, key):
        """Checks if key could be assigned within collection via
        setvalue
        """
        return key in ['enable_states', 'disable_states']


    def _initialize(self, key, value):
        if value is None:
            value = []
        if key == 'enable_states':
            self.enable(value, missingok=True)
        elif key == 'disable_states':
            self.disable(value)
        else:
            raise ValueError, "StateCollection can accept only enable_states " \
                  "and disable_states arguments for the initialization. " \
                  "Got %s" % key

    # XXX RF: used only in meta -- those should become a bit tighter coupled
    #         and .copy / .update should only be used
    def _copy_states_(self, fromstate, key=None, deep=False):
        """Copy known here states from `fromstate` object into current object

        Parameters
        ----------
        fromstate : Collection or ClassWithCollections
          Source states to copy from
        key : None or list of str
          If not to copy all set state variables, key provides
          selection of what to copy
        deep : bool
          Optional control over the way to copy

        Crafted to overcome a problem mentioned above in the comment
        and is to be called from __copy__ of derived classes

        Probably sooner than later will get proper __getstate__,
        __setstate__
        """
        # Bad check... doesn't generalize well...
        # if not issubclass(fromstate.__class__, self.__class__):
        #     raise ValueError, \
        #           "Class  %s is not subclass of %s, " % \
        #           (fromstate.__class__, self.__class__) + \
        #           "thus not eligible for _copy_states_"
        # TODO: FOR NOW NO TEST! But this beast needs to be fixed...
        operation = { True: copy.deepcopy,
                      False: copy.copy }[deep]

        if isinstance(fromstate, ClassWithCollections):
            fromstate = fromstate.states

        if key is None:
            # copy all set ones
            for name in fromstate.which_set():#self.names:
                #if fromstate.has_key(name):
                self[name] = operation(fromstate[name])
        else:
            has_key = fromstate.has_key
            for name in key:
                if has_key(name):
                    self[name] = operation(fromstate[name])


    def is_enabled(self, key):
        """Returns `True` if state `key` is enabled"""
        return self[key].enabled


    def is_active(self, key):
        """Returns `True` if state `key` is known and is enabled"""
        return self.has_key(key) and self.is_enabled(key)


    def enable(self, key, value=True, missingok=False):
        """Enable  state variable given in `key`"""
        self._action(key, StateVariable._set_enabled, missingok=missingok,
                     value=value)


    def disable(self, key):
        """Disable state variable defined by `key` id"""
        self._action(key,
                     StateVariable._set_enabled, missingok=False, value=False)


    # TODO XXX think about some more generic way to grab temporary
    # snapshot of IndexedCollectables to be restored later on...
    def change_temporarily(self, enable_states=None,
                           disable_states=None, other=None):
        """Temporarily enable/disable needed states for computation

        Enable or disable states which are enabled in `other` and listed in
        `enable _states`. Use `reset_enabled_temporarily` to reset
        to previous state of enabled.

        `other` can be a ClassWithCollections object or StateCollection
        """
        if enable_states == None:
            enable_states = []
        if disable_states == None:
            disable_states = []
        self.__storedTemporarily.append(self.enabled)
        other_ = other
        if isinstance(other, ClassWithCollections):
            other = other.states

        if not other is None:
            # lets take states which are enabled in other but not in
            # self
            add_enable_states = list(Set(other.enabled).difference(
                 Set(enable_states)).intersection(self.names))
            if len(add_enable_states)>0:
                if __debug__:
                    debug("ST",
                          "Adding states %s from %s to be enabled temporarily" %
                          (add_enable_states, other_) +
                          " since they are not enabled in %s" %
                          (self))
                enable_states += add_enable_states

        # Lets go one by one enabling only disabled once... but could be as
        # simple as
        self.enable(enable_states)
        self.disable(disable_states)


    def reset_changed_temporarily(self):
        """Reset to previousely stored set of enabled states"""
        if __debug__:
            debug("ST", "Resetting to previous set of enabled states")
        if len(self.enabled)>0:
            self.enabled = self.__storedTemporarily.pop()
        else:
            raise ValueError("Trying to restore not-stored list of enabled " \
                             "states")


    # XXX probably nondefault logic could be done at places?
    #     =False is used in __repr__ and _svmbase
    # XXX also may be we need enabled to return a subcollection
    #        with binds to StateVariables found to be enabled?
    def _get_enabled(self, nondefault=True, invert=False):
        """Return list of enabled states

        Parameters
        ----------
        nondefault : bool
          Either to return also states which are enabled simply by default
        invert : bool
          Would invert the meaning, ie would return disabled states
        """
        if invert:
            fmatch = lambda y: not self.is_enabled(y)
        else:
            fmatch = self.is_enabled

        if nondefault:
            ffunc = fmatch
        else:
            ffunc = lambda y: fmatch(y) and \
                        self[y]._defaultenabled != self.is_enabled(y)
        return [n for n in self.names if ffunc(n)]


    def _set_enabled(self, keys):
        """Given `keys` make only those in the list enabled

        It might be handy to store set of enabled states and then to restore
        it later on. It can be easily accomplished now::

        >>> from mvpa.misc.state import ClassWithCollections, StateVariable
        >>> class Blah(ClassWithCollections):
        ...   bleh = StateVariable(enabled=False, doc='Example')
        ...
        >>> blah = Blah()
        >>> states_enabled = blah.states.enabled
        >>> blah.states.enabled = ['bleh']
        >>> blah.states.enabled = states_enabled
        """
        for key in self.keys():
            self.enable(key, key in keys)


    # Properties
    enabled = property(fget=_get_enabled, fset=_set_enabled)


##################################################################
# Base classes (and metaclass) which use collections
#


#
# Helper dictionaries for AttributesCollector
#
_known_collections = {
    # Quite a generic one but mostly in classifiers
    'StateVariable': ("states", StateCollection),
    # For classifiers only
    'Parameter': ("params", ParameterCollection),
    'KernelParameter': ("kernel_params", ParameterCollection),
#MH: no magic for datasets
#    # For datasets
#    # XXX custom collections needed?
#    'SampleAttribute':  ("sa", SampleAttributesCollection),
#    'FeatureAttribute': ("fa", SampleAttributesCollection),
#    'DatasetAttribute': ("dsa", SampleAttributesCollection),
    }


_col2class = dict(_known_collections.values())
"""Mapping from collection name into Collection class"""


#MH: no magic for datasets
#_COLLECTIONS_ORDER = ['sa', 'fa', 'dsa',
#                      'params', 'kernel_params', 'states']
_COLLECTIONS_ORDER = ['params', 'kernel_params', 'states']


class AttributesCollector(type):
    """Intended to collect and compose collections for any child
    class of ClassWithCollections
    """


    def __init__(cls, name, bases, dict):
        """
        Parameters
        ----------
        name : str
          Name of the class
        bases : iterable
          Base classes
        dict : dict
          Attributes.
        """
        if __debug__:
            debug(
                "COLR",
                "AttributesCollector call for %s.%s, where bases=%s, dict=%s " \
                % (cls, name, bases, dict))

        super(AttributesCollector, cls).__init__(name, bases, dict)

        collections = {}
        for name, value in dict.iteritems():
            if isinstance(value, IndexedCollectable):
                baseclassname = value.__class__.__name__
                col = _known_collections[baseclassname][0]
                # XXX should we allow to throw exceptions here?
                if not collections.has_key(col):
                    collections[col] = {}
                collections[col][name] = value
                # and assign name if not yet was set
                if value.name is None:
                    value._set_name(name)
                # !!! We do not keep copy of this attribute static in the class.
                #     Due to below traversal of base classes, we should be
                #     able to construct proper collections even in derived classes
                delattr(cls, name)

        # XXX can we first collect parent's states and then populate with ours?
        # TODO

        for base in bases:
            if hasattr(base, "__metaclass__") and \
                   base.__metaclass__ == AttributesCollector:
                # TODO take care about overriding one from super class
                # for state in base.states:
                #    if state[0] =
                newcollections = base._collections_template
                if len(newcollections) == 0:
                    continue
                if __debug__: # XXX RF:  and "COLR" in debug.active:
                    debug("COLR",
                          "Collect collections %s for %s from %s" %
                          (newcollections, cls, base))
                for col, collection in newcollections.iteritems():
                    if collections.has_key(col):
                        collections[col].update(collection)
                    else:
                        collections[col] = collection


        if __debug__:
            debug("COLR",
                  "Creating StateCollection template %s with collections %s"
                  % (cls, collections.keys()))

        # if there is an explicit
        if hasattr(cls, "_ATTRIBUTE_COLLECTIONS"):
            for col in cls._ATTRIBUTE_COLLECTIONS:
                if not col in _col2class:
                    raise ValueError, \
                          "Requested collection %s is unknown to collector" % \
                          col
                if not col in collections:
                    collections[col] = None

        # TODO: check on conflict in names of Collections' items!  since
        # otherwise even order is not definite since we use dict for
        # collections.
        # XXX should we switch to tuple?

        for col, colitems in collections.iteritems():
            # so far we collected the collection items in a dict, but the new
            # API requires to pass a _list_ of collectables instead of a dict.
            # So, whenever there are items, we pass just the values of the dict.
            # There is no information last, since the keys of the dict are the
            # name attributes of each collectable in the list.
            if not colitems is None:
                collections[col] = _col2class[col](items=colitems.values())
            else:
                collections[col] = _col2class[col]()

        setattr(cls, "_collections_template", collections)

        #
        # Expand documentation for the class based on the listed
        # parameters an if it is stateful
        #
        # TODO -- figure nice way on how to alter __init__ doc directly...
        textwrapper = TextWrapper(subsequent_indent="    ",
                                  initial_indent="    ",
                                  width=70)

        # Parameters
        paramsdoc = ""
        paramscols = []
        for col in ('params', 'kernel_params'):
            if collections.has_key(col):
                paramscols.append(col)
                # lets at least sort the parameters for consistent output
                col_items = collections[col]
                iparams = [(v._instance_index, k)
                           for k,v in col_items.iteritems()]
                iparams.sort()
                paramsdoc += '\n'.join(
                    [col_items[iparam[1]].doc(indent='  ')
                     for iparam in iparams]) + '\n'

        # Parameters collection could be taked hash of to decide if
        # any were changed? XXX may be not needed at all?
        setattr(cls, "_paramscols", paramscols)

        # States doc
        statesdoc = ""
        if collections.has_key('states'):
            paramsdoc += """  enable_states : None or list of str
    Names of the state variables which should be enabled additionally
    to default ones
  disable_states : None or list of str
    Names of the state variables which should be disabled
"""
            if len(collections['states']):
                statesdoc += '\n'.join(['  * ' + x
                                        for x in collections['states'].listing])
                statesdoc += "\n\n(States enabled by default suffixed with `+`)"
            if __debug__:
                debug("COLR", "Assigning __statesdoc to be %s" % statesdoc)
            setattr(cls, "_statesdoc", statesdoc)

        if paramsdoc != "":
            if __debug__ and 'COLR' in debug.active:
                debug("COLR", "Assigning __paramsdoc to be %s" % paramsdoc)
            setattr(cls, "_paramsdoc", paramsdoc)

        if paramsdoc + statesdoc != "":
            cls.__doc__ = enhancedDocString(cls, *bases)



class ClassWithCollections(object):
    """Base class for objects which contain any known collection

    Classes inherited from this class gain ability to access
    collections and their items as simple attributes. Access to
    collection items "internals" is done via <collection_name> attribute
    and interface of a corresponding `Collection`.
    """

    _DEV__doc__ = """
    TODO: rename 'descr'? -- it should simply
          be 'doc' -- no need to drag classes docstring imho.
    """

    __metaclass__ = AttributesCollector

    def __new__(cls, *args, **kwargs):
        """Instantiate ClassWithCollections object
        """
        self = super(ClassWithCollections, cls).__new__(cls)

        s__dict__ = self.__dict__

        # init variable
        # XXX: Added as pylint complained (rightfully) -- not sure if false
        # is the proper default
        self.__params_set = False

        # need to check to avoid override of enabled states in the case
        # of multiple inheritance, like both ClassWithCollectionsl and
        # Harvestable
        if not s__dict__.has_key('_collections'):
            s__class__ = self.__class__

            collections = copy.deepcopy(s__class__._collections_template)
            s__dict__['_collections'] = collections
            s__dict__['_known_attribs'] = {}
            """Dictionary to contain 'links' to the collections from each
            known attribute. Is used to gain some speed up in lookup within
            __getattribute__ and __setattr__
            """

            # Assign owner to all collections
            for col, collection in collections.iteritems():
                if col in s__dict__:
                    raise ValueError, \
                          "Object %s has already attribute %s" % \
                          (self, col)
                s__dict__[col] = collection
                collection.name = col
                collection.owner = self

            self.__params_set = False

        if __debug__:
            descr = kwargs.get('descr', None)
            debug("COL", "ClassWithCollections.__new__ was done "
                  "for %s#%s with descr=%s" \
                  % (s__class__.__name__, id(self), descr))

        return self


    def __init__(self, descr=None, **kwargs):
        """Initialize ClassWithCollections object

        Parameters
        ----------
        descr : str
          Description of the instance
        """
        # Note: __params_set was initialized in __new__
        if not self.__params_set:
            self.__descr = descr
            """Set humane description for the object"""

            # To avoid double initialization in case of multiple inheritance
            self.__params_set = True

            collections = self._collections
            # Assign attributes values if they are given among
            # **kwargs
            for arg, argument in kwargs.items():
                isset = False
                for collection in collections.itervalues():
                    if collection._is_initializable(arg):
                        collection._initialize(arg, argument)
                        isset = True
                        break
                if isset:
                    _ = kwargs.pop(arg)
                else:
                    known_params = reduce(
                       lambda x,y:x+y,
                       [x.keys() for x in collections.itervalues()], [])
                    raise TypeError, \
                          "Unexpected keyword argument %s=%s for %s." \
                           % (arg, argument, self) \
                          + " Valid parameters are %s" % known_params

            ## Initialize other base classes
            ##  commented out since it seems to be of no use for now
            #if init_classes is not None:
            #    # return back stateful arguments since they might be
            #    # processed by underlying classes
            #    kwargs.update(kwargs_stateful)
            #    for cls in init_classes:
            #        cls.__init__(self, **kwargs)
            #else:
            #    if len(kwargs)>0:
            #        known_params = reduce(lambda x, y: x + y, \
            #                            [x.keys() for x in collections],
            #                            [])
            #        raise TypeError, \
            #              "Unknown parameters %s for %s." % (kwargs.keys(),
            #                                                 self) \
            #              + " Valid parameters are %s" % known_params
        if __debug__:
            debug("COL", "ClassWithCollections.__init__ was done "
                  "for %s#%s with descr=%s" \
                  % (self.__class__.__name__, id(self), descr))


    #__doc__ = enhancedDocString('ClassWithCollections', locals())

    if __debug__ and _debug_references:
        def __debug_references_call(self, method, key):
            """Helper for debugging location of the call
            """
            s_dict = _object_getattribute(self, '__dict__')
            known_attribs = s_dict['_known_attribs']
            if key in known_attribs:
                clsstr = str(self.__class__)
                # Skip some False positives
                if 'mvpa.datasets' in clsstr and 'Dataset' in clsstr and \
                       (key in ['labels', 'chunks', 'samples', 'mapper']):
                    return
                colname = known_attribs[key]
                # figure out and report invocation location
                ftb = traceback.extract_stack(limit=4)[-3]
                shit = '\n%s:%d:[%s %s.%s]: %s\n' % \
                       (ftb[:2] + (method, colname, key) + (ftb[3],))
                if not (shit in _debug_shits):
                    _debug_shits.append(shit)
                    sys.stderr.write(shit)


        def __getattribute__(self, key):
            # return all private ones first since smth like __dict__ might be
            # queried by copy before instance is __init__ed
            if key[0] == '_':
                return _object_getattribute(self, key)

            s_dict = _object_getattribute(self, '__dict__')
            # check if it is a known collection
            collections = s_dict['_collections']
            if key in collections:
                return collections[key]

            # MH: No implicite outbreak of collection items into the namespace of
            #     the parent class
            ## check if it is a part of any collection
            #known_attribs = s_dict['_known_attribs']
            #if key in known_attribs:
            #    return collections[known_attribs[key]].getvalue(key)

            # Report the invocation location if applicable
            self.__debug_references_call('get', key)

            # just a generic return
            return _object_getattribute(self, key)


        def __setattr__(self, key, value):
            if key[0] == '_':
                return _object_setattr(self, key, value)

            if __debug__ and _debug_references:
                # Report the invocation location if applicable
                self.__debug_references_call('set', key)

            ## YOH: if we are to disable access at instance level -- do it in
            ##      set as well ;)
            ##
            ## # Check if a part of a collection, and set appropriately
            ## s_dict = _object_getattribute(self, '__dict__')
            ## known_attribs = s_dict['_known_attribs']
            ## if key in known_attribs:
            ##     collections = s_dict['_collections']
            ##     return collections[known_attribs[key]].setvalue(key, value)

            # Generic setattr
            return _object_setattr(self, key, value)


    # XXX not sure if we shouldn't implement anything else...
    def reset(self):
        for collection in self._collections.values():
            collection.reset()


    def __str__(self):
        s = "%s:" % (self.__class__.__name__)
        if self.__descr is not None:
            s += "/%s " % self.__descr
        if hasattr(self, "_collections"):
            for col, collection in self._collections.iteritems():
                s += " %d %s:%s" % (len(collection), col, str(collection))
        return s


    def __repr__(self, prefixes=None, fullname=False):
        """String definition of the object of ClassWithCollections object

        Parameters
        ----------
        fullname : bool
          Either to include full name of the module
        prefixes : list of str
          What other prefixes to prepend to list of arguments
        """
        if prefixes is None:
            prefixes = []
        prefixes = prefixes[:]          # copy list
        id_str = ""
        module_str = ""
        if __debug__:
            if 'MODULE_IN_REPR' in debug.active:
                fullname = True
            if 'ID_IN_REPR' in debug.active:
                id_str = '#%r' % id(self)

        if fullname:
            modulename = '%s' % self.__class__.__module__
            if modulename != "__main__":
                module_str = "%s." % modulename

        # Collections' attributes
        collections = self._collections
        # we want them in this particular order
        for col in _COLLECTIONS_ORDER:
            collection = collections.get(col, None)
            if collection is None:
                continue
            prefixes += collection._cls_repr()

        # Description if present
        descr = self.__descr
        if descr is not None:
            prefixes.append("descr=%r" % descr)

        return "%s%s(%s)%s" % (module_str, self.__class__.__name__,
                               ', '.join(prefixes), id_str)


    descr = property(lambda self: self.__descr,
                     doc="Description of the object if any")



class Harvestable(ClassWithCollections):
    """Classes inherited from this class intend to collect attributes
    within internal processing.

    Subclassing Harvestable we gain ability to collect any internal
    data from the processing which is especially important if an
    object performs something in loop and discards some intermidiate
    possibly interesting results (like in case of
    CrossValidatedTransferError and states of the trained classifier
    or TransferError).

    """

    harvested = StateVariable(enabled=False, doc=
       """Store specified attributes of classifiers at each split""")

    _KNOWN_COPY_METHODS = [ None, 'copy', 'deepcopy' ]


    def __init__(self, harvest_attribs=None, copy_attribs='copy', **kwargs):
        """Initialize state of harvestable

        Parameters
        harvest_attribs : list of (str or dict)
          What attributes of call to store and return within
          harvested state variable. If an item is a dictionary,
          following keys are used ['name', 'copy'].
        copy_attribs : None or str, optional
          Default copying. If None -- no copying, 'copy'
          - shallow copying, 'deepcopy' -- deepcopying.

        """
        ClassWithCollections.__init__(self, **kwargs)

        self.__attribs = harvest_attribs
        self.__copy_attribs = copy_attribs

        self._set_attribs(harvest_attribs)


    def _set_attribs(self, attribs):
        """Set attributes to harvest

        Each attribute in self.__attribs must have following fields
         - name : functional (or arbitrary if 'obj' or 'attr' is set)
                  description of the thing to harvest,
                  e.g. 'transerror.clf.training_time'
         - obj : name of the object to harvest from (if empty,
                 'self' is assumed),
                 e.g 'transerror'
         - attr : attribute of 'obj' to harvest,
                 e.g. 'clf.training_time'
         - copy : None, 'copy' or 'deepcopy' - way to copy attribute
        """
        if attribs:
            # force the state
            self.states.enable('harvested')
            self.__attribs = []
            for i, attrib in enumerate(attribs):
                if isinstance(attrib, dict):
                    if not 'name' in attrib:
                        raise ValueError, \
                              "Harvestable: attribute must be a string or " + \
                              "a dictionary with 'name'"
                else:
                    attrib = {'name': attrib}

                # assign default method to copy
                if not 'copy' in attrib:
                    attrib['copy'] = self.__copy_attribs

                # check copy method
                if not attrib['copy'] in self._KNOWN_COPY_METHODS:
                    raise ValueError, "Unknown method %s. Known are %s" % \
                          (attrib['copy'], self._KNOWN_COPY_METHODS)

                if not ('obj' in attrib or 'attr' in attrib):
                    # Process the item to harvest
                    # split into obj, attr. If obj is empty, then assume self
                    split = attrib['name'].split('.', 1)
                    if len(split)==1:
                        obj, attr = split[0], None
                    else:
                        obj, attr = split
                    attrib.update({'obj':obj, 'attr':attr})

                if attrib['obj'] == '':
                    attrib['obj'] = 'self'

                # TODO: may be enabling of the states??

                self.__attribs.append(attrib)     # place value back
        else:
            # just to make sure it is not None or 0
            self.__attribs = []


    def _harvest(self, vars):
        """The harvesting function: must obtain dictionary of variables
        from the caller.

        Parameters
        ----------
        vars : dict
          Dictionary of available data. Most often locals() could be
          passed as `vars`. Mention that desired to be harvested
          private attributes better be bound locally to some variable

        Returns
        -------
        nothing
        """

        if not self.states.is_enabled('harvested') or len(self.__attribs)==0:
            return

        if not self.states.is_set('harvested'):
            self.states.harvested = dict([(a['name'], [])
                                        for a in self.__attribs])

        for attrib in self.__attribs:
            attrv = vars[attrib['obj']]

            # access particular attribute if needed
            if not attrib['attr'] is None:
                attrv = eval('attrv.%s' % attrib['attr'])

            # copy the value if needed
            attrv = {'copy':copy.copy,
                     'deepcopy':copy.deepcopy,
                     None:lambda x:x}[attrib['copy']](attrv)

            self.states.harvested[attrib['name']].append(attrv)


    harvest_attribs = property(fget=lambda self:self.__attribs,
                               fset=_set_attribs)



