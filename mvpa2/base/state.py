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
? check/possible assure order of parameters/ca to be as listed in the
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

from mvpa2.base.types import is_sequence_type
import mvpa2.support.copy as copy
from textwrap import TextWrapper

# Although not used here -- included into interface
from mvpa2.misc.exceptions import UnknownStateError
from mvpa2.base.attributes import IndexedCollectable, ConditionalAttribute
from mvpa2.base.dochelpers import enhanced_doc_string, borrowdoc, _repr_attrs, \
     get_docstring_split, _strid, _saferepr

from mvpa2.base import externals
# XXX local rename is due but later on
from mvpa2.base.collections import Collection as BaseCollection

if __debug__:
    from mvpa2.base import debug
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

     XXX Seems to be not used and duplicating functionality: `listing`
     (thus `listing` property)
    """

    def __init__(self, items=None, name=None):
        """Initialize the Collection

        Parameters
        ----------
        items : dict of IndexedCollectable's
          items to initialize with
        name : str
          name of the collection (as seen in the owner, e.g. 'ca')
        """
        # first set all stuff to nothing and later on charge it
        # this is important, since some of the stuff below relies in the
        # defaults
        self.name = name
        super(Collection, self).__init__(items)


    def __reduce__(self):
        #bcr = BaseCollection.__reduce__(self)
        res = (self.__class__, (self.items(), self.name,))
        #if __debug__ and 'COL_RED' in debug.active:
        #    debug('COL_RED', 'Returning %s for %s' % (res, self))
        return res


    @borrowdoc(BaseCollection)
    def copy(self, *args, **kwargs):
        # Create a generic copy of the collection
        anew = super(Collection, self).copy(*args, **kwargs)
        anew.name = self.name
        return anew


    def __str__(self):
        num = len(self)
        if __debug__ and "ST" in debug.active:
            maxnumber = 1000            # I guess all
        else:
            maxnumber = 4
        if self.name is not None:
            res = self.name
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
        #     arguments (like (dis|en)able_ca) is to be performed
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

        # MIH: explicitly comment out the rest, as it is unreachable and
        #      remained around for a while
        # items_s = ""
        # sep = ""
        # for item in self:
        #     try:
        #         itemvalue = "%r" % (self[item].value,)
        #         if len(itemvalue)>50:
        #             itemvalue = itemvalue[:10] + '...' + itemvalue[-10:]
        #         items_s += "%s'%s':%s" % (sep, item, itemvalue)
        #         sep = ', '
        #     except:
        #         pass
        # if items_s != "":
        #     s += "items={%s}" % items_s
        # s += ")"
        # return s


    def is_set(self, key=None):
        """If item (or any in the present or listed) was set

        Parameters
        ----------
        key : None or str or list of str
          What items to check if they were set in the collection
        """
        if key is not None:
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


    def _action(self, key, func, missingok=False, **kwargs):
        """Run specific func either on a single item or on all of them

        Parameters
        ----------
        key : str
          Name of the conditional attribute
        func
          Function (not bound) to call given an item, and **kwargs
        missingok : bool
          If True - do not complain about wrong key
        """
        if isinstance(key, basestring):
            if key.lower() == 'all':
                for key_ in self:
                    self._action(key_, func, missingok=missingok, **kwargs)
            else:
                try:
                    func(self[key], **kwargs)
                except:
                    if missingok:
                        return
                    raise
        elif is_sequence_type(key):
            for item in key:
                self._action(item, func, missingok=missingok, **kwargs)
        else:
            raise ValueError, \
                  "Don't know how to handle  variable given by %s" % key


    def reset(self, key=None):
        """Reset the conditional attribute defined by `key`"""

        if key is not None:
            keys = [ key ]
        else:
            keys = self.keys()

        if len(self):
            for key in keys:
                # XXX Check if that works as desired
                self._action(key, self.values()[0].__class__.reset,
                             missingok=False)

    # XXX RF: not used anywhere / myself -- hence not worth it?
    @property
    def listing(self):
        """Return a list of registered ca along with the documentation"""

        # lets assure consistent litsting order
        items_ = self.items()
        items_.sort()
        return [ "%s%s%s: %s" % (_def_sep, str(x[1]), _def_sep, x[1].__doc__)
                 for x in items_ ]



class ParameterCollection(Collection):
    """Container of Parameters for a stateful object.
    """

#    def __init__(self, items=None, name=None):
#        """Initialize the conditional attributes of a derived class
#
#        Parameters
#        ----------
#        items : dict
#          dictionary of ca
#        """
#        Collection.__init__(self, items, name)
#

    def _cls_repr(self):
        """Part of __repr__ for the owner object
        """
        prefixes = []
        for k in self.keys():
            # list only params with not default values
            if self[k].is_default:
                continue
            prefixes.append("%s=%s" % (k, _saferepr(self[k].value)))
        return prefixes


    def reset_value(self, key, missingok=False):
        """Reset all parameters to default values"""
        from param import Parameter
        self._action(key, Parameter.reset_value, missingok=False)


class ConditionalAttributesCollection(Collection):
    """Container of ConditionalAttributes for a stateful object.

    :Groups:
     - `Public Access Functions`: `has_key`, `is_enabled`, `is_active`
     - `Access Implementors`: `listing`, `names`, `_get_enabled`
     - `Mutators`: `__init__`, `enable`, `disable`, `_set_enabled`
     - `R/O Properties`: `listing`, `names`, `items`
     - `R/W Properties`: `enabled`
    """

    def __init__(self, items=None, name=None):
        """Initialize the conditional attributes of a derived class

        Parameters
        ----------
        items : dict
          dictionary of ca
        name : str
          literal description. Usually just attribute name for the
          collection, e.g. 'ca'
        """
        Collection.__init__(self, items=items, name=name)

        self.__storedTemporarily = []
        """List to contain sets of enabled ca which were enabled
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
            ca = self._get_enabled(nondefault=False,
                                       invert=invert)
            if len(ca):
                prefixes.append("%s_ca=%s" % (name, str(ca)))
        return prefixes


    def _is_initializable(self, key):
        """Checks if key could be assigned within collection via
        setvalue
        """
        return key in ['enable_ca', 'disable_ca']


    def _initialize(self, key, value):
        if value is None:
            value = []
        if key == 'enable_ca':
            self.enable(value, missingok=True)
        elif key == 'disable_ca':
            self.disable(value)
        else:
            raise ValueError, "ConditionalAttributesCollection can accept only enable_ca " \
                  "and disable_ca arguments for the initialization. " \
                  "Got %s" % key

    # XXX RF: used only in meta -- those should become a bit tighter coupled
    #         and .copy / .update should only be used
    def _copy_ca_(self, fromstate, key=None, deep=False):
        """Copy known here ca from `fromstate` object into current object

        Parameters
        ----------
        fromstate : Collection or ClassWithCollections
          Source ca to copy from
        key : None or list of str
          If not to copy all set conditional attributes, key provides
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
        #           "thus not eligible for _copy_ca_"
        # TODO: FOR NOW NO TEST! But this beast needs to be fixed...
        operation = { True: copy.deepcopy,
                      False: copy.copy }[deep]

        if isinstance(fromstate, ClassWithCollections):
            fromstate = fromstate.ca

        if key is None:
            # copy all set ones
            for name in fromstate.which_set():#self.keys():
                #if fromstate.has_key(name):
                self[name].value = operation(fromstate[name].value)
        else:
            # workaround for supporting py2 and py3 dictionary interface
            try:
                has_key = fromstate.has_key
            except AttributeError:
                has_key = fromstate.__contains__
            for name in key:
                if has_key(name):
                    self[name].value = operation(fromstate[name].value)


    def is_enabled(self, key):
        """Returns `True` if state `key` is enabled"""
        return key in self and self[key].enabled


    def is_active(self, key):
        """Returns `True` if state `key` is known and is enabled"""
        return key in self and self.is_enabled(key)


    def enable(self, key, value=True, missingok=False):
        """Enable  conditional attribute given in `key`"""
        self._action(key, ConditionalAttribute._set_enabled, missingok=missingok,
                     value=value)


    def disable(self, key):
        """Disable conditional attribute defined by `key` id"""
        self._action(key,
                     ConditionalAttribute._set_enabled, missingok=False, value=False)


    # TODO XXX think about some more generic way to grab temporary
    # snapshot of IndexedCollectables to be restored later on...
    def change_temporarily(self, enable_ca=None,
                           disable_ca=None, other=None):
        """Temporarily enable/disable needed ca for computation

        Enable or disable ca which are enabled in `other` and listed in
        `enable _ca`. Use `reset_enabled_temporarily` to reset
        to previous state of enabled.

        `other` can be a ClassWithCollections object or ConditionalAttributesCollection
        """
        if enable_ca == None:
            enable_ca = []
        if disable_ca == None:
            disable_ca = []
        self.__storedTemporarily.append(self.enabled)
        other_ = other
        if isinstance(other, ClassWithCollections):
            other = other.ca

        if other is not None:
            # lets take ca which are enabled in other but not in
            # self
            add_enable_ca = list(set(other.enabled).difference(
                 set(enable_ca)).intersection(self.keys()))
            if len(add_enable_ca)>0:
                if __debug__:
                    debug("ST",
                          "Adding ca %s from %s to be enabled temporarily "
                          "since they are not enabled in %s",
                          (add_enable_ca, other_, self))
                enable_ca += add_enable_ca

        # Lets go one by one enabling only disabled once... but could be as
        # simple as
        self.enable(enable_ca)
        self.disable(disable_ca)


    def reset_changed_temporarily(self):
        """Reset to previousely stored set of enabled ca"""
        if __debug__:
            debug("ST", "Resetting to previous set of enabled ca")
        if len(self.enabled)>0:
            self.enabled = self.__storedTemporarily.pop()
        else:
            raise ValueError("Trying to restore not-stored list of enabled " \
                             "ca")


    # XXX probably nondefault logic could be done at places?
    #     =False is used in __repr__ and _svmbase
    # XXX also may be we need enabled to return a subcollection
    #        with binds to ConditionalAttributes found to be enabled?
    def _get_enabled(self, nondefault=True, invert=False):
        """Return list of enabled ca

        Parameters
        ----------
        nondefault : bool
          Either to return also ca which are enabled simply by default
        invert : bool
          Would invert the meaning, ie would return disabled ca
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
        return [n for n in self.keys() if ffunc(n)]


    def _set_enabled(self, keys):
        """Given `keys` make only those in the list enabled

        It might be handy to store set of enabled ca and then to restore
        it later on. It can be easily accomplished now::

        >>> from mvpa2.base.state import ClassWithCollections, ConditionalAttribute
        >>> class Blah(ClassWithCollections):
        ...   bleh = ConditionalAttribute(enabled=False, doc='Example')
        ...
        >>> blah = Blah()
        >>> ca_enabled = blah.ca.enabled
        >>> blah.ca.enabled = ['bleh']
        >>> blah.ca.enabled = ca_enabled
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
    'ConditionalAttribute': ("ca", ConditionalAttributesCollection),
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
#                      'params', 'kernel_params', 'ca']
_COLLECTIONS_ORDER = ['params', 'kernel_params', 'ca']


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
                "AttributesCollector call for %s.%s, where bases=%s, dict=%s ",
                (cls, name, bases, dict))

        super(AttributesCollector, cls).__init__(name, bases, dict)

        collections = {}
        for name, value in dict.iteritems():
            if isinstance(value, IndexedCollectable):
                baseclassname = value.__class__.__name__
                col = _known_collections[baseclassname][0]
                # XXX should we allow to throw exceptions here?
                if col not in collections:
                    collections[col] = {}
                collections[col][name] = value
                # and assign name if not yet was set
                if value.name is None:
                    value.name = name
                # !!! We do not keep copy of this attribute static in the class.
                #     Due to below traversal of base classes, we should be
                #     able to construct proper collections even in derived classes
                delattr(cls, name)

        # XXX can we first collect parent's ca and then populate with ours?
        # TODO

        for base in bases:
            if hasattr(base, "__class__") and \
                   base.__class__ == AttributesCollector:
                # TODO take care about overriding one from super class
                # for state in base.ca:
                #    if state[0] =
                newcollections = base._collections_template
                if len(newcollections) == 0:
                    continue
                if __debug__: # XXX RF:  and "COLR" in debug.active:
                    debug("COLR",
                          "Collect collections %s for %s from %s",
                          (newcollections, cls, base))
                for col, super_collection in newcollections.iteritems():
                    if col in collections:
                        if __debug__:
                            debug("COLR", "Updating existing collection %s with the one from super class" % col)
                        collection = collections[col]
                        # Current class could have overriden a parameter, so
                        # we need to keep it without updating
                        for pname, pval in super_collection.iteritems():
                            if pname not in collection:
                                collection[pname] = pval
                            elif __debug__:
                                debug("COLR", "Not overriding %s.%s of cls %s from base %s"
                                      % (col, pname, cls, base))
                    else:
                        collections[col] = super_collection


        if __debug__:
            debug("COLR",
                  "Creating ConditionalAttributesCollection template %s "
                  "with collections %s", (cls, collections.keys()))

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
            if colitems is not None:
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
        paramsdoc = []
        paramscols = []
        for col in ('params', 'kernel_params'):
            if col in collections:
                paramscols.append(col)
                # lets at least sort the parameters for consistent output
                col_items = collections[col]
                iparams = [(v._instance_index, k)
                           for k,v in col_items.iteritems()]
                iparams.sort()
                paramsdoc += [(col_items[iparam[1]].name,
                               col_items[iparam[1]]._paramdoc())
                              for iparam in iparams]

        # Parameters collection could be taked hash of to decide if
        # any were changed? XXX may be not needed at all?
        setattr(cls, "_paramscols", paramscols)

        # States doc
        cadoc = ""
        if 'ca' in collections:
            paramsdoc += [
                ('enable_ca',
                 "enable_ca : None or list of str\n  "
                 "Names of the conditional attributes which should "
                 "be enabled in addition\n  to the default ones"),
                ('disable_ca',
                 "disable_ca : None or list of str\n  "
                 "Names of the conditional attributes which should "
                 "be disabled""")]
            if len(collections['ca']):
                cadoc += '\n'.join(['* ' + x
                                    for x in collections['ca'].listing])
                cadoc += "\n\n(Conditional attributes enabled by default suffixed with `+`)"
            if __debug__:
                debug("COLR", "Assigning __cadoc to be %s", (cadoc,))
            setattr(cls, "_cadoc", cadoc)

        if paramsdoc != "":
            if __debug__ and 'COLR' in debug.active:
                debug("COLR", "Assigning __paramsdoc to be %s", (paramsdoc,))
            setattr(cls, "_paramsdoc", paramsdoc)

        if len(paramsdoc) or cadoc != "":
            cls.__doc__ = enhanced_doc_string(cls, *bases)



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

        # need to check to avoid override of enabled ca in the case
        # of multiple inheritance, like both ClassWithCollectionsl and
        # Harvestable (note: Harvestable was refactored away)
        if '_collections' not in s__dict__:
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

            self.__params_set = False

        if __debug__:
            descr = kwargs.get('descr', None)
            debug("COL", "ClassWithCollections.__new__ was done "
                  "for %s%s with descr=%s",
                  (s__class__.__name__, _strid(self), descr))

        return self

    @classmethod
    def _custom_kwargs_sort_items(cls, **kwargs):
        """Custom sorting of kwargs to fulfill premises of pymvpa

        Some time ago the world was square and we could assume
        that enable_ca always comes before disable_ca.  But we
        were misguided, thus now we need to provide a custom
        sorting routine to sort disable_ca after enable_ca
        """

        def _key(x):
            k = x[0]
            if k == 'enable_ca':
                return 'ZZZZZZ'
            return k

        return sorted(kwargs.items(), key=_key)

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

            for arg, argument in self._custom_kwargs_sort_items(**kwargs):
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
                        [x.keys()
                         for x in collections.itervalues()
                         if not isinstance(x, ConditionalAttributesCollection)
                         ], [])
                    _, kwargs_list, _ = get_docstring_split(self.__init__)
                    known_params = sorted(known_params
                                          + [x[0] for x in kwargs_list])
                    raise TypeError(
                        "Unexpected keyword argument %s=%s for %s."
                        % (arg, argument, self)
                        + "\n\tValid parameters are: %s" % ', '.join(known_params))

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
                  "for %s%s with descr=%s",
                  (self.__class__.__name__, _strid(self), descr))


    #__doc__ = enhanced_doc_string('ClassWithCollections', locals())

    if __debug__ and _debug_references:
        def __debug_references_call(self, method, key):
            """Helper for debugging location of the call
            """
            s_dict = _object_getattribute(self, '__dict__')
            known_attribs = s_dict['_known_attribs']
            if key in known_attribs:
                clsstr = str(self.__class__)
                # Skip some False positives
                if 'mvpa2.datasets' in clsstr and 'Dataset' in clsstr and \
                       (key in ['targets', 'chunks', 'samples', 'mapper']):
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
            if key == '':
                raise AttributeError, "Silly to request attribute ''"

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
            if key == '':
                raise AttributeError, "Silly to set attribute ''"

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
        prefixes : list of str
          What other prefixes to prepend to list of arguments
        fullname : bool
          Either to include full name of the module
        """
        prefixes = prefixes or []
        prefixes = prefixes[:]          # copy list
        # filter using __init__doc__exclude__
        for f in getattr(self, '__init__doc__exclude__', []):
            prefixes = [x for x in prefixes if not x.startswith(f+'=')]
        id_str = ""
        module_str = ""

        if __debug__:
            if 'MODULE_IN_REPR' in debug.active:
                fullname = True
            if 'ID_IN_REPR' in debug.active:
                id_str = _strid(self)

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
        prefixes += _repr_attrs(self, ['descr'])

        out = "%s%s(%s)%s" % (module_str, self.__class__.__name__,
                               ', '.join(prefixes), id_str)
        # To possibly debug mass repr/str-fication
        # print str(self), ' REPR: ', out
        return out

    descr = property(lambda self: self.__descr,
                     doc="Description of the object if any")
