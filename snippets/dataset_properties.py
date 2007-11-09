#-*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import timeit, sys
import numpy


if not __debug__:
    try:
        import psyco
        psyco.profile()
        print "Psyco online compilation is on"
    except:
        if __name__ == '__main__':
            print "Psyco online compilation is not available"


class Plain(object):

    def __init__(self, data):
        self.data = data


class PlainDerived(Plain):

    def __init__(self, data):
        Plain.__init__(self, data)


class PropertyLambda(object):

    def __init__(self, data):
        self.__data = data

    data = property (fget=lambda x:x.__data)



class PropertyFunc(object):

    def __init__(self, data):
        self.__data = data

    def _getData(self):
        return self.__data

    data = property (fget=_getData)


class DictPublicPropertyLambda(object):

    def __init__(self, data):
        self.dict = {'data': data}

    data = property (fget=lambda x:x.dict['data'])


class DictPropertyLambda(object):

    def __init__(self, data):
        self.__dict = {'data': data}

    data = property (fget=lambda x:x.__dict['data'])


class DictPropertyFunc(object):

    def __init__(self, data):
        self.__dict = {'data': data}


    def _getData(self):
        return self.__dict['data']


    data = property (fget=_getData)


"""
Following classes create property in __init__.  Since __init__
modifies the actual class, we can simply check if there is already
such attribute defined to don't define property on each call which
brings the penalty (as tested) especially in case of 'exec'
"""

class CreatedPropertyLambda(object):
    """This one is pretty much the same as PropertyLambda
    """

    def __init__(self, data):
        self.__data = data

        if not self.__class__.__dict__.has_key("data"):
            self.__class__.data = property(fget=lambda x: x.__data)



class CreatedPropertyDict(object):

    def __init__(self, data):
        """This one is pretty much the same as PropertyLambda
        """
        self.__dict = {'data': data}
        if not self.__class__.__dict__.has_key("data"):
            self.__class__.data = property(fget=lambda x: x.__dict['data'])


class CreatedPropertyDictInput(object):

    def __init__(self, datadict):
        """This one is pretty much the same as PropertyLambda
        """
        self.__dict = datadict
        if not self.__class__.__dict__.has_key("data"):
            self.__class__.data = property(fget=lambda x: x.__dict['data'])


class CreatedPropertyDictInputExec(object):

    def __init__(self, datadict):
        """This one is pretty much the same as PropertyLambda
        """
        self.__dict = datadict
        if not self.__class__.__dict__.has_key("data"):
            exec "%s.%s = property(fget=lambda x: x._%s__dict['%s'])" %\
                 (self.__class__.__name__, "data", self.__class__.__name__, "data")


class CreatedPropertyDictInputExecFull(object):
    """
    This gets closer to the full implementation which also checks for existance
    of _get' and _set'ers
    """
    def __init__(self, datadict, copy=False):
        """This one is pretty much the same as PropertyLambda
        """
        # has to be not private since otherwise derived methods
        # would have problem accessing it... aren't we getting
        # Vproperty with this? :-)))
        self._dict = {}
        classdict = self.__class__.__dict__
        for key, value in datadict.iteritems():
            if copy:
                self._dict[key] = value.copy()
            else:
                self._dict[key] = value

            if not classdict.has_key(key):
                # define get function and use corresponding
                # _getATTR if such defined
                getter = '_get%s' % key
                if classdict.has_key(getter):
                    getter =  '%s.%s' % (self.__class__.__name__, getter)
                else:
                    getter="lambda x: x._dict['%s']" % (key)

                # define set function and use corresponding
                # _setATTR if such defined
                setter = '_set%s' % key
                if classdict.has_key(setter):
                    setter =  '%s.%s' % (self.__class__.__name__, setter)
                else:
                    setter = None

                exec "%s.%s = property(fget=%s,fset=%s)" %\
                     (self.__class__.__name__, key, getter, setter)



class CreatedPropertyDictInputExecFullDerivedWithGetter(CreatedPropertyDictInputExecFull):
    """This class has defined a getter for a property.
    Lets check if that one gets active
    """

    def __init__(self, datadict):
        CreatedPropertyDictInputExecFull.__init__(self, datadict)

    def _getdata(self):
        #print "Getting data" # it does work! sweet
        return self._dict["data"]

    def _setdata(self, x):
        #print "Setting data" # it does work! sweet
        self._dict["data"] = x


class CreatedPropertyDictInputExecFullDerived(CreatedPropertyDictInputExecFull):
    """
    Just to check if there is much penalty for *args, *kwargs
    in follow up class
    """

    def __init__(self, datadict):
        CreatedPropertyDictInputExecFull.__init__(self, datadict)


class CreatedPropertyDictInputExecFullDerivedWithArgs(CreatedPropertyDictInputExecFull):
    """
    Just to check if there is much penalty for *args, *kwargs
    """

    def __init__(self, *args, **kwargs):
        CreatedPropertyDictInputExecFull.__init__(self, *args, **kwargs)

class CreatedPropertyDictInputExecFullDerivedWithCopy(CreatedPropertyDictInputExecFull):

    def __init__(self, datadict):
        CreatedPropertyDictInputExecFull.__init__(self, datadict, copy=True)


l = numpy.random.normal(size=(1000))
l2 = numpy.random.normal(size=(1000))

if __name__ == "__main__":
    N = 10000
    K = 100

    def stat(t):
        """ Just to get better sense """
        return "Min:%f Mean:%f Max:%f StdDev%%:%f" % (min(t),  numpy.mean(t), max(t),
                                                      100.0*numpy.std(t)/numpy.mean(t))


    print "Working with list of length %d" % len(l)
    print "All numbers are # of seconds for %d runs, %d computations each" % (K, N)
    print "---------------------------------------------"
    types = ["Plain", "PlainDerived", "PropertyLambda",  "CreatedPropertyLambda",
             "PropertyFunc",
             "DictPublicPropertyLambda", "DictPropertyLambda",
             "DictPropertyFunc",
             "CreatedPropertyDict"]

    dictinputtypes = [ "CreatedPropertyDictInput",
                       "CreatedPropertyDictInputExec",
                       "CreatedPropertyDictInputExecFull",
                       "CreatedPropertyDictInputExecFullDerived",
                       "CreatedPropertyDictInputExecFullDerivedWithArgs",
                       "CreatedPropertyDictInputExecFullDerivedWithGetter",
                       "CreatedPropertyDictInputExecFullDerivedWithCopy",
                       ]

    def doit(types, l1, l2):
        for s in types:
            t = timeit.Timer(l1 % locals(), l2 % locals())
            results = t.repeat(K, N)
            print "%50s: " % s, stat(results)
            sys.stdout.flush()

    print "Constructors:"
    doit(types, "inst=%(s)s(l)", "from dataset_properties import %(s)s, l" )
    doit(dictinputtypes,
         "inst=%(s)s({'data':l})", "from dataset_properties import %(s)s, l")

    print "\nAccess to .data:"
    doit(types, "data = inst.data",
         "from dataset_properties import %(s)s, l; inst=%(s)s(l)")
    doit(dictinputtypes, "data = inst.data",
         "from dataset_properties import %(s)s, l; inst=%(s)s({'data':l})")

    print "\nSet .data:"
    doit(["CreatedPropertyDictInputExecFullDerivedWithGetter"], "inst.data = l2",
         "from dataset_properties import %(s)s, l, l2; inst=%(s)s({'data':l})")
