# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Python interface to the SWIG-wrapped libsvm"""

__docformat__ = 'restructuredtext'


from math import exp, fabs
import re, copy

import numpy as N

from mvpa.clfs.libsvmc import _svmc as svmc
from mvpa.clfs.libsvmc._svmc import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, \
                                  NU_SVR, LINEAR, POLY, RBF, SIGMOID, \
                                  PRECOMPUTED

if __debug__:
    from mvpa.base import debug

def intArray(seq):
    size = len(seq)
    array = svmc.new_int(size)
    i = 0
    for item in seq:
        svmc.int_setitem(array, i, item)
        i = i + 1
    return array


def doubleArray(seq):
    size = len(seq)
    array = svmc.new_double(size)
    i = 0
    for item in seq:
        svmc.double_setitem(array, i, item)
        i = i + 1
    return array


def freeIntArray(x):
    if x != 'NULL' and x != None:
        svmc.delete_int(x)


def freeDoubleArray(x):
    if x != 'NULL' and x != None:
        svmc.delete_double(x)


def intArray2List(x, n):
    return map(svmc.int_getitem, [x]*n, range(n))


def doubleArray2List(x, n):
    return map(svmc.double_getitem, [x]*n, range(n))


class SVMParameter(object):
    """
    SVMParameter class safe to be deepcopied.
    """
    # default values
    default_parameters = {
    'svm_type' : C_SVC,
    'kernel_type' : RBF,
    'degree' : 3,
    'gamma' : 0,        # 1/k
    'coef0' : 0,
    'nu' : 0.5,
    'cache_size' : 100,
    'C' : 1,
    'eps' : 1e-3,
    'p' : 0.1,
    'shrinking' : 1,
    'nr_weight' : 0,
    'weight_label' : [],
    'weight' : [],
    'probability' : 0
    }

    class _SVMCParameter(object):
        """Internal class to to avoid memory leaks returning away svmc's params"""

        def __init__(self, params):
            self.param = svmc.new_svm_parameter()
            for attr, val in params.items():
                # adjust val if necessary
                if attr == 'weight_label':
                    #self.__weight_label_len = len(val)
                    val = intArray(val)
                    # no need?
                    #freeIntArray(self.weight_label)
                elif attr == 'weight':
                    #self.__weight_len = len(val)
                    val = doubleArray(val)
                    # no need?
                    # freeDoubleArray(self.weight)
                # set the parameter through corresponding call
                set_func = getattr(svmc, 'svm_parameter_%s_set' % (attr))
                set_func(self.param, val)

        def __del__(self):
            if __debug__:
                debug('CLF_', 'Destroying libsvm._SVMCParameter %s' % str(self))
            freeIntArray(svmc.svm_parameter_weight_label_get(self.param))
            freeDoubleArray(svmc.svm_parameter_weight_get(self.param))
            svmc.delete_svm_parameter(self.param)


    def __init__(self, **kw):
        self._orig_params = kw
        self.untrain()

    def untrain(self):
        self._params = {}
        self._params.update(self.default_parameters) # kinda copy.copy ;-)
        self._params.update(**self._orig_params)       # update with new values
        self.__svmc_params = None       # none is computed 
        self.__svmc_recompute = False   # thus none to recompute

    def __repr__(self):
        return self._params

    def __str__(self):
        return "SVMParameter: %s" % `self._params`

    def __copy__(self):
        out = SVMParameter()
        out._params = copy.copy(self._params)
        return out

    def __deepcopy__(self, memo):
        out = SVMParameter()
        out._params = copy.deepcopy(self._params)
        return out

    def _clear_svmc_params(self):
        if not self.__svmc_params is None:
            del self.__svmc_params
        self.__svmc_params = None

    @property
    def param(self):
        if self.__svmc_recompute:
            self._clear_svmc_params()
        if self.__svmc_params is None:
            self.__svmc_params = SVMParameter._SVMCParameter(self._params)
            self.__svmc_recompute = False
        return self.__svmc_params.param

    def __del__(self):
        if __debug__:
            debug('CLF_', 'Destroying libsvm.SVMParameter %s' % str(self))
        self._clear_svmc_params()

    def _setParameter(self, key, value):
        """Not exactly proper one -- if lists are svmc_recompute, would fail anyways"""
        self.__svmc_recompute = True
        self._params[key] = value

    @classmethod
    def _register_properties(cls):
        for key in cls.default_parameters.keys():
            exec "%s.%s = property(fget=%s, fset=%s)"  % \
                 (cls.__name__, key,
                  "lambda self:self._params['%s']" % key,
                  "lambda self,val:self._setParameter('%s', val)" % key)


SVMParameter._register_properties()

def convert2SVMNode(x):
    """convert a sequence or mapping to an SVMNode array"""
    import operator

    if type(x) == dict:
        iter_range = list(x).sort()
    elif operator.isSequenceType(x):
        iter_range = range(len(x))
    else:
        raise TypeError, "data must be a mapping or a sequence"

    data = svmc.svm_node_array(len(iter_range)+1)
    svmc.svm_node_array_set(data, len(iter_range), -1, 0)

    [svmc.svm_node_array_set(data, j, k, x[k]) for j, k in enumerate(iter_range)]
    return data



class SVMProblem:
    def __init__(self, y, x):
        assert len(y) == len(x)
        self.prob = prob = svmc.new_svm_problem()
        self.size = size = len(y)

        self.y_array = y_array = svmc.new_double(size)
        for i in range(size):
            svmc.double_setitem(y_array, i, y[i])

        self.x_matrix = x_matrix = svmc.svm_node_matrix(size)
        self.data = []
        self.maxlen = 0
        for i in range(size):
            data = convert2SVMNode(x[i])
            self.data.append(data)
            svmc.svm_node_matrix_set(x_matrix, i, data)
            if type(x[i]) == dict:
                if (len(x[i]) > 0):
                    self.maxlen = max(self.maxlen, max(x[i].keys()))
            else:
                self.maxlen = max(self.maxlen, len(x[i]))

        svmc.svm_problem_l_set(prob, size)
        svmc.svm_problem_y_set(prob, y_array)
        svmc.svm_problem_x_set(prob, x_matrix)


    def __repr__(self):
        return "<SVMProblem: size = %s>" % (self.size)


    def __del__(self):
        if __debug__:
            debug('CLF_', 'Destroying libsvm.SVMProblem %s' % `self`)

        svmc.delete_svm_problem(self.prob)
        svmc.delete_double(self.y_array)
        for i in range(self.size):
            svmc.svm_node_array_destroy(self.data[i])
        svmc.svm_node_matrix_destroy(self.x_matrix)



class SVMModel:
    def __init__(self, arg1, arg2=None):
        if arg2 == None:
            # create model from file
            filename = arg1
            self.model = svmc.svm_load_model(filename)
        else:
            # create model from problem and parameter
            prob, param = arg1, arg2
            self.prob = prob
            if param.gamma == 0:
                param.gamma = 1.0/prob.maxlen
            msg = svmc.svm_check_parameter(prob.prob, param.param)
            if msg:
                raise ValueError, msg
            self.model = svmc.svm_train(prob.prob, param.param)

        #setup some classwide variables
        self.nr_class = svmc.svm_get_nr_class(self.model)
        self.svm_type = svmc.svm_get_svm_type(self.model)
        #create labels(classes)
        intarr = svmc.new_int(self.nr_class)
        svmc.svm_get_labels(self.model, intarr)
        self.labels = intArray2List(intarr, self.nr_class)
        svmc.delete_int(intarr)
        #check if valid probability model
        self.probability = svmc.svm_check_probability_model(self.model)


    def __repr__(self):
        """
        Print string representation of the model or easier comprehension
        and some statistics
        """
        ret = '<SVMModel:'
        try:
            ret += ' type = %s, ' % `self.svm_type`
            ret += ' number of classes = %d (%s), ' \
                   % ( self.nr_class, `self.labels` )
        except:
            pass
        return ret+'>'


    def predict(self, x):
        data = convert2SVMNode(x)
        ret = svmc.svm_predict(self.model, data)
        svmc.svm_node_array_destroy(data)
        return ret


    def getNRClass(self):
        return self.nr_class


    def getLabels(self):
        if self.svm_type == NU_SVR \
           or self.svm_type == EPSILON_SVR \
           or self.svm_type == ONE_CLASS:
            raise TypeError, "Unable to get label from a SVR/ONE_CLASS model"
        return self.labels


    #def getParam(self):
    #    return SVMParameter(
    #                svmc_parameter=svmc.svm_model_param_get(self.model))


    def predictValuesRaw(self, x):
        #convert x into SVMNode, allocate a double array for return
        n = self.nr_class*(self.nr_class-1)//2
        data = convert2SVMNode(x)
        dblarr = svmc.new_double(n)
        svmc.svm_predict_values(self.model, data, dblarr)
        ret = doubleArray2List(dblarr, n)
        svmc.delete_double(dblarr)
        svmc.svm_node_array_destroy(data)
        return ret


    def predictValues(self, x):
        v = self.predictValuesRaw(x)
        if self.svm_type == NU_SVR \
           or self.svm_type == EPSILON_SVR \
           or self.svm_type == ONE_CLASS:
            return v[0]
        else: #self.svm_type == C_SVC or self.svm_type == NU_SVC
            count = 0
            d = {}
            for i in range(len(self.labels)):
                for j in range(i+1, len(self.labels)):
                    d[self.labels[i], self.labels[j]] = v[count]
                    d[self.labels[j], self.labels[i]] = -v[count]
                    count += 1
            return  d


    def predictProbability(self, x):
        #c code will do nothing on wrong type, so we have to check ourself
        if self.svm_type == NU_SVR or self.svm_type == EPSILON_SVR:
            raise TypeError, "call get_svr_probability or get_svr_pdf " \
                             "for probability output of regression"
        elif self.svm_type == ONE_CLASS:
            raise TypeError, "probability not supported yet for one-class " \
                             "problem"
        #only C_SVC, NU_SVC goes in
        if not self.probability:
            raise TypeError, "model does not support probabiliy estimates"

        #convert x into SVMNode, alloc a double array to receive probabilities
        data = convert2SVMNode(x)
        dblarr = svmc.new_double(self.nr_class)
        pred = svmc.svm_predict_probability(self.model, data, dblarr)
        pv = doubleArray2List(dblarr, self.nr_class)
        svmc.delete_double(dblarr)
        svmc.svm_node_array_destroy(data)
        p = {}
        for i in range(len(self.labels)):
            p[self.labels[i]] = pv[i]
        return pred, p


    def getSVRProbability(self):
        #leave the Error checking to svm.cpp code
        ret = svmc.svm_get_svr_probability(self.model)
        if ret == 0:
            raise TypeError, "not a regression model or probability " \
                             "information not available"
        return ret


    def getSVRPdf(self):
        #get_svr_probability will handle error checking
        sigma = self.getSVRProbability()
        return lambda z: exp(-fabs(z)/sigma)/(2*sigma)


    def save(self, filename):
        svmc.svm_save_model(filename, self.model)


    def __del__(self):
        if __debug__:
            debug('CLF_', 'Destroying libsvm.SVMModel %s' % (`self`))

        try:
            svmc.svm_destroy_model(self.model)
        except:
            # blind way to overcome problem of already deleted model and
            # "SVMModel instance has no attribute 'model'" in  ignored
            pass


    def getTotalNSV(self):
        return svmc.svm_model_l_get(self.model)


    def getNSV(self):
        """Returns a list with the number of support vectors per class.
        """
        return [ svmc.int_getitem(svmc.svm_model_nSV_get( self.model ), i) 
                    for i in range( self.nr_class ) ]


    def getSV(self):
        """Returns an array with the all support vectors.

        array( nSV x <nFeatures>)
        """
        return svmc.svm_node_matrix2numpy_array(
                    svmc.svm_model_SV_get(self.model),
                    self.getTotalNSV(),
                    self.prob.maxlen)


    def getSVCoef(self):
        """Return coefficients for SVs... Needs to be used directly with caution!

        Summary on what is happening in libsvm internals with sv_coef

        svm_model's sv_coef (especially) are "cleverly" packed into a matrix
        nr_class - 1 x #SVs_total which stores
        coefficients for
        nr_class x (nr_class-1) / 2
        binary classifiers' SV coefficients.

        For classifier i-vs-j
        General packing rule can be described as:

          i-th row contains sv_coefficients for SVs of class i it took
          in all i-vs-j or j-vs-i classifiers.

        Another useful excerpt from svm.cpp is

                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

        It can also be described as j-th column lists coefficients for SV # j which
        belongs to some class C, which it took (if it was an SV, ie != 0)
        in classifiers i vs C (iff i<C), or C vs i+1 (iff i>C)

        This way no byte of storage is wasted but imho such setup is quite convolved
        """
        return svmc.doubleppcarray2numpy_array(
                    svmc.svm_model_sv_coef_get(self.model),
                    self.nr_class - 1,
                    self.getTotalNSV())


    def getRho(self):
        """Return constant(s) in decision function(s) (if multi-class)"""
        return doubleArray2List(svmc.svm_model_rho_get(self.model),
                                self.nr_class * (self.nr_class-1)/2)
