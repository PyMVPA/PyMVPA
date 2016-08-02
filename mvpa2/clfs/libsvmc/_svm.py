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

import numpy as np

from mvpa2.base.types import is_sequence_type

from mvpa2.clfs.libsvmc import svmc
from mvpa2.clfs.libsvmc.svmc import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, \
                                  NU_SVR, LINEAR, POLY, RBF, SIGMOID, \
                                  PRECOMPUTED

if __debug__:
    from mvpa2.base import debug

##REF: Name was automagically refactored
def int_array(seq):
    size = len(seq)
    array = svmc.new_int(size)
    for i, item in enumerate(seq):
        svmc.int_setitem(array, i, int(item))
    return array


##REF: Name was automagically refactored
def double_array(seq):
    size = len(seq)
    array = svmc.new_double(size)
    for i, item in enumerate(seq):
        svmc.double_setitem(array, i, item)
    return array


##REF: Name was automagically refactored
def free_int_array(x):
    if x != 'NULL' and x != None:
        svmc.delete_int(x)


##REF: Name was automagically refactored
def free_double_array(x):
    if x != 'NULL' and x != None:
        svmc.delete_double(x)


def int_array_to_list(x, n):
    return [svmc.int_getitem(x, i) for i in xrange(n)]


def double_array_to_list(x, n):
    return [svmc.double_getitem(x, i) for i in xrange(n)]


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
            self.param = svmc.svm_parameter()
            for attr, val in params.items():
                # adjust val if necessary
                if attr == 'weight_label':
                    #self.__weight_label_len = len(val)
                    val = int_array(val)
                    # no need?
                    #free_int_array(self.weight_label)
                elif attr == 'weight':
                    #self.__weight_len = len(val)
                    val = double_array(val)
                    # no need?
                    # free_double_array(self.weight)
                # set the parameter through corresponding call
                setattr(self.param, attr, val)

        def __del__(self):
            if __debug__:
                debug('SVM_', 'Destroying libsvm._SVMCParameter %s' % str(self))
            free_int_array(self.param.weight_label)
            free_double_array(self.param.weight)
            del self.param


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
        if self.__svmc_params is not None:
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
            debug('SVM_', 'Destroying libsvm.SVMParameter %s' % str(self))
        self._clear_svmc_params()

    ##REF: Name was automagically refactored
    def _set_parameter(self, key, value):
        """Not exactly proper one -- if lists are svmc_recompute, would fail anyways"""
        self.__svmc_recompute = True
        self._params[key] = value

    @classmethod
    def _register_properties(cls):
        for key in cls.default_parameters.keys():
            exec "%s.%s = property(fget=%s, fset=%s)"  % \
                 (cls.__name__, key,
                  "lambda self:self._params['%s']" % key,
                  "lambda self,val:self._set_parameter('%s', val)" % key)


SVMParameter._register_properties()

##REF: Name was automagically refactored
def seq_to_svm_node(x):
    """convert a sequence or mapping to an SVMNode array"""

    length = len(x)

    # make two lists, one of indices, one of values
    # YYY Use isinstance  instead of type...is so we could
    #     easily use derived subclasses
    if isinstance(x, np.ndarray):
        iter_range = range(length)
        iter_values = x
    elif isinstance(x, dict):
        iter_range = list(x).sort()
        iter_values = np.ndarray(x.values())
    elif is_sequence_type(x):
        iter_range = range(length)
        iter_values = np.asarray(x)
    else:
        raise TypeError, "data must be a mapping or an ndarray or a sequence"

    # allocate c struct
    data = svmc.svm_node_array(length + 1)
    # insert markers into the c struct
    svmc.svm_node_array_set(data, length, -1, 0.0)
    # pass the list and the ndarray to the c struct
    svmc.svm_node_array_set(data, iter_range, iter_values)

    return data



class SVMProblem:
    def __init__(self, y, x):
        assert len(y) == len(x)
        self.prob = prob = svmc.svm_problem()
        self.size = size = len(y)

        self.y_array = y_array = svmc.new_double(size)
        for i in xrange(size):
            svmc.double_setitem(y_array, i, y[i])

        self.x_matrix = x_matrix = svmc.svm_node_matrix(size)
        data = [None for i in xrange(size)]
        maxlen = 0
        for i in xrange(size):
            x_i = x[i]
            lx_i = len(x_i)
            data[i] = d = seq_to_svm_node(x_i)
            svmc.svm_node_matrix_set(x_matrix, i, d)
            if isinstance(x_i, dict):
                if (lx_i > 0):
                    maxlen = max(maxlen, max(x_i.keys()))
            else:
                maxlen = max(maxlen, lx_i)

        # bind to instance
        self.data = data
        self.maxlen = maxlen
        prob.l = size
        prob.y = y_array
        prob.x = x_matrix


    def __repr__(self):
        return "<SVMProblem: size = %s>" % (self.size)


    def __del__(self):
        if __debug__ and debug is not None:
            debug('SVM_', 'Destroying libsvm.SVMProblem %s' % `self`)

        del self.prob
        if svmc is not None:
            svmc.delete_double(self.y_array)
        for i in range(self.size):
            svmc.svm_node_array_destroy(self.data[i])
        svmc.svm_node_matrix_destroy(self.x_matrix)
        del self.data
        del self.x_matrix



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
        self.labels = int_array_to_list(intarr, self.nr_class)
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
        data = seq_to_svm_node(x)
        ret = svmc.svm_predict(self.model, data)
        svmc.svm_node_array_destroy(data)
        return ret


    ##REF: Name was automagically refactored
    def get_nr_class(self):
        return self.nr_class


    ##REF: Name was automagically refactored
    def get_labels(self):
        if self.svm_type == NU_SVR \
           or self.svm_type == EPSILON_SVR \
           or self.svm_type == ONE_CLASS:
            raise TypeError, "Unable to get label from a SVR/ONE_CLASS model"
        return self.labels


    #def getParam(self):
    #    return SVMParameter(
    #                svmc_parameter=svmc.svm_model_param_get(self.model))


    ##REF: Name was automagically refactored
    def predict_values_raw(self, x):
        #convert x into SVMNode, allocate a double array for return
        n = self.nr_class*(self.nr_class-1)//2
        data = seq_to_svm_node(x)
        dblarr = svmc.new_double(n)
        svmc.svm_predict_values(self.model, data, dblarr)
        ret = double_array_to_list(dblarr, n)
        svmc.delete_double(dblarr)
        svmc.svm_node_array_destroy(data)
        return ret


    ##REF: Name was automagically refactored
    def predict_values(self, x):
        v = self.predict_values_raw(x)
        if self.svm_type == NU_SVR \
           or self.svm_type == EPSILON_SVR \
           or self.svm_type == ONE_CLASS:
            return v[0]
        else: #self.svm_type == C_SVC or self.svm_type == NU_SVC
            count = 0
            d = {}
            for i, li in enumerate(self.labels):
                for lj in self.labels[i+1:]:
                    d[li, lj] = v[count]
                    d[lj, li] = -v[count]
                    count += 1
            return  d


    ##REF: Name was automagically refactored
    def predict_probability(self, x):
        #c code will do nothing on wrong type, so we have to check ourself
        if self.svm_type == NU_SVR or self.svm_type == EPSILON_SVR:
            raise TypeError, "call get_svr_probability or get_svr_pdf " \
                             "for probability output of regression"
        elif self.svm_type == ONE_CLASS:
            raise TypeError, "probability not supported yet for one-class " \
                             "problem"
        #only C_SVC, NU_SVC goes in
        if not self.probability:
            raise TypeError, "model does not support probability estimates"

        #convert x into SVMNode, alloc a double array to receive probabilities
        data = seq_to_svm_node(x)
        dblarr = svmc.new_double(self.nr_class)
        pred = svmc.svm_predict_probability(self.model, data, dblarr)
        pv = double_array_to_list(dblarr, self.nr_class)
        svmc.delete_double(dblarr)
        svmc.svm_node_array_destroy(data)
        p = {}
        for i, l in enumerate(self.labels):
            p[l] = pv[i]
        return pred, p


    ##REF: Name was automagically refactored
    def get_svr_probability(self):
        #leave the Error checking to svm.cpp code
        ret = svmc.svm_get_svr_probability(self.model)
        if ret == 0:
            raise TypeError, "not a regression model or probability " \
                             "information not available"
        return ret


    ##REF: Name was automagically refactored
    def get_svr_pdf(self):
        #get_svr_probability will handle error checking
        sigma = self.get_svr_probability()
        return lambda z: exp(-fabs(z)/sigma)/(2*sigma)


    def save(self, filename):
        svmc.svm_save_model(filename, self.model)


    def __del__(self):
        if __debug__ and debug:
            # TODO: place libsvm versioning information into externals
            debug('SVM_', 'Destroying libsvm v. %s SVMModel %s',
                  (svmc is not None \
                   and hasattr(svmc, '__version__') \
                   and svmc.__version__ or "unknown",
                   `self`))
        try:
            svmc.svm_destroy_model_helper(self.model)
            del self.model
        except Exception, e:
            # blind way to overcome problem of already deleted model and
            # "SVMModel instance has no attribute 'model'" in  ignored
            if __debug__:
                debug('SVM_', 'Failed to destroy libsvm.SVMModel due to %s' % (e,))
            pass


    ##REF: Name was automagically refactored
    def get_total_n_sv(self):
        return self.model.l


    ##REF: Name was automagically refactored
    def get_n_sv(self):
        """Returns a list with the number of support vectors per class.
        """
        return [ svmc.int_getitem(self.model.nSV, i)
                 for i in range( self.nr_class ) ]


    ##REF: Name was automagically refactored
    def get_sv(self):
        """Returns an array with the all support vectors.

        array( nSV x <nFeatures>)
        """
        return svmc.svm_node_matrix2numpy_array(
                    self.model.SV,
                    self.get_total_n_sv(),
                    self.prob.maxlen)


    ##REF: Name was automagically refactored
    def get_sv_coef(self):
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
                    self.model.sv_coef,
                    self.nr_class - 1,
                    self.get_total_n_sv())


    ##REF: Name was automagically refactored
    def get_rho(self):
        """Return constant(s) in decision function(s) (if multi-class)"""
        return double_array_to_list(self.model.rho, self.nr_class * (self.nr_class-1)//2)
