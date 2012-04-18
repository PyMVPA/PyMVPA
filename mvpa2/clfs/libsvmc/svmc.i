//-*-c++-*-
/*emacs: -*- mode: c++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: t -*-
  ex: set sts=4 ts=4 sw=4 noet: */

%module svmc
%{
#include "svm.h"
#include <Python.h>
#include <numpy/arrayobject.h>

// Compatibility layer for Python3
#if PY_MAJOR_VERSION >= 3
# define PyInt_AS_LONG PyLong_AS_LONG
#endif

// Just to assure its availability even on older versions of libsvm
// where no such preprocessor variable was available (when was it? ;))
#ifndef LIBSVM_VERSION
# define LIBSVM_VERSION 0
#endif

enum { __version__ = LIBSVM_VERSION };

#if LIBSVM_VERSION < 300
struct svm_model
{
	svm_parameter param;// parameter
	int nr_class;		// number of classes, = 2 in regression/one class svm
	int l;				// total #SV
	svm_node **SV;		// SVs (SV[l])
	double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[k-1][l])
	double *rho;		// constants in decision functions (rho[k*(k-1)/2])
	double *probA;		// pariwise probability information
	double *probB;

	// for classification only

	int *label;		// label of each class (label[k])
	int *nSV;		// number of SVs for each class (nSV[k])
					// nSV[0] + nSV[1] + ... + nSV[k-1] = l
	// XXX
	int free_sv;	// 1 if svm_model is created by svm_load_model
					// 0 if svm_model is created by svm_train
};
#endif

/* convert node matrix into a numpy array */
static PyObject*
svm_node_matrix2numpy_array(struct svm_node** matrix, int rows, int cols)
{
	npy_intp dims[2] = {rows,cols};

	PyObject* array = 0;
	array = PyArray_SimpleNew ( 2, dims, NPY_DOUBLE );

	/* array subscription is [row][column] */
	PyArrayObject* a = (PyArrayObject*) array;

	double* data = (double *)a->data;

	int i,j;

	for (i = 0; i<rows; ++i)
	{
		for (j = 0; j<cols; ++j)
		{
			data[cols*i+j] = (matrix[i][j]).value;
		}
	}

	return PyArray_Return ( (PyArrayObject*) array	);
}

static PyObject* doubleppcarray2numpy_array(double** carray, int rows, int cols)
{
	if (!carray)
	{
		PyErr_SetString(PyExc_RuntimeError, "Zero pointer passed instead of valid double**.");
		return(NULL);
	}

	npy_intp dims[2] = {rows,cols};

	PyObject* array = 0;
	array = PyArray_SimpleNew ( 2, dims, NPY_DOUBLE );

	/* array subscription is [row][column] */
	PyArrayObject* a = (PyArrayObject*) array;

	double* data = (double *)a->data;

	int i,j;
	for (i = 0; i<rows; ++i)
	{
		for (j = 0; j<cols; ++j)
		{
			data[cols*i+j] = carray[i][j];
		}
	}

	return PyArray_Return ( (PyArrayObject*) array	);
}

/* rely on built-in facility to control verbose output
 * in the versions of libsvm >= 2.89
 */
#if LIBSVM_VERSION >= 289

/* borrowed from original libsvm code */
static void print_null(const char *s) {}

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

/* provide convenience wrapper */
void svm_set_verbosity(int verbosity_flag){
	if (verbosity_flag)
# if LIBSVM_VERSION < 291
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = &print_null;
# else
		svm_set_print_string_function(&print_string_stdout);
	else
		svm_set_print_string_function(&print_null);
# endif
}
#endif

%}

%init
%{
	import_array();
%}

enum { __version__ = LIBSVM_VERSION };

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR }; /* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };	/* kernel_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree; 	// for poly
	double gamma;	// for poly/rbf/sigmoid
	double coef0;	// for poly/sigmoid

	// these are for training only
	double cache_size;	// in MB
	double eps; 		// stopping criteria
	double C;			// for C_SVC, EPSILON_SVR and NU_SVR
	int nr_weight;		// for C_SVC
	int *weight_label;	// for C_SVC
	double* weight;		// for C_SVC
	double nu;			// for NU_SVC, ONE_CLASS, and NU_SVR
	double p;			// for EPSILON_SVR
	int shrinking;		// use the shrinking heuristics
	int probability;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

//
// svm_model
//
struct svm_model
{
	svm_parameter param;	// parameter
	int nr_class;			// number of classes, = 2 in regression/one class svm
	int l;					// total #SV
	svm_node **SV;			// SVs (SV[l])
	double **sv_coef;		// coefficients for SVs in decision functions (sv_coef[k-1][l])
	double *rho;			// constants in decision functions (rho[k*(k-1)/2])
	double *probA;			// pariwise probability information
	double *probB;

	// for classification only

	int *label;		// label of each class (label[k])
	int *nSV;		// number of SVs for each class (nSV[k])
					// nSV[0] + nSV[1] + ... + nSV[k-1] = l
	// XXX
	int free_sv;	// 1 if svm_model is created by svm_load_model
					// 0 if svm_model is created by svm_train
};

/* one really wants to configure verbosity within python! */
void svm_set_verbosity(int verbosity_flag);

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);

void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

void svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* decvalue);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

%inline %{
/* Just for bloody compatibility with deprecated method, which would
 not be used any ways, but declaring it for newer versions should allow to
 build across different versions without patching */
#if LIBSVM_VERSION >= 310
void svm_destroy_model(struct svm_model *model);
#endif
%}
/* Not necessary: the weight vector is (de)allocated at python-part
   void svm_destroy_param(struct svm_parameter *param); */

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

static PyObject* svm_node_matrix2numpy_array(struct svm_node** matrix, int rows, int cols);
static PyObject* doubleppcarray2numpy_array(double** data, int rows, int cols);

%include carrays.i
%array_functions(int,int)
%array_functions(double,double)

%inline %{

struct svm_node *svm_node_array(int size)
{
	return (struct svm_node *)malloc(sizeof(struct svm_node)*size);
}

void svm_node_array_set(struct svm_node *array, int i, int index, double value)
{
	array[i].index = index;
	array[i].value = value;
}

void svm_node_array_set(struct svm_node *array, PyObject *indices, PyObject *values)
{
	int length = PyList_Size(indices);
	int i;
	for (i = 0; i< length; i++){
		array[i].index = (int)PyInt_AS_LONG(PyList_GetItem(indices, i));
		PyObject* obj = PyArray_GETITEM(values, PyArray_GETPTR1(values, i));
		array[i].value = (double)PyFloat_AS_DOUBLE(obj);
		Py_DECREF(obj);
	}
}

void svm_node_array_destroy(struct svm_node *array)
{
	free(array);
}

struct svm_node **svm_node_matrix(int size)
{
	return (struct svm_node **)malloc(sizeof(struct svm_node *)*size);
}

void svm_node_matrix_set(struct svm_node **matrix, int i, struct svm_node* array)
{
	matrix[i] = array;
}

void svm_node_matrix_destroy(struct svm_node **matrix)
{
	free(matrix);
}

void svm_destroy_model_helper(svm_model *model_ptr)
{
#if LIBSVM_VERSION >= 300
    // yoh: Silence the blurber
	// fprintf(stderr, "warning: svm_destroy_model is deprecated and should not be used. Please use svm_free_and_destroy_model(svm_model **model_ptr_ptr)\n");
	svm_free_and_destroy_model(&model_ptr);
#else
	svm_destroy_model(model_ptr);
#endif
}

%}
