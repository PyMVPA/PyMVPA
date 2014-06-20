/*emacs: -*- mode: c-mode; tab-width: 8; c-basic-offset: 2; indent-tabs-mode: t -*-
  ex: set sts=4 ts=8 sw=4 noet: */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <Python.h>

/* Following code is for compatibility with Python3
   Example taken from: http://docs.python.org/py3k/howto/cporting.html#module-initialization-and-state
*/

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "Error!");
    return NULL;
}

static PyMethodDef smlr_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int smlr_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int smlr_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "smlrc",
        NULL,
        sizeof(struct module_state),
        smlr_methods,
        NULL,
        smlr_traverse,
        smlr_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_smlrc(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initsmlrc(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("smlrc", smlr_methods);
#endif
    struct module_state *st = GETSTATE(module);

    if (module == NULL) {
        INITERROR;
    }
    st->error = PyErr_NewException("smlrc.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
/* End of Python 3 compatibility layer */ 

/* Workaround for Python 3, which does not define the DL_EXPORT macro any more */
#ifndef DL_EXPORT     /* declarations for DLL import/export */
#define DL_EXPORT(RTYPE) RTYPE
#endif

DL_EXPORT(int)
stepwise_regression(int w_rows, int w_cols, double w[],
			int X_rows, int X_cols, double X[],
			int XY_rows, int XY_cols, double XY[],
			int Xw_rows, int Xw_cols, double Xw[],
			int E_rows, int E_cols, double E[],
			int ac_rows, double ac[],
			int lm_2_ac_rows, double lm_2_ac[],
			int S_rows, double S[],
			int M,
			int maxiter,
			double convergence_tol,
			float resamp_decay,
			float min_resamp,
			int verbose,
			long long int seed)
{
  // initialize the iterative optimization
  double incr = DBL_MAX;
  long non_zero = 0;
  long wasted_basis = 0;
  long needed_basis = 0;
  int changed = 0;

  // for calculating stepwise changes
  double w_old;
  double w_new;
  double w_diff;
  double grad;
  double XdotP;
  double E_new_m;
  double sum2_w_diff;
  double sum2_w_old;

  long cycle = 0;
  int basis = 0;
  int m = 0;
  float rval = 0;

  // get the num features and num classes
  int nd = w_rows;
  int ns = E_rows;

  // loop indexes
  int i = 0;

  // pointers to elements to avoid explicit indexing
  double* Sp = (double*) NULL;
  double* Ep = (double*) NULL;
  double* Xp = (double*) NULL;
  double* Xwp = (double*) NULL;

  // prob of resample each weight
  // allocate everything in heap -- not on stack
  float** p_resamp = (float **)calloc(w_rows, sizeof(float*));

  for (i=0; i<w_rows; i++)
    p_resamp[i] = (float*)calloc(w_cols, sizeof(float));

  // initialize random seed
  if (seed == 0)
    seed = (long long int)time(NULL);

  if (verbose)
  {
    fprintf(stdout, "SMLR: random seed=%lld\n", seed);
    fflush(stdout);
  }

  srand (seed);

  // loop over cycles

  i = 0;
  for (cycle=0; cycle<maxiter; cycle++)
  {
    // zero out the diffs for assessing change
    sum2_w_diff = 0.0;
    sum2_w_old = 0.0;
    wasted_basis = 0;
    if (cycle==1)
      needed_basis = 0;

    // update each weight
    for (basis=0; basis<nd; basis++)
    {
      for (m=0; m<w_cols; m++)
      {
	// get the starting weight
	w_old = w[w_cols*basis+m];

	// set the p_resamp if it's the first cycle
	if (cycle == 0)
	{
	  p_resamp[basis][m] = 1.0;
	}

	// see if we're gonna update
	rval = (float)rand()/(float)RAND_MAX;
	if ((w_old != 0) || (rval < p_resamp[basis][m]))
	{
	  // calc the probability
	  XdotP = 0.0;
	  for (i=0, Xp=X+basis, Ep=E+m;
	       i<ns; i++)
	  {
	    XdotP += (*Xp) * (*Ep)/S[i];
	    Xp += X_cols;
	    Ep += E_cols;
	  }

	  // get the gradient
	  grad = XY[XY_cols*basis+m] - XdotP;

	  // set the new weight
	  w_new = w_old + grad/ac[basis];

	  // test that we're within bounds
	  if (w_new > lm_2_ac[basis])
	  {
	    // more towards bounds, but keep it
	    w_new -= lm_2_ac[basis];
	    changed = 1;

	    // umark from being zero if necessary
	    if (w_old == 0.0)
	    {
	      non_zero += 1;

	      // reset the p_resample
	      p_resamp[basis][m] = 1.0;

	      // we needed the basis
	      needed_basis += 1;
	    }
	  }
	  else if (w_new < -lm_2_ac[basis])
	  {
	    // more towards bounds, but keep it
	    w_new += lm_2_ac[basis];
	    changed = 1;

	    // umark from being zero if necessary
	    if (w_old == 0.0)
	    {
	      non_zero += 1;

	      // reset the p_resample
	      p_resamp[basis][m] = 1.0;

	      // we needed the basis
	      needed_basis += 1;
	    }

	  }
	  else
	  {
	    // gonna zero it out
	    w_new = 0.0;

	    // decrease the p_resamp
	    p_resamp[basis][m] -= (p_resamp[basis][m] - min_resamp) * resamp_decay;

	    // set the number of non-zero
	    if (w_old == 0.0)
	    {
	      // we didn't change
	      changed = 0;

	      // and wasted a basis
	      wasted_basis += 1;
	    }
	    else
	    {
	      // we changed
	      changed = 1;

	      // must update num non_zero
	      non_zero -= 1;
	    }
	  }

	  // process changes if necessary
	  if (changed == 1)
	  {
	    // update the expected values
	    w_diff = w_new - w_old;
	    for (Sp=S, Xp=X+basis, Ep=E+m, Xwp=Xw+m;
		 Sp<S+S_rows; Sp++)
	    {
	      (*Xwp) += (*Xp)*w_diff;
	      E_new_m = exp(*Xwp);
	      *Sp += E_new_m - *Ep;
	      *Ep = E_new_m;

	      Xp += X_cols;
	      Ep += E_cols;
	      Xwp += Xw_cols;
	    }

	    // update the weight
	    w[w_cols*basis+m] = w_new;

	    // keep track of the sqrt sum squared diffs
	    sum2_w_diff += w_diff*w_diff;
	  }

	  // no matter what we keep track of the old
	  sum2_w_old += w_old*w_old;
	}
      }
    }

    // finished a cycle, assess convergence
    incr = sqrt(sum2_w_diff) / (sqrt(sum2_w_old)+DBL_EPSILON);

    if (verbose)
    {
      fprintf(stdout, "SMLR: cycle=%ld ; incr=%g ; non_zero=%ld ; wasted_basis=%ld ; needed_basis=%ld ; sum2_w_old=%g ; sum2_w_diff=%g\n",
	      cycle, incr, non_zero, wasted_basis, needed_basis, sum2_w_old, sum2_w_diff);
      fflush(stdout);
    }

    if (incr < convergence_tol)
    {
      // we converged!!!
      break;
    }
  }

  // finished updating weights
  // assess convergence

  // free up used heap
  for (i=0; i<w_rows; i++)
    free(p_resamp[i]);

  free(p_resamp);

  return cycle;
}

