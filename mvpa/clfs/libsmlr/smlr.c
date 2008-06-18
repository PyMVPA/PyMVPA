/*emacs: -*- mode: c-mode; tab-width: 8; c-basic-offset: 2; indent-tabs-mode: t -*-
  ex: set sts=4 ts=8 sw=4 noet: */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <Python.h>

DL_EXPORT(int)
stepwise_regression(int w_rows, int w_cols, double w[w_rows][w_cols],
			int X_rows, int X_cols, double X[X_rows][X_cols],
			int XY_rows, int XY_cols, double XY[XY_rows][XY_cols],
			int Xw_rows, int Xw_cols, double Xw[Xw_rows][Xw_cols],
			int E_rows, int E_cols, double E[E_rows][E_cols],
			int ac_rows, double ac[ac_rows],
			int lm_2_ac_rows, double lm_2_ac[lm_2_ac_rows],
			int S_rows, double S[S_rows],
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

  // get the num features and num classes
  int nd = w_rows;
  int ns = E_rows;

  // loop indexes
  int i = 0;

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
  long cycle = 0;
  int basis = 0;
  int m = 0;
  float rval = 0;
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
	w_old = w[basis][m];

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
	  for (i=0; i<ns; i++)
	  {
	    XdotP += X[i][basis] * E[i][m]/S[i];
	  }

	  // get the gradient
	  grad = XY[basis][m] - XdotP;

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
	    for (i=0; i<ns; i++)
	    {
	      Xw[i][m] += X[i][basis]*w_diff;
	      E_new_m = exp(Xw[i][m]);
	      S[i] += E_new_m - E[i][m];
	      E[i][m] = E_new_m;
	    }

	    // update the weight
	    w[basis][m] = w_new;

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

/* make dummy module definition to satisfy distutils on win32
 * which cannot compile non-extension libraries
 */
PyMODINIT_FUNC initsmlrc(void)
{
        Py_InitModule3("smlrc", NULL, "");
}


