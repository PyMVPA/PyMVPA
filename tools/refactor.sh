#!/bin/bash
# emacs: -*- mode: shell-script; c-basic-offset: 4; tab-width: 4; indent-tabs-mode: t -*-
# vi: set ft=sh sts=4 ts=4 sw=4 noet:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


git grep -l mvpa | grep -v refactor | \
 xargs sed -i \
 -e 's,optHelp,opt.help,g' \
 -e "s,optVerbose,opt.verbose,g" \
 -e "s,optClf,opt.clf,g" \
 -e "s,optRadius,opt.radius,g" \
 -e "s,optKNearestDegree,opt.knearestdegree,g" \
 -e "s,optSVMC,opt.svm_C,g" \
 -e "s,optSVMNu,opt.svm_nu,g" \
 -e "s,optSVMGamma,opt.svm_gamma,g" \
 -e "s,optCrossfoldDegree,opt.crossfolddegree,g" \
 -e "s,optZScore,opt.zscore,g" \
 -e "s,optTr,opt.tr,g" \
 -e "s,optDetrend,opt.detrend,g" \
 -e "s,optBoxLength,opt.boxlength,g" \
 -e "s,optBoxOffset,opt.boxoffset,g" \
 -e "s,optChunk,opt.chunk,g" \
 -e "s,optChunkLimits,opt.chunklimits,g" \
 -e "s,optsCommon,opts.common,g" \
 -e "s,optsKNN,opts.KNN,g" \
 -e "s,optsSVM,opts.SVM,g" \
 -e "s,optsGener,opts.general,g" \
 -e "s,optsPreproc,opts.preproc,g" \
 -e "s,optsBox,opts.box,g" \
 -e "s,optsChunk,opts.chunk,g"

## Uncomment and move up any additional refactorings which needed
#	-e 's,training_stats,stats,g'

