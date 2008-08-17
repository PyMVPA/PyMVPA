#!/bin/bash
#emacs: -*- mode: shell-script; c-basic-offset: 4; tab-width: 4; indent-tabs-mode: t -*- 
#ex: set sts=4 ts=4 sw=4 noet:
#-------------------------- =+- Shell script -+= --------------------------
#
# @file      refactor.sh
# @date      Fri Oct 19 16:14:27 2007
# @brief
#
#
#  Yaroslav Halchenko                                      CS@UNM, CS@NJIT
#  web:     http://www.onerussian.com                      & PSYCH@RUTGERS
#  e-mail:  yoh@onerussian.com                              ICQ#: 60653192
#
# DESCRIPTION (NOTES):
#
# COPYRIGHT: Yaroslav Halchenko 2007
#
# LICENSE:
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the
#  Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
# On Debian system see /usr/share/common-licenses/GPL for the full license.
#
#-----------------\____________________________________/------------------


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
#	-e 's,training_confusions,confusion,g'

