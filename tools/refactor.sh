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
	-e 's,training_confusions,confusion,g'

