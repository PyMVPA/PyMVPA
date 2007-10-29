#!/bin/bash
#emacs: -*- mode: shell-script; c-basic-offset: 4; tab-width: 4; indent-tabs-mode: t -*- 
#ex: set sts=4 ts=4 sw=4 noet:
#-------------------------- =+- Shell script -+= --------------------------
#
# @file      replace_header.sh
# @date      Fri Oct 26 15:34:06 2007
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

#echo "./tests/crossval.py" | \
#echo "tests/test_algorithms.py" | \
#echo "mvpa/datasets/mapper.py" | \
find -iname '*.py' | \
	while read fname; do
		grep -q  'This package is distributed in the hope' $fname || continue
		
		descr="`grep '[#"]* *PyMVPA: ' "$fname"  | head -1 | sed -e 's/^.*PyMVPA: //g' -e 's/"""//g'`"
		[ "$descr" == "" ] && \
			descr="`sed -n -e '/###/,/^[^#]/p' "$fname"  | grep '"""' | head -1 | sed -e 's/^"""\(PyMVPA: *\)*\(.*\)"""/\2/g'`"
		[ "$descr" == "" ] && \
			descr="`sed -n -e '/###/,/^[^#]/p' $fname | sed -n -e '3s/^# *//gp'`"
		echo "$fname:$descr"

		cat $fname \
			| sed -e '0,/### ###/d' -e '0,/### ###/d' \
			| sed -e '1rdoc/misc/header.py' \
			| sed -e 's/\t/    /g' \
			| sed -n -e "s/\(\"PyMVPA: \)\"/\1${descr}\"/g" -e '2,$p' \
            | sed -e 's/ -- loosely implemented//g' \
			| sponge $fname

done

exit 0
problematic: mvpa/misc/fsl/__init__.py -- removed header completely for some reason...
 tests/test_algorithms.py os gone!
mvpa/datasets/mapper.py is gone
