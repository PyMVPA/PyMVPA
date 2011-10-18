#!/bin/bash
# emacs: -*- mode: shell-script; c-basic-offset: 4; tab-width: 4; indent-tabs-mode: t -*-
# vi: set ft=sh sts=4 ts=4 sw=4 noet:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

#echo "./tests/crossval.py" | \
#echo "tests/test_algorithms.py" | \
#echo "mvpa2/datasets/mapper.py" | \
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
problematic: mvpa2/misc/fsl/__init__.py -- removed header completely for some reason...
 tests/test_algorithms.py os gone!
mvpa2/datasets/mapper.py is gone
