#!/bin/bash

# This is the script to be used with git bisect to figure out first
# commit when testBasic starts to fail

# to use it just do
#  git bisect start yoh/master maint/0.4 --
#  git bisect run tools/bisect_mtrand.sh
# where yoh/master is known place where that test fails
# and maint/0.4 where it doesn't
#
# it would stop at the first bad commit

make clean
make || exit 125 # skip commits where build is broken

PYTHONPATH=$PWD nosetests mvpa2/tests/test_datameasure.py:SensitivityAnalysersTests.testBasic
