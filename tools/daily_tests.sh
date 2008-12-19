#!/bin/bash
#emacs: -*- mode: shell-script; indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
# Helper to run all the tests daily at night
#

set -eu

# what branches to test
BRANCHES='master yoh/master mh/master maint/0.4'
# where to send reports
EMAILS=yoh@onerussian.com
# michael.hanke@gmail.com

repo=git://git.debian.org/git/pkg-exppsy/pymvpa.git

tmpdir=/tmp/pymvpa_tests-$(date +"20%y%m%d-%H:%M:%S")
mkdir $tmpdir
tmpfile=$tmpdir.tmp
logfile=$tmpdir.log
trap "echo 'I: removing $tmpdir'; rm -fr $tmpdir;" EXIT

indent() {
	sed -e 's/^/  /g'
}

do_checkout() {
    git clean -df | indent
    git reset --hard
    if [ ! $branch = 'master' ]; then
    	git checkout -b $branch origin/$branch
    fi
    git checkout $branch
}

do_build() {
    make clean
    make
}

do_test() {
    make -k test
}

ACTIONS='checkout build test'

{
    cd $tmpdir

    echo "I:" $(date)

    # checkout the repository
    echo "I: Cloning repository"
    #git clone $repo 2>&1 | indent
    #cd pymvpa
	failed=0
	succeeded=0
    #
    # Sweep through the branches and actionsto test
    #
	branches_with_problems=
    for branch in $BRANCHES; do
		branch_has_problems=
		echo
		echo "I: ---------------{ Branch $branch }--------------"
		for action in $ACTIONS; do
			echo -n "I: $action "
			cmd="do_$action"
			if $cmd >| $tmpfile 2>&1 ; then
				echo " ok"
				succeeded=$(($succeeded+1))
			else
				branch_has_problems+=" $action"
				failed=$(($failed+1))
				echo " ! FAILED ! Output was:"
				cat $tmpfile | indent
			fi
		done
		if [ "x$branch_has_problems" != x ]; then
			branches_with_problems+="\n  $branch: $branch_has_problems"
		fi
    done
	echo "I: Succeeded $succeeded actions, failed $failed."
	if [ "x$branches_with_problems" != x ]; then
		echo -e "I: Branches which experienced problems: $branches_with_problems"
	fi
} 2>&1 | tee $logfile

echo "I: Exiting. Logfile $logfile"
