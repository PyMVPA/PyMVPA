#!/bin/sh
#
# Little helper to create a fresh snapshot tarball, wit a proper dev-version and
# also the corresponding Debian source package.
# Must be run in the -dev packaging branch, in a clean working tree.
#
if [ ! -d debian ]; then
	echo "This command must be run in a debian packaging branch."
	exit 1
fi

if [ ! "x$(git diff)" = "x" ]; then
	echo "The working directory is not clean. Please commit all changes first."
	exit 1
fi

# create the tarball, and set fresh versions
make orig-src

# create the Debian source package
cwd=$(pwd)
cd ..
dpkg-source -i -b $(basename $cwd)
cd $cwd

# prevent accidental resetting of something else
if [ ! "$cwd" = "$(pwd)" ]; then
	echo "Something went wrong. We are not were we started."
	exit 1
fi

# safe to do, since we checked for no diff in the beginning
# this will only wipe out temporary version changes
git reset --hard
