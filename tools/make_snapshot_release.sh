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

# clone repository locally so we don't drag any possible trash along

git clone -l . dist/pymvpa-snapshot
cd dist/pymvpa-snapshot

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

cd ..

# Wipe out the distribution clone
rm -rf dist/pymvpa-snapshot
