#!/usr/bin/env bash

set -eu

newver=$1
oldver=`PYTHONPATH=. python  -c 'import mvpa2; print(mvpa2.__version__)'`

sed -i -e "s,$oldver,$newver,g" mvpa2/__init__.py
sed -i -e "s, version='.*, version='$newver'\,,g" setup.py