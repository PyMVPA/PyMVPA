#!/bin/sh

set -e
set -u

#% EXAMPLE START

#% Query properties and features of a PyMVPA installation
#% ======================================================

#% The ``info``` command is the central interface to get information on a PyMVPA
#% installation.

#% PyMVPA has versatile debugging capabilities and allows for fine-grained
#% access to process information. The ``--debug`` option lists all configured
#% debug channels to can be enabled via the MVPA_DEBUG environment variable.

# what was the name of the debug channel for 'searchlight' analyses?
pymvpa2 info --debug | grep -i searchlight

#% With ``--external`` information on known software dependencies can be
#% retrieved.

# what version of nibabel is used by pymvpa?
pymvpa2 info --externals | grep nibabel

#% When run without arguments, the ``info`` command generates a comprehensive
#% report on the computational environment that is very useful for productive
#% bug reporting.

# create a description of the computing environment that can be posted on the
# pymvpa mailing list to make a bug report more informative
pymvpa2 info

#% EXAMPLE END
