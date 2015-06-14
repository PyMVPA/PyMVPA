# emacs: -*- mode: python; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 noet:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Stub file for a guaranteed safe import of duecredit constructs:  if duecredit is not
available.

To use it, just place it into your project codebase to be imported, e.g. copy as

    cp stub.py /path/tomodule/module/due.py

Note that it might be better to avoid naming it duecredit.py to avoid shadowing
installed duecredit.

Then use in your code as

    from .due import due


Examples
--------

TODO


License:
Originally a part of the duecredit, which is distributed under BSD-2 license.
"""

__version__ = '0.0.1'

class InactiveDueCreditCollector(object):
    def _donothing(self, *args, **kwargs):
        pass

    def dcite(self, *args, **kwargs):
        def nondecorating_decorator(func):
             return func
        return nondecorating_decorator

    cite = load = add = _donothing

    def __repr__(self):
        return self.__class__.__name__ + '()'

def _donothing_func(*args, **kwargs):
    pass

try:
    from duecredit import *
except ImportError:
    # Initiate due stub
    due = InactiveDueCreditCollector()
    BibTeX = Doi = Donate = _donothing_func
except Exception as e:
    import logging
    logging.getLogger("duecredit").error(
        "Failed to import duecredit due to %s" % str(e))
    # TODO: remove duplication
    due = InactiveDueCreditCollector()
    BibTeX = Doi = Donate = _donothing_func
