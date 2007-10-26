#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Data split (aka Cross-Validation fold) processing"""


class SplitProcessor(object):
    """
    Base/dummy class

      splitter    - splitter instance used to generate the dataset split
      split       - the actual datasplit tuple (returned by the splitter)
      classifier  - classifier instance trained on the first dataset in the
                   'split' tuple

    Every SplitProcessing subclass has to implement a __call__() method
    that returns the result of the processing. The __call__() method has to deal
    with multiple calls to it and must make sure that previously returned
    results are not modified (e.g. when sharing NumPy array).
    """
    def __call__(self, splitter, split, classifier):
        raise NotImplementedError
