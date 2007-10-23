### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Data split (aka Cross-Validation fold) processing
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


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
