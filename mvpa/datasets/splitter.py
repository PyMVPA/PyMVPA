### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Base class of all dataset splitter
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

class Splitter(object):
    """ Base class of a data splitter.

    Each splitter should be initialized with all its necessary parameters. The
    final splitting is done running the splitter object on a certain Dataset
    via __call__(). This method has to be implemented like a generator, i.e. it
    has to return every possible split with a yield() call.

    Each split has to be returned as a tuple of Dataset(s). The properties
    of the splitted dataset may vary between implementations. It is possible
    to declare tuple element as 'None'. 

    Please note, that even if there is only one Dataset returned it has to be
    an element in a tupleand not just the Dataset object!
    """
    def __call__(self, dataset):
        """
        """
        raise NotImplementedError



class NoneSplitter(Splitter):
    """ This is a dataset splitter that does NOT split. It simply returns the
    full dataset that it is called with.
    """
    def __call__(self, dataset):
        """ This splitter returns the passed dataset as the second element of
        a 2-tuple. The first element of that tuple will always be 'None'.
        """
        return (None, dataset)
