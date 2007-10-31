#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Verbose output and debugging facility

Examples:
from verbosity import verbose, debug; debug.active = [1,2,3]; debug(1, "blah")

"""

from sys import stdout, stderr

# GOALS
#  any logger should be able
#   to log into a file or stdout/stderr
#   provide ability to log with/without a new line at the end
#
#  debug logger should be able
#    to log sets of debug statements
#    add/remove debug setid items
#    give verbose description about registered debugset items

class Logger(object):
    """
    Base class to provide logging
    """

    def __init__(self, handlers=[stdout]):
        """ Initialize the logger with a set of handlers to use for output

        Each hanlder must have write() method implemented
        """
        self._setHandlers(handlers)
        self.__newlineprev = True


    def _setHandlers(self, handlers):
        """Set list of handlers for the log.

        Handlers can be logfiles, stdout, stderr, etc
        """
        self.__handlers = handlers


    def _getHandlers(self):
        """Return active handlers
        """
        return self.__handlers


    def __call__(self, msg, newline=True):
        """
        Write msg to each of the handlers, appending newline if requested

        it appends a newline since most commonly each call is a separate
        message
        """
        if newline:
            msg = msg + "\n"
        for handler in self.__handlers:
            handler.write(msg)
        self.__newlineprev = newline


    handlers = property(fget=_getHandlers, fset=_setHandlers)
    newlineprev = property(fget=lambda self:self.__newlineprev)



class LevelLogger(Logger):
    """
    Logger which prints based on level -- ie everything which is smaller
    than specified level
    """

    def __init__(self, level=0, indent=True, *args, **kwargs):
        Logger.__init__(self, *args, **kwargs)
        self.__level = level            # damn pylint ;-)
        self.__indent = indent
        self._setLevel(level)
        self._setIndent(indent)

    def _setLevel(self, level):
        """Set logging level
        """
        ilevel = int(level)
        if ilevel < 0:
            raise ValueError, \
                  "Negative verbosity levels (got %d) are not supported" \
                  % ilevel
        self.__level = ilevel


    def _setIndent(self, indent):
        """Either to indent the lines based on message log level"""
        self.__indent = indent


    def __call__(self, level, msg, newline=True):
        """
        Write msg and space indent it if it was requested

        it appends a newline since most commonly each call is a separate
        message
        """
        if level >= self.level:
            if self.newlineprev and self.indent:
                # indent if previous line ended with newline
                msg = " "*level + msg
            Logger.__call__(self, msg, newline)

    level = property(fget=lambda self: self.__level, fset=_setLevel)
    indent = property(fget=lambda self: self.__indent, fset=_setIndent)


class SetLogger(Logger):
    """
    Logger which prints based on defined sets identified by Id.
    """

    def __init__(self, active=[], printsetid=True, *args, **kwargs):
        Logger.__init__(self, *args, **kwargs)
        self.__active = active    # sets which to output
        self.__printsetid = printsetid
        self.__registered = {}      # all "registered" sets descriptions
        self._setActive(active)
        self._setPrintsetid(printsetid)


    def _setActive(self, active):
        """Set active logging set
        """
        self.__active = active


    def _setPrintsetid(self, printsetid):
        """Either to print set Id at each line"""
        self.__printsetid = printsetid


    def __call__(self, setid, msg, newline=True):
        """
        Write msg

        It appends a newline since most commonly each call is a separate
        message
        """
        if not self.__registered.has_key(setid):
            self.__registered[setid] = "No Description"

        if setid in self.__active:
            if self.__printsetid:
                msg = "[%s] " % (setid) + msg
            Logger.__call__(self, msg, newline)


    def register(self, setid, description):
        """ "Register" a new setid with a given description for easy finding
        """

        if self.__registered.has_key(setid):
            raise ValueError, \
                  "Setid %s is already known with description '%s'" %\
                  ( `setid`, self.__registered[setid] )
        self.__registered[setid] = description


    printsetid = property(fget=lambda self: self.__printsetid, \
                          fset=_setPrintsetid)
    active = property(fget=lambda self: self.__active, fset=_setActive)
    registered = property(fget=lambda self: self.__registered)


if __debug__:

    import os, re

    def parseStatus(field='VmSize'):
        """Return stat information on current process.

        Usually it is needed to know where the memory is gone, that is
        why VmSize is the default for the field to spit out
        TODO: Spit out multiple fields. Use some better way than parsing proc
        """

        fd = open('/proc/%d/status'%os.getpid())
        lines = fd.readlines()
        fd.close()
        return filter(lambda x:re.match('^%s:'%field, x), lines)[0].strip()


    class DebugLogger(SetLogger):
        """
        Logger for debugging purposes.

        Expands SetLogger with ability to print some interesting information
        (named Metric... XXX) about current process at each debug printout
        """

        _known_metrics = {
            'vmem' : lambda : parseStatus(field='VmSize'),
            'pid' : lambda : parseStatus(field='Pid')
            }


        def __init__(self, metrics=[], *args, **kwargs):
            SetLogger.__init__(self, *args, **kwargs)
            self.__metrics = []
            for metric in metrics:
                self._registerMetric(metric)


        def registerMetric(self, func):
            """Register some metric to report

            func can be either a function call or a string which should
            correspond to known metrics
            """
            if isinstance(func, basestring):
                if DebugLogger._known_metrics.has_key(func):
                    func = DebugLogger._known_metrics[func]
                else:
                    raise ValueError, \
                          "Unknown name %s for metric in DebugLogger" %\
                          func + " Known metrics are " + \
                          `DebugLogger._known_metrics.keys()`
            self.__metrics.append(func)


        def __call__(self, setid, msg, *args, **kwargs):
            msg_ = ""

            for metric in self.__metrics:
                msg_ += " %s" % `metric()`

            if len(msg_)>0:
                msg_ = "{%s}" % msg_

            SetLogger.__call__(self, setid, "DEBUG%s: %s" % (msg_, msg))

    # TODO: check if they are actually singletons...

    # NOTE: all calls to debug must be preconditioned with
    # if __debug__:
    debug = DebugLogger()

verbose = LevelLogger()

