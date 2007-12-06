#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Verbose output and debugging facility

Examples:
from verbosity import verbose, debug; debug.active = [1,2,3]; debug(1, "blah")

"""

from sys import stdout

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
        self.__lfprev = True
        self.__crprev = 0               # number of symbols in previous cr-ed


    def _setHandlers(self, handlers):
        """Set list of handlers for the log.

        Handlers can be logfiles, stdout, stderr, etc
        """
        self.__handlers = handlers


    def _getHandlers(self):
        """Return active handlers
        """
        return self.__handlers


    def __call__(self, msg, lf=True, cr=False, *args, **kwargs):
        """
        Write msg to each of the handlers.

        It can append a newline (lf = Line Feed) or return
        to the beginning before output and to take care about
        cleaning previous message if present

        it appends a newline (lf = Line Feed) since most commonly each
        call is a separate message
        """
        if cr:
            msg_ = ""
            if self.__crprev > 0:
                # wipe out older line to make sure to see no ghosts
                msg_ = "\r%s\r" % (" "*self.__crprev)
            msg_ += msg
            self.__crprev = len(msg)
            msg = msg_
            # since it makes no sense this days for cr and lf,
            # override lf
            lf = False
        else:
            self.__crprev += len(msg)

        if lf:
            msg = msg + "\n"
            self.__crprev = 0           # nothing to clear

        for handler in self.__handlers:
            handler.write(msg)
            try:
                handler.flush()
            except:
                # it might be not implemented..
                pass

        self.__lfprev = lf

    handlers = property(fget=_getHandlers, fset=_setHandlers)
    lfprev = property(fget=lambda self:self.__lfprev)



class LevelLogger(Logger):
    """
    Logger which prints based on level -- ie everything which is smaller
    than specified level
    """

    def __init__(self, level=0, indent=" ", *args, **kwargs):
        """Define level logger.

        It is defined by
          `level`, int: to which messages are reported
          `indent`, string: symbol used to indent
        """
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
        self.__indent = "%s" % indent


    def __call__(self, level, msg, *args, **kwargs):
        """
        Write msg and indent using self.indent it if it was requested

        it appends a newline since most commonly each call is a separate
        message
        """
        if level <= self.level:
            if self.lfprev and self.indent:
                # indent if previous line ended with newline
                msg = self.indent*level + msg
            Logger.__call__(self, msg, *args, **kwargs)

    level = property(fget=lambda self: self.__level, fset=_setLevel)
    indent = property(fget=lambda self: self.__indent, fset=_setIndent)


class OnceLogger(Logger):
    """
    Logger which prints a message for a given ID just once.

    It could be used for one-time warning to don't overfill the output
    with useless repeatative messages
    """

    def __init__(self, *args, **kwargs):
        """Define once logger.
        """
        Logger.__init__(self, *args, **kwargs)
        self._known = {}


    def __call__(self, ident, msg, count=1, *args, **kwargs):
        """
        Write `msg` if `ident` occured less than `count` times by now.
        """
        if not self._known.has_key(ident):
            self._known[ident] = 0

        if self._known[ident] < count:
            self._known[ident] += 1
            Logger.__call__(self, msg, *args, **kwargs)


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


    def __call__(self, setid, msg, *args, **kwargs):
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
            Logger.__call__(self, msg, *args, **kwargs)


    def register(self, setid, description):
        """ "Register" a new setid with a given description for easy finding
        """

        if self.__registered.has_key(setid):
            raise ValueError, \
                  "Setid %s is already known with description '%s'" % \
                  ( `setid`, self.__registered[setid] )
        self.__registered[setid] = description


    def setActiveFromString(self, value):
        """Given a string listing registered(?) setids, make then active
        """
        # somewhat evil but works since verbose must be initiated
        # by now
        from mvpa.misc import verbose
        entries = value.split(",")
        if entries != "":
            if 'ALL' in entries or 'all' in entries:
                verbose(2, "Enabling all registered debug handlers")
                entries = self.registered.keys()

            verbose(2, "Enabled debug handlers: %s" % `entries`)
            self.active = entries


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
                          "Unknown name %s for metric in DebugLogger" % \
                          func + " Known metrics are " + \
                          `DebugLogger._known_metrics.keys()`
            self.__metrics.append(func)


        def __call__(self, setid, msg, *args, **kwargs):
            msg_ = ""

            for metric in self.__metrics:
                msg_ += " %s" % `metric()`

            if len(msg_)>0:
                msg_ = "{%s}" % msg_

            SetLogger.__call__(self, setid, "DEBUG%s: %s" % (msg_, msg),
                               *args, **kwargs)

