# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

__docformat__ = 'restructuredtext'

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
    """Base class to provide logging
    """

    def __init__(self, handlers=None):
        """Initialize the logger with a set of handlers to use for output

        Each hanlder must have write() method implemented
        """
        if handlers == None:
            handlers = [stdout]
        self.__close_handlers = []
        self.__handlers = []            # pylint friendliness
        self._set_handlers(handlers)
        self.__lfprev = True
        self.__crprev = 0               # number of symbols in previous cr-ed

    def __del__(self):
        self._close_opened_handlers()

    ##REF: Name was automagically refactored
    def _set_handlers(self, handlers):
        """Set list of handlers for the log.

        A handler can be opened files, stdout, stderr, or a string, which
        will be considered a filename to be opened for writing
        """
        handlers_ = []
        self._close_opened_handlers()
        for handler in handlers:
            if isinstance(handler, basestring):
                try:
                    handler = {'stdout' : stdout,
                               'stderr' : stderr}[handler.lower()]
                except:
                    try:
                        handler = open(handler, 'w')
                        self.__close_handlers.append(handler)
                    except:
                        raise RuntimeError, \
                              "Cannot open file %s for writing by the logger" \
                              % handler
            handlers_.append(handler)
        self.__handlers = handlers_

    ##REF: Name was automagically refactored
    def _close_opened_handlers(self):
        """Close opened handlers (such as opened logfiles
        """
        for handler in self.__close_handlers:
            handler.close()

    ##REF: Name was automagically refactored
    def _get_handlers(self):
        """Return active handlers
        """
        return self.__handlers


    def __call__(self, msg, args=None, lf=True, cr=False, **kwargs):
        """Write msg to each of the handlers.

        It can append a newline (lf = Line Feed) or return
        to the beginning before output and to take care about
        cleaning previous message if present

        it appends a newline (lf = Line Feed) since most commonly each
        call is a separate message
        """

        if args is not None:
            try:
                msg = msg % args
            except Exception as e:
                msg = "%s [%% FAILED due to %s]" % (msg, e)

        if 'msgargs' in kwargs:
            msg = msg % kwargs['msgargs']

        if cr:
            msg_ = ""
            if self.__crprev > 0:
                # wipe out older line to make sure to see no ghosts
                msg_ = "\r%s" % (" "*self.__crprev)
            msg_ += "\r" + msg
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
            try:
                handler.write(msg)
            except:
                print "Failed writing on handler %s" % handler
                raise
            try:
                handler.flush()
            except:
                # it might be not implemented..
                pass

        self.__lfprev = lf

    handlers = property(fget=_get_handlers, fset=_set_handlers)
    lfprev = property(fget=lambda self:self.__lfprev)



class LevelLogger(Logger):
    """Logger not to log anything with a level smaller than specified.
    """

    def __init__(self, level=0, indent=" ", *args, **kwargs):
        """
        Parameters
        ----------
        level : int, optional
          Level to consider be active.
        indent : str, optional
          String to use for indentation.
        """
        Logger.__init__(self, *args, **kwargs)
        self.__level = level            # damn pylint ;-)
        self.__indent = indent
        self._set_level(level)
        self._set_indent(indent)

    ##REF: Name was automagically refactored
    def _set_level(self, level):
        """Set logging level
        """
        if __debug__:
            try:
                from mvpa2.base import debug
                debug('VERBOSE', 'Setting verbosity to %r from %r',
                      (self.__level, level))
            except:
                pass
        ilevel = int(level)
        if ilevel < 0:
            raise ValueError, \
                  "Negative verbosity levels (got %d) are not supported" \
                  % ilevel
        self.__level = ilevel


    ##REF: Name was automagically refactored
    def _set_indent(self, indent):
        """Either to indent the lines based on message log level"""
        self.__indent = "%s" % indent


    def __call__(self, level, msg, *args, **kwargs):
        """Write msg and indent using self.indent it if it was requested.

        It appends a newline since most commonly each call is a separate
        message
        """
        if level <= self.level:
            if self.lfprev and self.indent:
                # indent if previous line ended with newline
                msg = self.indent * level + msg
            Logger.__call__(self, msg, *args, **kwargs)

    level = property(fget=lambda self: self.__level, fset=_set_level)
    indent = property(fget=lambda self: self.__indent, fset=_set_indent)


class OnceLogger(Logger):
    """Logger which prints a message for a given ID just once.

    It could be used for one-time warning to don't overfill the output
    with useless repeatative messages.
    """

    def __init__(self, *args, **kwargs):
        """Define once logger.
        """
        Logger.__init__(self, *args, **kwargs)
        self._known = {}


    def __call__(self, ident, msg, count=1, *args, **kwargs):
        """Write `msg` if `ident` occured less than `count` times by now.
        """
        if ident not in self._known:
            self._known[ident] = 0

        if count < 0 or self._known[ident] < count:
            self._known[ident] += 1
            Logger.__call__(self, msg, *args, **kwargs)


class SetLogger(Logger):
    """Logger which prints based on defined sets identified by Id.
    """

    def __init__(self, register=None, active=None, printsetid=True,
                 *args, **kwargs):
        """
        Parameters
        ----------
        register : dict or None
          What Ids are to be known. Each item dictionary contains consists
          of concise key and a description as the value.
        active : iterable
          What Ids to consider active upon initialization.
        printsetid : bool, optional
          Either to prefix each line with the target Id of a set in which
          the line was printed to (default behavior).
        """
        if register is None:
            register = {}
        if active == None:
            active = []
        Logger.__init__(self, *args, **kwargs)
        self.__printsetid = printsetid
        self.__registered = register    # all "registered" sets descriptions
        # which to output... pointless since __registered
        self._set_active(active)
        self._set_printsetid(printsetid)


    ##REF: Name was automagically refactored
    def _set_active(self, active):
        """Set active logging set
        """
        # just unique entries... we could have simply stored Set I guess,
        # but then smth like debug.active += ["BLAH"] would not work
        from mvpa2.base import verbose
        self.__active = []
        registered_keys = self.__registered.keys()
        for item in list(set(active)):
            if item == '':
                continue
            if isinstance(item, basestring):
                if item in ['?', 'list', 'help']:
                    self.print_registered(detailed=(item != '?'))
                    raise SystemExit(0)
                if item.upper() == "ALL":
                    verbose(2, "Enabling all registered debug handlers")
                    self.__active = registered_keys
                    break
                # try to match item as it is regexp
                regexp_str = "^%s$" % item
                try:
                    regexp = re.compile(regexp_str)
                except:
                    raise ValueError, \
                          "Unable to create regular expression out of  %s" % item
                matching_keys = filter(regexp.match, registered_keys)
                toactivate = matching_keys
                if len(toactivate) == 0:
                    ids = self.registered.keys()
                    ids.sort()
                    raise ValueError, \
                          "Unknown debug ID '%s' was asked to become active," \
                          " or regular expression '%s' did not get any match" \
                          " among known ids: %s" \
                          % (item, regexp_str, ids)
            else:
                toactivate = [item]

            # Lets check if asked items are known
            for item_ in toactivate:
                if not (item_ in registered_keys):
                    raise ValueError, \
                          "Unknown debug ID %s was asked to become active" \
                          % item_
            self.__active += toactivate

        self.__active = list(set(self.__active)) # select just unique ones
        self.__maxstrlength = max([len(str(x)) for x in self.__active] + [0])
        if len(self.__active):
            verbose(2, "Enabling debug handlers: %s" % `self.__active`)


    ##REF: Name was automagically refactored
    def _set_printsetid(self, printsetid):
        """Either to print set Id at each line"""
        self.__printsetid = printsetid


    def __call__(self, setid, msg, *args, **kwargs):
        """
        Write msg

        It appends a newline since most commonly each call is a separate
        message
        """

        if setid in self.__active:
            if len(msg) > 0 and self.__printsetid:
                msg = "[%%-%ds] " % self.__maxstrlength % (setid) + msg
            Logger.__call__(self, msg, *args, **kwargs)


    def register(self, setid, description):
        """ "Register" a new setid with a given description for easy finding
        """

        if setid in self.__registered:
            raise ValueError, \
                  "Setid %s is already known with description '%s'" % \
                  (`setid`, self.__registered[setid])
        self.__registered[setid] = description


    ##REF: Name was automagically refactored
    def set_active_from_string(self, value):
        """Given a string listing registered(?) setids, make then active
        """
        # somewhat evil but works since verbose must be initiated
        # by now
        self.active = value.split(",")


    def print_registered(self, detailed=True):
        print "Registered debug entries: ",
        kd = self.registered
        rks = sorted(kd.keys())
        maxl = max([len(k) for k in rks])
        if not detailed:
            # short list
            print ', '.join(rks)
        else:
            print
            for k in rks:
                print '%%%ds  %%s' % maxl % (k, kd[k])


    printsetid = property(fget=lambda self: self.__printsetid, \
                          fset=_set_printsetid)
    active = property(fget=lambda self: self.__active, fset=_set_active)
    registered = property(fget=lambda self: self.__registered)


if __debug__:

    import os, re
    import traceback
    import time
    from os import getpid
    from os.path import basename, dirname

    __pymvpa_pid__ = getpid()


    def parse_status(field='VmSize', value_only=False):
        """Return stat information on current process.

        Usually it is needed to know where the memory is gone, that is
        why VmSize is the default for the field to spit out

        TODO: Spit out multiple fields. Use some better way than parsing proc
        """
        regex = re.compile('^%s:' % field)
        match = None
        try:
            for l in open('/proc/%d/status' % __pymvpa_pid__):
                if regex.match(l):
                    match = l.strip()
                    break
            if match:
                match = re.sub('[ \t]+', ' ', match)
        except IOError:
            pass
        if match and value_only:
            match = match.split(':', 1)[1].lstrip()
        return match

    def get_vmem_from_status():
        """Return utilization of virtual memory

        Deprecated implementation which relied on parsing proc/PID/status
        """
        rss, vms = [parse_status(field=x, value_only=True)
                  for x in ['VmRSS', 'VmSize']]
        if rss is None or vms is None:
            # So not available on this system -- signal with negatives
            # but do not crash
            return (-1, -1)
        if rss[-3:] == vms[-3:] and rss[-3:] == ' kB':
            # the same units
            rss = int(rss[:-3])                # strip from rss
            vms = int(vms[:-3])
        return (rss, vms)

    try:
        # we prefer to use psutil if available
        # and let's stay away from "externals" module for now
        # Note: importing as __Process so it does not get
        #       'queried' by autodoc leading to an exception
        #       while being unable to get values for the properties
        from psutil import Process as __Process
        __pymvpa_process__ = __Process(__pymvpa_pid__)
        __pymvpa_memory_info = __pymvpa_process__.memory_info if hasattr(__pymvpa_process__, 'memory_info') \
                             else __pymvpa_process__.get_memory_info


        def get_vmem():
            """Return utilization of virtual memory

            Generic implementation using psutil
            """
            mi = __pymvpa_memory_info()
            # in later versions of psutil mi is a named tuple.
            # but that is not the case on Debian squeeze with psutil 0.1.3
            rss = mi[0] / 1024
            vms = mi[1] / 1024
            return (rss, vms)

    except ImportError:
        get_vmem = get_vmem_from_status

    def get_vmem_str():
        """Return  a string summary about utilization of virtual_memory
        """
        vmem = get_vmem()
        try:
            return "RSS/VMS: %d/%d kB" % vmem
        except:
            return "RSS/VMS: %s" % str(vmem)

    def _get_vmem_max_str_gen():
        """Return peak vmem utilization so far.

        It is a generator, get_vmem_max_str later is bound to .next
        of it - to mimic static variables
        """
        rss_max = 0
        vms_max = 0

        while True:
            rss, vms = get_vmem()
            rss_max = max(rss, rss_max)
            vms_max = max(vms, vms_max)
            yield "max RSS/VMS: %d/%d kB" % (rss_max, vms_max)
    get_vmem_max_str = _get_vmem_max_str_gen().next

    def mbasename(s):
        """Custom function to include directory name if filename is too common

        Also strip .py at the end
        """
        base = basename(s)
        if base.endswith('.py'):
            base = base[:-3]
        if base in set(['base', '__init__']):
            base = basename(dirname(s)) + '.' + base
        return base

    class TraceBack(object):
        """Customized traceback to be included in debug messages
        """

        def __init__(self, collide=False):
            """Initialize TrackBack metric

            Parameters
            ----------
            collide : bool
              if True then prefix common with previous invocation gets
              replaced with ...
            """
            self.__prev = ""
            self.__collide = collide

        def __call__(self):
            ftb = traceback.extract_stack(limit=100)[:-2]
            entries = [[mbasename(x[0]), str(x[1])] for x in ftb]
            entries = [ e for e in entries if e[0] != 'unittest' ]

            # lets make it more consize
            entries_out = [entries[0]]
            for entry in entries[1:]:
                if entry[0] == entries_out[-1][0]:
                    entries_out[-1][1] += ',%s' % entry[1]
                else:
                    entries_out.append(entry)
            sftb = '>'.join(['%s:%s' % (mbasename(x[0]),
                                        x[1]) for x in entries_out])
            if self.__collide:
                # lets remove part which is common with previous invocation
                prev_next = sftb
                common_prefix = os.path.commonprefix((self.__prev, sftb))
                common_prefix2 = re.sub('>[^>]*$', '', common_prefix)

                if common_prefix2 != "":
                    sftb = '...' + sftb[len(common_prefix2):]
                self.__prev = prev_next

            return sftb


    class RelativeTime(object):
        """Simple helper class to provide relative time it took from previous
        invocation"""

        def __init__(self, format="%3.3f sec"):
            """
            Parameters
            ----------
            format : str
              String format to use for reporting time.
            """
            self.__prev = None
            self.__format = format

        def __call__(self):
            dt = 0.0
            ct = time.time()
            if self.__prev is not None:
                dt = ct - self.__prev
            self.__prev = ct
            return self.__format % dt


    class DebugLogger(SetLogger):
        """
        Logger for debugging purposes.

        Expands SetLogger with ability to print some interesting information
        (named Metric... XXX) about current process at each debug printout
        """

        _known_metrics = {
            # TODO: make up Windows-friendly version or pure Python platform
            # independent version (probably just make use of psutil)
            'vmem' : get_vmem_str,
            'vmem_max' : get_vmem_max_str,
            'pid' : getpid, # lambda : parse_status(field='Pid'),
            'asctime' : time.asctime,
            'tb' : TraceBack(),
            'tbc' : TraceBack(collide=True),
            }

        def __init__(self, metrics=None, offsetbydepth=True, *args, **kwargs):
            """
            Parameters
            ----------
            metrics : iterable of (func or str) or None
              What metrics (functions) to be reported.  If item is a string,
              it is matched against `_known_metrics` keys.
            offsetbydepth : bool, optional
              Either to offset lines depending on backtrace depth (default
              behavior).
            *args, **kwargs
              Passed to SetLogger initialization  XXX
            """
            if metrics == None:
                metrics = []
            SetLogger.__init__(self, *args, **kwargs)
            self.__metrics = []
            self._offsetbydepth = offsetbydepth
            self._reltimer = RelativeTime()
            self._known_metrics = DebugLogger._known_metrics
            self._known_metrics['reltime'] = self._reltimer
            for metric in metrics:
                self._registerMetric(metric)


        ##REF: Name was automagically refactored
        def register_metric(self, func):
            """Register some metric to report

            func can be either a function call or a string which should
            correspond to known metrics
            """

            if isinstance(func, basestring):
                if func in ['all', 'ALL']:
                    func = self._known_metrics.keys()

            if isinstance(func, basestring):
                if func in DebugLogger._known_metrics:
                    func = DebugLogger._known_metrics[func]
                else:
                    if func in ['?', 'list', 'help']:
                        print 'Known debug metrics: ', \
                              ', '.join(DebugLogger._known_metrics.keys())
                        raise SystemExit(0)
                    else:
                        raise ValueError, \
                              "Unknown name %s for metric in DebugLogger" % \
                              func + " Known metrics are " + \
                              `DebugLogger._known_metrics.keys()`
            elif isinstance(func, list):
                self.__metrics = []     # reset
                for item in func:
                    self.register_metric(item)
                return

            if not func in self.__metrics:
                try:
                    from mvpa2.base import debug
                    debug("DBG", "Registering metric %s" % func)
                    self.__metrics.append(func)
                except:
                    pass


        def __call__(self, setid, msg, *args, **kwargs):

            if setid not in self.registered:
                raise ValueError, "Not registered debug ID %s" % setid

            if not setid in self.active:
                # don't even compute the metrics, since they might
                # be statefull as RelativeTime
                return

            msg_ = ' / '.join([str(x()) for x in self.__metrics])

            if len(msg_) > 0:
                msg_ = "{%s}" % msg_

            if len(msg) > 0:
                # determine blank offset using backstacktrace
                if self._offsetbydepth:
                    level = len(traceback.extract_stack()) - 2
                else:
                    level = 1

                if len(msg) > 250 and 'DBG' in self.active and not setid.endswith('_TB'):
                    tb = traceback.extract_stack(limit=2)
                    msg += "  !!!2LONG!!!. From %s" % str(tb[0])

                msg = "DBG%s:%s%s" % (msg_, " "*level, msg)
                SetLogger.__call__(self, setid, msg, *args, **kwargs)
            else:
                msg = msg_
                Logger.__call__(self, msg, *args, **kwargs)




        ##REF: Name was automagically refactored
        def _set_offset_by_depth(self, b):
            self._offsetbydepth = b

        offsetbydepth = property(fget=lambda x:x._offsetbydepth,
                                 fset=_set_offset_by_depth)

        metrics = property(fget=lambda x:x.__metrics,
                           fset=register_metric)


if not __debug__:
    class BlackHoleLogger(SetLogger):
        '''A logger that does absolutely nothing - it is used as a fallback
        so that debug(...) can still be called even if not __debug__'''
        def __init__(self, metrics=None, offsetbydepth=True, *args, **kwargs):
            '''Initializes the logger - ignores all input arguments'''

            # do not be evil - initialize through the parent class
            SetLogger.__init__(self, *args, **kwargs)

        def __call__(self, setid, msg, *args, **kwargs):
            pass

        def register_metric(self, func):
            pass

        def register(self, setid, description):
            pass

        def set_active_from_string(self, value):
            pass

        def print_registered(self, detailed=True):
            print "BlackHoleLogger: nothing registered "
