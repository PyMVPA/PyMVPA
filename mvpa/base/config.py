#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Registry-like monster"""

__docformat__ = 'restructuredtext'

from ConfigParser import SafeConfigParser
import os.path

if __debug__:
    from mvpa.misc import debug


class Config(SafeConfigParser):
    """Central configuration registry for PyMVPA.

    The purpose of this class is to collect all configurable settings used by
    various parts of PyMVPA. It is fairly simple and does only little more
    than the standard Python ConfigParser. Like ConfigParser it is blind to the
    data that it stores, i.e. not type checking is performed.

    Configuration files (INI syntax) in multiple location are passed when the
    class is instanciated or whenever `Config.reload()` is called later on.
    By default it looks for a config file named `pymvpa.cfg` in the current
    directory and `.pymvpa.cfg` in the user's home directory. Morever, the
    constructor takes an optional argument with a list of additional file names
    to parse.

    In addition to configuration files, this class also looks for special
    environment variables to read settings from. Names of such variables have to
    start with `MVPA_` following by the an optional section name and the
    variable name itself ('_' as delimiter). If no section name is provided,
    the variables will be associated with section `general`. Some examples::

        MVPA_VERBOSE=1

    will become::

        [general]
        verbose = 1

    However, `MVPA_VERBOSE_OUTPUT=stdout` becomes::

        [verbose]
        output = stdout

    Any lenght of variable name as allowed, e.g. MVPA_SEC1_LONG_VARIABLE_NAME=1
    becomes::

        [sec1]
        long variable name = 1

    Settings from custom configuration files (specified by the constructor
    argument) have the highest priority and override settings found in the
    current directory. They in turn override user-specific settings and finally
    the content of any `MVPA_*` environment variables overrides all settings
    read from any file.
    """

    # things we want to count on to be available
    _DEFAULTS = {'general':
                  {
                    'verbose': '1',
                  }
                }


    def __init__(self, filenames=None):
        """Initialization reads settings from config files and env. variables.

        :Parameters:
          filenames: list of filenames
        """
        SafeConfigParser.__init__(self)

        # store additional config file names
        if not filenames is None:
            self.__cfg_filenames = filenames
        else:
            self.__cfg_filenames = []

        # set critical defaults
        for sec, vars in Config._DEFAULTS.iteritems():
            self.add_section(sec)
            for key, value in vars.iteritems():
                self.set(sec, key, value)

        # now get the setting
        self.reload()


    def reload(self):
        """Re-read settings from all configured locations.
        """
        # listof filenames to parse (custom plus some standard ones)
        filenames = self.__cfg_filenames \
                    + ['pymvpa.cfg',
                       os.path.join(os.path.expanduser('~'), '.pymvpa.cfg')]

        # read local and user-specific config
        files = self.read(filenames)

        # no look for variables in the environment 
        for var in [v for v in os.environ.keys() if v.startswith('MVPA_')]:
            if __debug__:
                debug('INIT', "Found env. variable '%s'" % var)

            # strip leading 'MVPA_'
            svar = var[5:]

            # section is next element in name (or 'general' if simple name)
            if not svar.count('_'):
                sec = 'general'
            else:
                cut = svar.find('_')
                sec = svar[:cut]
                svar = svar[cut + 1:].replace('_', ' ')

            # check if section is already known and add it if not
            if not self.has_section(sec):
                self.add_section(sec)

            # set value
            self.set(sec, svar, os.environ[var])


        if __debug__:
            debug('INIT', "Read config files '%s'" % str(files))


    def __repr__(self):
        """Generate INI file content with current configuration.
        """
        # make adaptor to use str as file-like (needed for ConfigParser.write()
        class file2str(object):
            def __init__(self):
                self.__s = ''
            def write(self, val):
                self.__s += val
            def str(self):
                return self.__s

        r = file2str()
        self.write(r)

        return r.str()


    def save(self, filename):
        """Write current configuration to a file.
        """
        f = open(filename, 'w')
        self.write(f)
        f.close()
