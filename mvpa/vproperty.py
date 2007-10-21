#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: t -*- 
#ex: set sts=4 ts=4 sw=4 noet:
#------------------------- =+- Python script -+= -------------------------
"""

 COPYRIGHT: XXX

 LICENSE:

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the 
  Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
  MA 02110-1301, USA.

 On Debian system see /usr/share/common-licenses/GPL for the full license.
"""
#-----------------\____________________________________/------------------

class vproperty(object):
	"""
	Provides "virtual" property: uses derived class's method
	"""

    def __init__(self, fget=None, fset=None, fdel=None, doc=''):
        for attr in ('fget', 'fset'):
            func = locals()[attr]
            if callable(func):
                setattr(self, attr, func.func_name)
        setattr(self, '__doc__', doc)

    def __get__(self, obj=None, type=None):
        if not obj:
            return 'property'
        if self.fget:
            return getattr(obj, self.fget)()

    def __set__(self, obj, arg):
        if self.fset:
            return getattr(obj, self.fset)(arg)
