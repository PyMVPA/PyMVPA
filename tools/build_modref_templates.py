#!/usr/bin/env python
# Attempt to generate templates for module reference with Sphinx

import os
import re
import mvpa


modref_path = os.path.join('doc', 'modref')
if not os.path.exists(modref_path):
    os.mkdir(modref_path)

# only separating first two levels
rst_section_levels = ['*', '=', '-', '-', '-']

def writeAPIDocTemplate(uri):
    try:
        #exec 'import ' + uri + ' as obj'
        tf = open(os.path.join(modref_path, uri + '.txt'), 'w')

        ad = '.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n'
        title = uri
        ad += title + '\n' + rst_section_levels[uri.count('.')] * len(title)
        ad += '\n\n.. automodule:: ' + uri + '\n'
        # also listing inherited members swamps the index with lots of
        # quasi-duplicates
        #ad += '  :members:\n  :inherited-members:\n  :undoc-members:\n'
        ad += '  :members:\n  :undoc-members:\n' \
              '  :show-inheritance:\n  :noindex:\n'
        tf.write(ad)
        tf.close()
    except:
        print 'Warning: Cannot import', uri


root_path = mvpa.__path__[0]

# compose list of modules
modules = []

# raw directory parsing
for dirpath, dirnames, filenames in os.walk(mvpa.__path__[0]):
    # determine the importable location of the module
    module_uri = re.sub(os.path.sep,
                        '.',
                        re.sub(root_path,
                               'mvpa',
                               dirpath))

    modules.append(module_uri)

    for filename in filenames:
        # XXX maybe check for extenstions as well?
        if not filename.endswith('.py') or filename == '__init__.py':
            continue

        modules.append('.'.join([module_uri, filename[:-3]]))

# write the list
for m in modules:
    writeAPIDocTemplate(m)
