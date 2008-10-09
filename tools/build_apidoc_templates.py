# Attempt to generate templates for API docs with Sphinx
# Do not work so far (missing externals cause sphinx to fail


import os
import re
import mvpa


def writeAPIDocTemplate(uri, toctree):
    try:
        #exec 'import ' + uri + ' as obj'
        tf = open(uri + '.txt', 'w')
        tf.write('Title\n*****\n\n.. automodule:: ')
        tf.write(uri)
        tf.write('\n')
        tf.write('  :members:\n')
        tf.close()

        toctree.write('  ' + uri + '\n')
    except:
        print 'Warning: Cannot import', uri


root_path = mvpa.__path__[0]


toctree = open('api.txt', 'w')
toctree.write('API\n***\n\n.. toctree::\n\n')

# raw directory parsing
for dirpath, dirnames, filenames in os.walk(mvpa.__path__[0]):
    # determine the importable location of the module
    module_uri = re.sub(os.path.sep,
                        '.',
                        re.sub(root_path,
                               'mvpa',
                               dirpath))

    writeAPIDocTemplate(module_uri, toctree)

    for filename in filenames:
        # XXX maybe check for extenstions as well?
        if not filename.endswith('.py') or filename == '__init__.py':
            continue

        writeAPIDocTemplate('.'.join([module_uri, filename[:-3]]),
                            toctree)


toctree.close()
