#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to automagically generate ReST versions of examples under doc/"""

__docformat__ = 'restructuredtext'


import os, sys, re, glob

exclude_list = ['searchlight.py']

outpath = os.path.join('build', 'docsrc', 'examples')
if not os.path.exists(outpath):
    os.mkdir(outpath)


def procExample(filename):
    #  doc filename
    dfilename = filename[:-3] + '.rst'

    # open source file
    xfile = open(filename)
    # open dest file
    dfile = open(os.path.join(outpath, os.path.basename(dfilename)), 'w')

    # place header
    dfile.write('.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n')

    # place cross-ref target
    dfile.write('.. _example_' + os.path.basename(filename)[:-3] + ':\n\n')

    inheader = True
    indocs = False
    doc2code = False
    code2doc = False

    for line in xfile:
        # skip header
        if inheader and not line.startswith('"""'):
            continue
        # determine end of header
        if inheader and line.startswith('"""'):
            inheader = False

        # strip comments and remove trailing whitespace
        cleanline = line[:line.find('#')].rstrip()
        # if we have something that should go into the text
        if indocs or cleanline.startswith('"""'):
            proc_line = None
            # handle doc start
            if not indocs:
                # guarenteed to start with """
                if len(cleanline) > 3 and cleanline.endswith('"""'):
                    # single line doc
                    code2doc = True
                    doc2code = True
                    proc_line = cleanline[3:-3]
                else:
                    # must be start of multiline block
                    indocs = True
                    code2doc = True
                    # rescue what is left on the line
                    proc_line = cleanline[3:] # strip """
            else:
                # we are already in the docs
                # handle doc end
                if cleanline.endswith('"""'):
                    indocs = False
                    doc2code = True
                    # rescue what is left on the line
                    proc_line = cleanline[:-3]
                else:
                    # has to be documentation
                    proc_line = line

            if code2doc:
                code2doc = False
                dfile.write('\n')

            if proc_line:
                dfile.write(proc_line.rstrip() + '\n')

        else:
            if doc2code:
                doc2code = False
                dfile.write('\n')

                # if there is nothing on the line, do nothing
                if not line.strip():
                    continue

            # has to be code
            dfile.write('  >>> ' + line)

    # write post example see also box
    dfile.write("\n.. seealso::\n  The full source code of this example is "
                "included in the PyMVPA source distribution (`%s`).\n"
                % filename)

    xfile.close()
    dfile.close()


# for all examples
for f in glob.glob(os.path.join('doc','examples','*.py')):
    if not os.path.basename(f) in exclude_list:
        procExample(f)
