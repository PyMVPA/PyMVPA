#!/usr/bin/python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""A little helper to convert docstrings from epydoc/rest to match
numpy convention
"""

import sys, re

fname = sys.argv[1]
mappings = {':Parameters?:': 'Parameters',
            ':Examples?:': 'Examples',
            ':Raises:': 'Raises',
            '.. note::': 'Notes',
            '.. seealso::': 'See Also',
            ':Returns:': 'Returns'}

alltext = ''.join(open(fname).readlines())
counts = {}
for mn, mt in mappings.iteritems():
    reparam = re.compile('(?P<spaces>\n *)(?P<header>'
                         + mn
                         + ')(?P<body>\n.*?(?:\n\s*\n|"""))',
                         flags=re.DOTALL)
    count = 0
    #for i in [1,2]:#
    while True:
        res = reparam.search(alltext)
        if not res:
            break
        #print ">", res.group(), "<"
        resd = res.groupdict()
        s, e = res.start(), res.end()

        # Lets adjust alltext
        body = resd['body']
        for i in xrange(2):
            body = body.replace(resd['spaces'] + ' ', resd['spaces'])
        # if any 4-spaces survived
        # body = body.replace(resd['spaces'] + '    ', resd['spaces'] + '  ')
        body = body.replace('``', '`')
        body = re.sub(' *\| *', ' or ', body)
        body = body.replace('basestring', 'str')
        body = body.replace('basestr', 'str')
        adjusted = resd['spaces'] + mt + resd['spaces'] \
                   + '-'*len(mt) + body
        #remove Cheap initialization.
        #print "Converted >%s< to >%s<" % (res.group(), adjusted)
        alltext = alltext[:s] + adjusted + alltext[e:]
        count += 1
    counts[mn] = count

print "File %s: %s" % (
    fname, ', '.join(['%s: %i' % i for i in counts.items() if i[1]]))
file(fname, 'w').write(alltext)
