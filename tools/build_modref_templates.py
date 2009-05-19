#!/usr/bin/env python
"""Script to auto-generate our API docs."""

import os

from apigen import ApiDocWriter

if __name__ == '__main__':
    package = 'mvpa'
    outdir = os.path.join('build', 'doc', 'modref')
    docwriter = ApiDocWriter(package, rst_extension='.rst')
    #docwriter.package_skip_patterns += ['\\.fixes$',
    #                                    '\\.externals$']
    docwriter.write_api_docs(outdir)
    #docwriter.write_index(outdir, 'gen', relative_to='api')
    print '%d files written' % len(docwriter.written_modules)
