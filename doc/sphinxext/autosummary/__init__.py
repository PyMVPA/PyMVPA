# -*- coding: utf-8 -*-
"""
    sphinx.ext.autosummary
    ~~~~~~~~~~~~~~~~~~~~~~

    Sphinx extension that adds an autosummary:: directive, which can be
    used to generate function/method/attribute/etc. summary lists, similar
    to those output eg. by Epydoc and other API doc generation tools.

    An :autolink: role is also provided.

    autosummary directive
    ---------------------

    The autosummary directive has the form::

        .. autosummary::
           :nosignatures:
           :toctree: generated/

           module.function_1
           module.function_2
           ...

    and it generates an output table (containing signatures, optionally)

        ========================  =============================================
        module.function_1(args)   Summary line from the docstring of function_1
        module.function_2(args)   Summary line from the docstring
        ...
        ========================  =============================================

    If the :toctree: option is specified, files matching the function names
    are inserted to the toctree with the given prefix:

        generated/module.function_1
        generated/module.function_2
        ...

    Note: The file names contain the module:: or currentmodule:: prefixes.

    .. seealso:: autosummary_generate.py


    autolink role
    -------------

    The autolink role functions as ``:obj:`` when the name referred can be
    resolved to a Python object, and otherwise it becomes simple emphasis.
    This can be used as the default role to make links 'smart'.

    :copyright: Copyright 2007-2009 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import os
import re
import sys
import inspect
import posixpath
from os import path

from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from docutils import nodes

from sphinx import addnodes, roles
from sphinx.util import patfilter
from sphinx.util.compat import Directive


# -- autosummary_toc node ------------------------------------------------------

class autosummary_toc(nodes.comment):
    pass

def process_autosummary_toc(app, doctree):
    """
    Insert items described in autosummary:: to the TOC tree, but do
    not generate the toctree:: list.
    """
    env = app.builder.env
    crawled = {}
    def crawl_toc(node, depth=1):
        crawled[node] = True
        for j, subnode in enumerate(node):
            try:
                if (isinstance(subnode, autosummary_toc)
                    and isinstance(subnode[0], addnodes.toctree)):
                    env.note_toctree(env.docname, subnode[0])
                    continue
            except IndexError:
                continue
            if not isinstance(subnode, nodes.section):
                continue
            if subnode not in crawled:
                crawl_toc(subnode, depth+1)
    crawl_toc(doctree)

def autosummary_toc_visit_html(self, node):
    """Hide autosummary toctree list in HTML output."""
    raise nodes.SkipNode

def autosummary_noop(self, node):
    pass


# -- autosummary_table node ----------------------------------------------------

class autosummary_table(nodes.comment):
    pass

def autosummary_table_visit_html(self, node):
    """Make the first column of the table non-breaking."""
    try:
        tbody = node[0][0][-1]
        for row in tbody:
            col1_entry = row[0]
            par = col1_entry[0]
            for j, subnode in enumerate(list(par)):
                if isinstance(subnode, nodes.Text):
                    new_text = unicode(subnode.astext())
                    new_text = new_text.replace(u" ", u"\u00a0")
                    par[j] = nodes.Text(new_text)
    except IndexError:
        pass


# -- autodoc integration -------------------------------------------------------

try:
    ismemberdescriptor = inspect.ismemberdescriptor
    isgetsetdescriptor = inspect.isgetsetdescriptor
except AttributeError:
    def ismemberdescriptor(obj):
        return False
    isgetsetdescriptor = ismemberdescriptor

def get_documenter(obj):
    """
    Get an autodoc.Documenter class suitable for documenting the given object
    """
    import sphinx.ext.autodoc as autodoc

    if inspect.isclass(obj):
        if issubclass(obj, Exception):
            return autodoc.ExceptionDocumenter
        return autodoc.ClassDocumenter
    elif inspect.ismodule(obj):
        return autodoc.ModuleDocumenter
    elif inspect.ismethod(obj) or inspect.ismethoddescriptor(obj):
        return autodoc.MethodDocumenter
    elif (ismemberdescriptor(obj) or isgetsetdescriptor(obj)
          or inspect.isdatadescriptor(obj)):
        return autodoc.AttributeDocumenter
    elif inspect.isroutine(obj):
        return autodoc.FunctionDocumenter
    else:
        return autodoc.DataDocumenter


# -- .. autosummary:: ----------------------------------------------------------

class Autosummary(Directive):
    """
    Pretty table containing short signatures and summaries of functions etc.

    autosummary also generates a (hidden) toctree:: node.
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True
    option_spec = {
        'toctree': directives.unchanged,
        'nosignatures': directives.flag,
        'template': directives.unchanged,
    }

    def warn(self, msg):
        self.warnings.append(self.state.document.reporter.warning(
            msg, line=self.lineno))

    def run(self):
        self.env = env = self.state.document.settings.env
        self.genopt = {}
        self.warnings = []

        names = [x.strip().split()[0] for x in self.content
                 if x.strip() and re.search(r'^[~a-zA-Z_]', x.strip()[0])]
        items = self.get_items(names)
        nodes = self.get_table(items)

        if 'toctree' in self.options:
            suffix = env.config.source_suffix
            all_docnames = env.found_docs.copy()
            dirname = posixpath.dirname(env.docname)

            tree_prefix = self.options['toctree'].strip()
            docnames = []
            for name, sig, summary, real_name in items:
                docname = posixpath.join(tree_prefix, real_name)
                if docname.endswith(suffix):
                    docname = docname[:-len(suffix)]
                docname = posixpath.normpath(posixpath.join(dirname, docname))
                if docname not in env.found_docs:
                    self.warn('toctree references unknown document %r'
                              % docname)
                docnames.append(docname)

            tocnode = addnodes.toctree()
            tocnode['includefiles'] = docnames
            tocnode['entries'] = [(None, docname) for docname in docnames]
            tocnode['maxdepth'] = -1
            tocnode['glob'] = None

            tocnode = autosummary_toc('', '', tocnode)
            nodes.append(tocnode)

        return self.warnings + nodes

    def get_items(self, names):
        """
        Try to import the given names, and return a list of
        ``[(name, signature, summary_string, real_name), ...]``.
        """
        env = self.state.document.settings.env

        prefixes = ['']
        if env.currmodule:
            prefixes.insert(0, env.currmodule)

        items = []

        max_item_chars = 50

        for name in names:
            display_name = name
            if name.startswith('~'):
                name = name[1:]
                display_name = name.split('.')[-1]

            try:
                obj, real_name = import_by_name(name, prefixes=prefixes)
            except ImportError:
                self.warn('failed to import %s' % name)
                items.append((name, '', '', name))
                continue

            # NB. using real_name here is important, since Documenters
            #     handle module prefixes slightly differently
            documenter = get_documenter(obj)(self, real_name)
            if not documenter.parse_name():
                self.warn('failed to parse name %s' % real_name)
                items.append((display_name, '', '', real_name))
                continue
            if not documenter.import_object():
                self.warn('failed to import object %s' % real_name)
                items.append((display_name, '', '', real_name))
                continue

            # -- Grab the signature

            sig = documenter.format_signature()
            if not sig:
                sig = ''
            else:
                max_chars = max(10, max_item_chars - len(display_name))
                sig = mangle_signature(sig, max_chars=max_chars)
                sig = sig.replace('*', r'\*')

            # -- Grab the summary

            doc = list(documenter.process_doc(documenter.get_doc()))

            while doc and not doc[0].strip():
                doc.pop(0)
            m = re.search(r"^([A-Z][^A-Z]*?\.\s)", " ".join(doc).strip())
            if m:
                summary = m.group(1).strip()
            elif doc:
                summary = doc[0].strip()
            else:
                summary = ''

            items.append((display_name, sig, summary, real_name))

        return items

    def get_table(self, items):
        """
        Generate a proper list of table nodes for autosummary:: directive.

        *items* is a list produced by :meth:`get_items`.
        """
        table_spec = addnodes.tabular_col_spec()
        table_spec['spec'] = 'LL'

        table = autosummary_table('')
        real_table = nodes.table('')
        table.append(real_table)
        group = nodes.tgroup('', cols=2)
        real_table.append(group)
        group.append(nodes.colspec('', colwidth=10))
        group.append(nodes.colspec('', colwidth=90))
        body = nodes.tbody('')
        group.append(body)

        def append_row(*column_texts):
            row = nodes.row('')
            for text in column_texts:
                node = nodes.paragraph('')
                vl = ViewList()
                vl.append(text, '<autosummary>')
                self.state.nested_parse(vl, 0, node)
                try:
                    if isinstance(node[0], nodes.paragraph):
                        node = node[0]
                except IndexError:
                    pass
                row.append(nodes.entry('', node))
            body.append(row)

        for name, sig, summary, real_name in items:
            qualifier = 'obj'
            if 'nosignatures' not in self.options:
                col1 = ':%s:`%s <%s>`\ %s' % (qualifier, name, real_name, sig)
            else:
                col1 = ':%s:`%s <%s>`' % (qualifier, name, real_name)
            col2 = summary
            append_row(col1, col2)

        return [table_spec, table]

def mangle_signature(sig, max_chars=30):
    """Reformat a function signature to a more compact form."""
    sig = re.sub(r"^\((.*)\)$", r"\1", sig) + ", "
    r = re.compile(r"(?P<name>[a-zA-Z0-9_*]+)(?P<default>=.*?)?, ")
    items = r.findall(sig)

    args = [name for name, default in items if not default]
    opts = [name for name, default in items if default]

    sig = limited_join(", ", args, max_chars=max_chars-2)
    if opts:
        if not sig:
            sig = "[%s]" % limited_join(", ", opts, max_chars=max_chars-4)
        elif len(sig) < max_chars - 4 - 2 - 3:
            sig += "[, %s]" % limited_join(", ", opts,
                                           max_chars=max_chars-len(sig)-4-2)

    return u"(%s)" % sig

def limited_join(sep, items, max_chars=30, overflow_marker="..."):
    """
    Join a number of strings to one, limiting the length to *max_chars*.

    If the string overflows this limit, replace the last fitting item by
    *overflow_marker*.

    Returns: joined_string
    """
    full_str = sep.join(items)
    if len(full_str) < max_chars:
        return full_str

    n_chars = 0
    n_items = 0
    for j, item in enumerate(items):
        n_chars += len(item) + len(sep)
        if n_chars < max_chars - len(overflow_marker):
            n_items += 1
        else:
            break

    return sep.join(list(items[:n_items]) + [overflow_marker])

# -- Importing items -----------------------------------------------------------

def import_by_name(name, prefixes=[None]):
    """
    Import a Python object that has the given *name*, under one of the
    *prefixes*.  The first name that succeeds is used.
    """
    tried = []
    for prefix in prefixes:
        try:
            if prefix:
                prefixed_name = '.'.join([prefix, name])
            else:
                prefixed_name = name
            return _import_by_name(prefixed_name), prefixed_name
        except ImportError:
            tried.append(prefixed_name)
    raise ImportError('no module named %s' % ' or '.join(tried))

def _import_by_name(name):
    """Import a Python object given its full name."""
    try:
        name_parts = name.split('.')

        # try first interpret `name` as MODNAME.OBJ
        modname = '.'.join(name_parts[:-1])
        if modname:
            try:
                __import__(modname)
                return getattr(sys.modules[modname], name_parts[-1])
            except (ImportError, IndexError, AttributeError):
                pass

        # ... then as MODNAME, MODNAME.OBJ1, MODNAME.OBJ1.OBJ2, ...
        last_j = 0
        modname = None
        for j in reversed(range(1, len(name_parts)+1)):
            last_j = j
            modname = '.'.join(name_parts[:j])
            try:
                __import__(modname)
            except ImportError:
                continue
            if modname in sys.modules:
                break

        if last_j < len(name_parts):
            obj = sys.modules[modname]
            for obj_name in name_parts[last_j:]:
                obj = getattr(obj, obj_name)
            return obj
        else:
            return sys.modules[modname]
    except (ValueError, ImportError, AttributeError, KeyError), e:
        raise ImportError(*e.args)


# -- :autolink: (smart default role) -------------------------------------------

def autolink_role(typ, rawtext, etext, lineno, inliner,
                  options={}, content=[]):
    """
    Smart linking role.

    Expands to ':obj:`text`' if `text` is an object that can be imported;
    otherwise expands to '*text*'.
    """
    r = roles.xfileref_role('obj', rawtext, etext, lineno, inliner,
                            options, content)
    pnode = r[0][0]

    prefixes = [None]
    #prefixes.insert(0, inliner.document.settings.env.currmodule)
    try:
        obj, name = import_by_name(pnode['reftarget'], prefixes)
    except ImportError:
        content = pnode[0]
        r[0][0] = nodes.emphasis(rawtext, content[0].astext(),
                                 classes=content['classes'])
    return r


def process_generate_options(app):
    genfiles = app.config.autosummary_generate

    ext = app.config.source_suffix

    if genfiles and not hasattr(genfiles, '__len__'):
        env = app.builder.env
        genfiles = [x + ext for x in env.found_docs
                    if os.path.isfile(env.doc2path(x))]

    if not genfiles:
        return

    from sphinx.ext.autosummary.generate import generate_autosummary_docs

    genfiles = [genfile + (not genfile.endswith(ext) and ext or '')
                for genfile in genfiles]

    generate_autosummary_docs(genfiles, builder=app.builder,
                              warn=app.warn, info=app.info, suffix=ext,
                              base_path=app.srcdir)


def setup(app):
    # I need autodoc
    app.setup_extension('sphinx.ext.autodoc')
    app.add_node(autosummary_toc,
                 html=(autosummary_toc_visit_html, autosummary_noop),
                 latex=(autosummary_noop, autosummary_noop),
                 text=(autosummary_noop, autosummary_noop))
    app.add_node(autosummary_table,
                 html=(autosummary_table_visit_html, autosummary_noop),
                 latex=(autosummary_noop, autosummary_noop),
                 text=(autosummary_noop, autosummary_noop))
    app.add_directive('autosummary', Autosummary)
    app.add_role('autolink', autolink_role)
    app.connect('doctree-read', process_autosummary_toc)
    app.connect('builder-inited', process_generate_options)
    app.add_config_value('autosummary_generate', [], True)
