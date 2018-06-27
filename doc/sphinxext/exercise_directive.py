# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Sphinx extension add a 'exercise' directive similar to 'seealso' and 'note'.

This directive can be used to add exercise boxes to tutorials.
"""

__docformat__ = 'restructuredtext'

from sphinx import addnodes
try:
    from sphinx.util.compat import Directive
except ImportError:
    from docutils.parsers.rst import Directive
# DeprecationWarning: make_admonition is deprecated, use docutils.parsers.rst.directives.admonitions.BaseAdmonition instead
try:
    from sphinx.util.compat import make_admonition
except ImportError:
    from docutils.parsers.rst.directives.admonitions \
        import BaseAdmonition
    make_admonition = None

from docutils import nodes

class excercise_node(nodes.Admonition, nodes.Element):
    pass

def visit_exercise_node(self, node):
    self.visit_admonition(node)

def depart_exercise_node(self, node):
    self.depart_admonition(node)


if make_admonition:
    class BaseExcerciseDirective(Directive):
        def run(self):
            return make_admonition(
                excercise_node, self.name, ['Exercise'], self.options,
                self.content, self.lineno, self.content_offset, self.block_text,
                self.state, self.state_machine)
else:
    class BaseExcerciseDirective(BaseAdmonition):
        def run(self):
            return super(BaseExcerciseDirective, self).run()


class ExcerciseDirective(BaseExcerciseDirective):
    """
    An admonition mentioning an exercise to perform (e.g. in a tutorial).
    """

    node_class = excercise_node
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {}

    def run(self):
        ret = super(ExcerciseDirective, self).run()
        if self.arguments:
            argnodes, msgs = self.state.inline_text(self.arguments[0],
                                                    self.lineno)
            para = nodes.paragraph()
            para += argnodes
            para += msgs
            ret[0].insert(1, para)
        return ret


def setup(app):
    app.add_node(excercise_node,
                 html=(visit_exercise_node, depart_exercise_node),
                 latex=(visit_exercise_node, depart_exercise_node),
                 text=(visit_exercise_node, depart_exercise_node))
    app.add_directive('exercise', ExcerciseDirective)
