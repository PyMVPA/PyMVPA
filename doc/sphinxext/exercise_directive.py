# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Sphinx extension add a 'exercise' directive similar to 'seealso' and 'note'.

This directive can be used to add exercize boxes to tutorials.
"""

__docformat__ = 'restructuredtext'

from sphinx import addnodes
from sphinx.util.compat import Directive, make_admonition

from docutils import nodes

class exercise(nodes.Admonition, nodes.Element):
    pass

def visit_exercise_node(self, node):
    self.visit_admonition(node)

def depart_exercise_node(self, node):
    self.depart_admonition(node)


class TaskDirective(Directive):
    """
    An admonition mentioning a exercise to perform (e.g. in a tutorial).
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {}

    def run(self):
        ret = make_admonition(
            exercise, self.name, ['Exercise'], self.options,
            self.content, self.lineno, self.content_offset, self.block_text,
            self.state, self.state_machine)
        if self.arguments:
            argnodes, msgs = self.state.inline_text(self.arguments[0],
                                                    self.lineno)
            para = nodes.paragraph()
            para += argnodes
            para += msgs
            ret[0].insert(1, para)
        return ret


def setup(app):
    app.add_node(exercise,
                 html=(visit_exercise_node, depart_exercise_node),
                 latex=(visit_exercise_node, depart_exercise_node),
                 text=(visit_exercise_node, depart_exercise_node))
    app.add_directive('exercise', TaskDirective)
