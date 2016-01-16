# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Creating simple PDF reports using reportlab
"""

__docformat__ = 'restructuredtext'


import os
from os.path import join as pathjoin
from datetime import datetime

import mvpa2
from mvpa2.base import externals, verbose
from mvpa2.base.dochelpers import borrowkwargs

if __debug__:
    from mvpa2.base import debug

if externals.exists('reportlab', raise_=True):
    import reportlab as rl
    import reportlab.platypus as rplp
    import reportlab.lib.styles as rpls
    import reportlab.lib.units as rplu

    # Actually current reportlab's Image can't deal directly with .pdf images
    # Lets use png for now
    if externals.versions['reportlab'] >= '1112.2':
        _fig_ext_default = 'pdf'
    else:
        _fig_ext_default = 'png'


__all__ = [ 'rl', 'Report', 'escape_xml' ]


def escape_xml(s):
    """Replaces &<> symbols with corresponding XML tokens
    """
    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    return s

class Report(object):
    """Simple PDF reports using reportlab

    Named report 'report' generates 'report.pdf' and directory 'report/' with
    images which were requested to be included in the report

    You can attach report to the existing 'verbose' with

    >>> report = Report()
    >>> verbose.handlers += [report]

    and then all verbose messages present on the screen will also be recorded
    in the report.  Use

    >>> report.text("string")          #  to add arbitrary text
    >>> report.xml("<H1>skajdsf</H1>") # to add XML snippet

    or

    >>> report.figure()  # to add the current figure to the report.
    >>> report.figures() # to add existing figures to the report

    Note that in the later usecase, figures might not be properly
    interleaved with verbose messages if there were any between the
    creations of the figures.

    Inspired by Andy Connolly
    """

    def __init__(self, name='report', title=None, path=None,
                 author=None, style="Normal",
                 fig_ext=None, font='Helvetica',
                 pagesize=None):
        """Initialize report

        Parameters
        ----------
        name : string
          Name of the report
        title : string or None
          Title to start the report, if None, name will be used
        path : string
          Top directory where named report will be stored. Has to be
          set now to have correct path for storing image renderings.
          Default: current directory
        author : string or None
          Optional author identity to be printed
        style : string
          Default Paragraph to be used. Must be the one of the known
          to reportlab styles, e.g. Normal
        fig_ext : string
          What extension to use for figures by default. If None, a default
          will be used. Since versions prior 2.2 of reportlab might do not
          support pdf, 'png' is default for those, 'pdf' otherwise
        font : string
          Name of the font to use
        pagesize : tuple of floats
          Optional page size if not to be default
        """

        if pagesize is None:
            pagesize = rl.rl_config.defaultPageSize
        self.pagesize = pagesize

        self.name = name
        self.author = author
        self.font = font
        self.title = title
        if fig_ext is None:
            self.fig_ext = _fig_ext_default
        else:
            self.fig_ext = fig_ext

        if path is None:
            self._filename = name
        else:
            self._filename = pathjoin(path, name)

        self.__nfigures = 0

        try:
            styles = rpls.getSampleStyleSheet()
            self.style = styles.byName[style]
        except KeyError:
            raise ValueError, \
                  "Style %s is not know to reportlab. Known are %s" \
                  % (styles.keys())

        self._story = []


    @property
    def __preamble(self):
        """Compose the beginning of the report
        """
        date = datetime.today().isoformat(' ')

        owner = 'PyMVPA v. %s' % mvpa2.__version__
        if self.author is not None:
            owner += '   Author: %s' % self.author

        return [ rplp.Spacer(1, 0.8*rplu.inch),
                 rplp.Paragraph("Generated on " + date, self.style),
                 rplp.Paragraph(owner, self.style)] + self.__flowbreak


    def clear(self):
        """Clear the report
        """
        self._story = []


    def xml(self, line, style=None):
        """Adding XML string to the report
        """
        if __debug__ and not self in debug.handlers:
            debug("REP", "Adding xml '%s'" % line.strip())
        if style is None:
            style = self.style
        self._story.append(rplp.Paragraph(line, style=style))

    # Can't use here since Report isn't yet defined at this point
    #@borrowkwargs(Report, 'xml')
    def text(self, line, **kwargs):
        """Add a text string to the report
        """
        if __debug__ and not self in debug.handlers:
            debug("REP_", "Adding text '%s'" % line.strip())
        # we need to convert some of the characters to make it
        # legal XML
        line = escape_xml(line)
        self.xml(line, **kwargs)

    write = text
    """Just an alias for .text, so we could simply provide report
    as a handler for verbose
    """

    # can't do here once again since it needs to conditional on externals
    # TODO: workaround -- either passing symbolic names or assign
    #       post-class creation
    #@borrowkwargs(reportlab.platypus.Image, '__init__')
    def figure(self, fig=None, name=None, savefig_kwargs=None, **kwargs):
        """Add a figure to the report

        Parameters
        ----------
        fig : None or str or `figure.Figure`
          Figure to place into report: `str` is treated as a filename,
          `Figure` stores it into a file under directory and embeds
          into the report, and `None` takes the current figure
        savefig_kwargs : dict
          Additional keyword arguments to provide savefig with (e.g. dpi)
        **kwargs
          Passed to :class:`reportlab.platypus.Image` constructor
        """
        if savefig_kwargs is None:
            savefig_kwargs = {}

        if externals.exists('pylab', raise_=True):
            import pylab as pl
            figure = pl.matplotlib.figure

        if fig is None:
            fig = pl.gcf()

        if isinstance(fig, figure.Figure):
            # Create directory if needed
            if not (os.path.exists(self._filename) and
                    os.path.isdir(self._filename)):
                os.makedirs(self._filename)

            # Figure out the name for image
            self.__nfigures += 1
            if name is None:
                name = 'Figure#'
            name = name.replace('#', str(self.__nfigures))

            # Save image
            fig_filename = pathjoin(self._filename,
                                        '%s.%s' % (name, self.fig_ext))
            if __debug__ and not self in debug.handlers:
                debug("REP_", "Saving figure '%s' into %s"
                      % (fig, fig_filename))

            fig.savefig(fig_filename, **savefig_kwargs)

            # adjust fig to the one to be included
            fig = fig_filename

        if __debug__ and not self in debug.handlers:
            debug("REP", "Adding figure '%s'" % fig)

        im = rplp.Image(fig, **kwargs)

        # If the inherent or provided width/height are too large -- shrink down
        imsize = (im.drawWidth, im.drawHeight)

        # Reduce the size if necessary so reportlab does not puke later on
        r = [float(d)/m for d,m in zip(imsize, self.pagesize)]
        maxr = max(r)
        if maxr > 1.0:
            if __debug__ and not self in debug.handlers:
                debug("REP_", "Shrinking figure by %.3g" % maxr)
            im.drawWidth  /= maxr
            im.drawHeight /= maxr

        self._story.append(im)


    def figures(self, *args, **kwargs):
        """Adds all present figures at once

        If called twice, it might add the same figure multiple times,
        so make sure to close all previous figures if you use
        figures() multiple times
        """
        if externals.exists('pylab', raise_=True):
            import pylab as pl
        figs = pl.matplotlib._pylab_helpers.Gcf.figs
        if __debug__ and not self in debug.handlers:
            debug('REP', "Saving all %d present figures" % len(figs))
        for fid, f in figs.iteritems():
            self.figure(f.canvas.figure, *args, **kwargs)

    @property
    def __flowbreak(self):
        return [rplp.Spacer(1, 0.2*rplu.inch),
                rplp.Paragraph("-" * 150, self.style),
                rplp.Spacer(1, 0.2*rplu.inch)]

    def flowbreak(self):
        """Just a marker for the break of the flow
        """
        if __debug__ and not self in debug.handlers:
            debug("REP", "Adding flowbreak")

        self._story.append(self.__flowbreak)


##     def __del__(self):
##         """Store report upon deletion
##         """
##         if __debug__ and not self in debug.handlers:
##             debug("REP", "Report is being deleted. Storing")
##         self.save()


    def save(self, add_preamble=True):
        """Saves PDF

        Parameters
        ----------
        add_preamble : bool
          Either to add preamble containing title/date/author information
        """

        if self.title is None:
            title = self.name + " report"
        else:
            title = self.title

        pageinfo = self.name + " data"

        ##REF: Name was automagically refactored
        def my_first_page(canvas, doc):
            canvas.saveState()
            canvas.setFont(self.font, 16)
            canvas.drawCentredString(self.pagesize[0]/2.0,
                                     self.pagesize[1]-108, title)
            canvas.setFont(self.font, 9)
            canvas.drawString(rplu.inch, 0.75 * rplu.inch,
                              "First Page / %s" % pageinfo)
            canvas.restoreState()

        ##REF: Name was automagically refactored
        def my_later_pages(canvas, doc):
            canvas.saveState()
            canvas.setFont(self.font, 9)
            canvas.drawString(rplu.inch, 0.75 * rplu.inch,
                              "Page %d %s" % (doc.page, pageinfo))
            canvas.restoreState()

        filename = self._filename + ".pdf"
        doc = rplp.SimpleDocTemplate(filename)

        story = self._story
        if add_preamble:
            story = self.__preamble + story

        if __debug__ and not self in debug.handlers:
            debug("REP", "Saving the report into %s" % filename)

        doc.build(story,
                  onFirstPage=my_first_page,
                  onLaterPages=my_later_pages)

