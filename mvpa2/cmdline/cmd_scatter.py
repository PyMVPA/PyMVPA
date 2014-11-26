#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*- 
#ex: set sts=4 ts=4 sw=4 noet:
#------------------------- =+- Python script -+= -------------------------
"""
  Yaroslav Halchenko                                            Dartmouth
  web:     http://www.onerussian.com                              College
  e-mail:  yoh@onerussian.com                              ICQ#: 60653192

 DESCRIPTION (NOTES):

  Simple tool to plot scatter plots of two NIfTI volumes (or PyMVPA datasets)
  with coloring depicting the locations.

 COPYRIGHT: Yaroslav Halchenko 2011-2012

 LICENSE: MIT

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
"""
#-----------------\____________________________________/------------------

__author__ = 'Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2011 Yaroslav Halchenko'
__license__ = 'MIT'

from mvpa2.base import verbose
from mvpa2.misc.cmdline import opts, parser

import sys, os
import pylab as pl
from optparse import OptionParser, Option

from scatter import plot_scatter_files

def get_opt_parser():
    # use module docstring for help output
    ## p = OptionParser(
    ##             usage="%s [OPTIONS] FILE1 FILE2 [FILE...] \n\n" % sys.argv[0] + __doc__,
    ##             version="%prog " + 'TODO') #nib.__version__)

    parser.add_options([
        Option("-t", "--volume", action="store", type="int",
               dest="volume", default=None,
               help="If 4D image given which volume to plot.  If 5D with rudimentary "
               "4th, it gets removed. Default -- all"),

        Option("-m", "--mask-file", action="store",
               dest="mask_file", default=None,
               help="Filename to use as a mask to decide which voxels to plot."),

        Option("--thresholds", action="store",
               dest="thresholds", default=None,
               help="How to threshold the mask volume.  Single value specifies "
               "lower threshold. Two comma-separated values specify exclusion "
               "range: e.g. '-3,3' would include all abs values >=3.  '3,-3' "
               "would then include all abs values < 3"),

        Option("-M", "--masked-opacity", action="store", type='float',
               dest="masked_opacity", default=0.,
               help="Opacity at which plot masked-out points.  Default is 0, i.e."
               " when they are not plotted at all"),

        Option("-u", "--unique-points", action="store_true",
               default=False,
               help="Plot those points which are present only in 1 of the volumes "
               "and not in the other along corresponding axis"),

        Option("-l", "--limits", type="choice",
               choices=['auto', 'same', 'per-axis'],
               default='auto',
               help="How to decide on limits for the axes. When 'auto' -- if data "
               "ranges overlap is more than 50% of the union range, 'same' is "
               "considered."),

        Option("-x", "--x-jitter", action="store", type='float',
               dest="x_jitter", default=0.,
               help="Half-width of the uniform jitter to add to x-coords. Useful"
               " for quantized (thus overlapping) data"),

        Option("-y", "--y-jitter", action="store", type='float',
               dest="y_jitter", default=0.,
               help="Half-width of the uniform jitter to add to y-coords. Useful"
               " for quantized (thus overlapping) data"),

        ])

    parser.usage = "%s [OPTIONS] FILE1 FILE2 [FILE...] \n\n" % sys.argv[0] + __doc__
    parser.version="%prog " + 'TODO'

    parser.option_groups = [opts.common]

    return parser


if __name__ == "__main__":
    """
    TODO:

    - proper cmdline options
    - add specification/use of the mask
    - Michael: fa option which would specify color
    """

    if '_ip' in dir():
        # are we in interactive ipython... lets filter out sys.argv
        sys.argv = sys.argv[:1]

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    #files = [
    #    '/data/famface/subjects/km00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/mni/svmsl3_gross_acc-chance_familiarity_noself_cvidentity.nii.gz',
    #    '/data/famface/subjects/em00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/mni/svmsl3_gross_acc-chance_familiarity_noself_cvidentity.nii.gz'
    ##    '/data/famface/subjects/kl00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/svmsl3_gross_acc-chance_familiarity_noself_cvidentity.nii.gz',
    ##    '/data/famface/subjects/kl00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/gnbsl3_acc-chance_familiarity_noself_cvidentity.nii.gz',
    #    #'/data/famface/subjects/kl00/results/mvpa110714-ev+2_3/gauss_fwhm_4.0/gnbsl3_familiarity_noself_cvidentity_acc-chance.nii.gz'
    #    ]

    ## files = [
    ##     '/data/famface/subjects/sz00/results/mvpa110716-ev+1_3/gauss_fwhm_4.0/mni/similarity:all_sl3_identity_noself_correlation:correlation.nii.gz',
    ##     '/data/famface/subjects/sz00/results/mvpa110716-ev+1_3/gauss_fwhm_4.0/mni/similarity:all5_sl3_identity_noself_correlation:spearmanr.nii.gz']

    files_ = [
        '/data/famface/results/mvpa111229+tempderivs2-ev+1_3-23.good/__best_betas4__sl3___good_bestbad3/t-tests/mni/_familiarity_noself_cvidentity_gross_acc-chance_z.nii.gz',
        '/data/famface/results/mvpa111229+tempderivs2-ev+1_3-23.good/__best_betas4__sl3___good_bestbad3/t-tests/mni/_familiarity_noself_cvidentity_gross_acc-chance_bestm.nii.gz',
        ]
    assert(len(files) >= 2)             # just 2 files
    if opts.thresholds:
        opts.thresholds = [float(x) for x in opts.thresholds.split(',')]
    plot_scatter_files(files,
                       mask_file=opts.mask_file,
                       masked_opacity=opts.masked_opacity,
                       volume=opts.volume,
                       thresholds=opts.thresholds,
                       limits=opts.limits,
                       x_jitter=opts.x_jitter,
                       y_jitter=opts.y_jitter,
                       uniq=opts.unique_points)
    pl.show()
