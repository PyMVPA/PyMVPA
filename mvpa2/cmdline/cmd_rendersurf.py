# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Render statistical maps on a Freesurfer surfer for visualization.

This command is mostly useful for generating figure components for later
assembly in, for example, inkscape. It is very similar to the ``pysurfer``
tool (and internally uses the pysurfer package), but offers a slightly
different API that may some things a bit easier. Most importantly, it can
render multiple views in a single run. This command does not support
interactive visualization, but does require a running X server. Moreover,
a functional Freesurfer installation is required.

Usage tip

A common need is to visualize statistical maps in MNI space on a surface.
Simply run the 1mm MNI152 template through Freesurfer's `recon-all`. Point
the SUBJECTS_DIR variable to the directory where Freesurfer placed the
output folder. Now run something like the following with any image that
is aligned to (but not necessarily resliced to) MNI152 space::

  SUBJECTS_DIR=~/freesurfer_tmplsubjs pymvpa2 rendersurf \
        -i mystats.nii.gz -s mni152 -o mysurfacerendering \
        --hemi lh --views lat med -t 2.3 --show-locus green -66 -28 8

This will yield the two requested views as PNG files in the current directory.
Use the --size option to generate high-resolution rendering for publications.
Be aware that large resolutions will take forever to render.
"""

# magic line for manpage summary
# man: -*- % export dataset components into other (file) formats

__docformat__ = 'restructuredtext'

import sys
import argparse
from mvpa2.base import verbose, warning, error

if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import arg2ds, parser_add_common_opt, hdf5compression

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}


def setup_parser(parser):
    parser.add_argument(
        '-s', '--subject', required=True,
        help="""""")
    parser.add_argument(
        '-i', '--overlay-volume', required=True,
        help="""""")
    parser.add_argument(
        '-o', '--output-prefix', required=True,
        help="""""")
    parser.add_argument(
        '--surface', default='inflated',
        choices=('pial', 'white', 'inflated'),
        help="""""")
    parser.add_argument(
        '--hemifield', default=('lh', 'rh'), nargs='+',
        help="""""")
    parser.add_argument(
        '--size', default='400x400',
        help="""""")
    parser.add_argument(
        '--views', default=['lat', 'med', 'front', 'pari', 'vent', 'dors'],
        nargs='+',
        help="""""")
    parser.add_argument(
        '--imgtype', default='png',
        help="""""")
    parser.add_argument(
        '-r', '--overlay-range',
        nargs=2, default=('min', 'max'), type=float,
        help="""""")
    parser.add_argument(
        '-c', '--overlay-colormap', default='afmhot',
        help="""""")
    parser.add_argument(
        '-t', '--overlay-thresh', default=None, type=float,
        help="""""")
    parser.add_argument(
        '-a', '--overlay-alpha', default=1.0, type=float,
        help="""""")
    parser.add_argument(
        '--background-color', default='black',
        help="""""")
    parser.add_argument(
        '--surf-fwhm', default=3.0, type=float,
        help="""""")
    parser.add_argument(
        '--projsum', default='avg', choices=('avg', 'max'),
        help="""""")
    parser.add_argument(
        '--show-locus', nargs=4, action='append',
        help="""COLOR X Y Z in mm""")


def run(args):
    invol = args.overlay_volume
    colmap = args.overlay_colormap
    omin, omax = args.overlay_range
    if omin == 'min' or omax == 'max':
        import nibabel
        data = nibabel.load(invol).get_data()
        if omin == 'min':
            omin = data.min()
        if omax == 'max':
            omax = data.max()
    img_xsize, img_ysize = args.size.lower().split('x')
    img_size = (int(img_xsize), int(img_ysize))

    from surfer import Brain, io
    for hemi in args.hemifield:
        render = Brain(
            args.subject,
            hemi,
            args.surface,
            offscreen=True,
            background=args.background_color,
            size=img_size)
        surf_data = io.project_volume_data(
            invol,
            hemi,
            subject_id=args.subject,
            projsum=args.projsum,
            smooth_fwhm=args.surf_fwhm)
        render.add_data(
            surf_data,
            min=omin, max=omax,
            thresh=args.overlay_thresh,
            colormap=colmap,
            alpha=args.overlay_alpha,
            hemi=hemi)
        if args.show_locus is not None:
            for locus in args.show_locus:
                x, y, z = [float(i) for i in locus[1:]]
                render.add_foci(
                    [[x, y, z]],
                    map_surface=args.surface,
                    color=locus[0])
        render.save_imageset(
            '%s_%s' % (args.output_prefix, hemi),
            args.views,
            args.imgtype)

