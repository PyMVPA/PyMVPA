#!/usr/bin/python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Query atlases for anatomical labels of voxel coordinates, or their statistics

Examples:

> pymvpa2 atlaslabeler -s -A talairach-dist -d 10 -R Closest\ Gray -l Structure,Brodmann\ area  -cC mask.nii.gz

produces a summary per each structure and brodmann area, for each voxel
looking within 10mm radius for the closest gray matter voxel.

Simpler, more reliable, and faster usage is by providing a
corresponding atlas volume image registered to the volume at hands,
e.g.:

> pymvpa2 atlaslabeler -f MNI-prob-bold.nii.gz -A MNI -s mask_vt.nii.gz
> pymvpa2 atlaslabeler -f HarvardOxford-cort-prob-bold.nii.gz -A HarvardOxford-Cortical -s mask_vt.nii.gz

would provide summary over the MNI (or HarvardOxford-Cortical) atlas,
.nii.gz of which were previously flirted (or fnirted) into the space of
mask_vt.nii.gz and provided in '-f' argument.
"""
# magic line for manpage summary
# man: -*- % query stereotaxic atlases

__docformat__ = 'restructuredtext'

import re, sys, os
import argparse

import mvpa2
from mvpa2.base import verbose, warning, externals

if externals.exists('nibabel', raise_=True):
    import nibabel as nb

if __debug__:
    from mvpa2.base import debug

from mvpa2.atlases.transformation import *

if externals.exists('lxml', raise_=True):
    from mvpa2.atlases import Atlas, ReferencesAtlas, FSLProbabilisticAtlas, \
         KNOWN_ATLASES, KNOWN_ATLAS_FAMILIES, XMLAtlasException

import numpy as np
#import numpy.linalg as la
# to read in transformation matrix


def select_from_volume_iterator(volFileName, lt=None, ut=None):
    """
    Generator which returns value + coordinates with values of non-0 entries
    from the `volFileName`

    Returns
    -------
    tuple with 0th entry value, the others are voxel coordinates
    More effective than previous loopy iteration since uses numpy's where
    function, but for now is limited only to non-0 voxels selection
    """
    try:
        volFile = nb.load(volFileName)
    except:
        raise IOError("Cannot open image file %s" % volFileName)

    volData = volFile.get_data()
    voxdim = volFile.get_header().get_zooms()[:3]
    if lt is None and ut is None:
        mask = volData != 0.0
    elif lt is None and ut is not None:
        mask = volData <= ut
    elif lt is not None and ut is None:
        mask = volData >= lt
    else:
        mask = np.logical_and(volData >= lt, volData <= ut)

    matchingVoxels = np.where(mask)
    for e in zip(volData[matchingVoxels], *matchingVoxels):
        e_ = tuple(e)
        if len(e_) < 5:
            e_ = e_ + (0,) # add time=0
        yield  e_


def parsed_coordinates_iterator(
    parseString="^\s*(?P<x>\S+)[ \t,](?P<y>\S+)[ \t,](?P<z>\S+)\s*$",
    inputStream=sys.stdin,
    ctype=float,
    dtype=float):
    """Iterator to provide coordinates/values parsed from the string stream,
    most often from the stdin
    """
    parser = re.compile(parseString)
    for line in inputStream.readlines():
        line = line.strip()
        match = parser.match(line)
        if not match:
            if __debug__:
                debug('ATL', "Line '%s' did not match '%s'"
                      % (line, parseString))
        else:
            r = match.groupdict()
            if r.has_key('v'): v = dtype(r['v'])
            else:              v = 0.0
            if r.has_key('t'): t = dtype(r['t'])
            else:              t = 0.0
            yield (v, ctype(r['x']), ctype(r['y']), ctype(r['z']), t)


# XXX helper to process labels... move me
##REF: Name was automagically refactored
def present_labels(args, labels):
    if isinstance(labels, list):
        res = []
        for label in labels:
            # XXX warning -- some inconsistencies in atlas.py
            #     need refactoring
            s = label['label'] #.text
            if label.has_key('prob') and not args.createSummary:
                s += "(%d%%%%)" % label['prob']
            res += [s]
        if res == []:
            res = ['None']
        return '/'.join(res)
    else:
        if args.abbreviatedLabels:
            return labels['label'].abbr
        else:
            return labels['label'].text

def statistics(values):
    N_ = len(values)
    if N_==0:
        return 0, None, None, None, None, ""
    mean = np.mean(values)
    std = np.std(values)
    minv = np.min(values)
    maxv = np.max(values)
    ssummary = "[%3.2f : %3.2f] %3.2f+-%3.2f" % (minv, maxv, mean, std)
    return N_, mean, std, minv, maxv, ssummary


##REF: Name was automagically refactored
def get_summary(args, summary, output):
    """Output the summary
    """
    # Sort either by the name (then ascending) or by the number of
    # elements (then descending)
    sort_keys = [(k, len(v['values']), v['maxcoord'][1])
                 for k,v in summary.iteritems()]
    sort_index, sort_reverse = {
        'name' : (0, False),
        'count': (1, True),
        'a-p': (2, True)}[args.sortSummaryBy]
    sort_keys.sort(cmp=lambda x,y: cmp(x[sort_index], y[sort_index]),
                   reverse=sort_reverse)
    # and here are the keys
    keys = [x[0] for x in sort_keys]
    maxkeylength = max (map(len, keys))

    # may be I should have simply made a counter ;-)
    total = sum(map(lambda x:len(x['values']), summary.values()))
    count_reported = 0
    for index in keys:
        if index.rstrip(' /') == 'None' and args.suppressNone:
            continue
        summary_ = summary[index]
        values = summary_['values']
        N, mean, std, minv, maxv, ssummary = statistics(values)
        Npercent = 100.0*N/total
        if N < args.countThreshold \
               or Npercent < args.countPercentThreshold:
            continue
        count_reported += N
        msg = "%%%ds:" % maxkeylength
        output.write(msg % index)
        output.write("%4d/%4.1f%% items" \
                     % (N, Npercent))

        if args.createSummary>1:
            output.write(" %s" % ssummary)

        if args.createSummary>2:
            output.write(" max at %s" % summary_['maxcoord'])
            if args.showOriginalCoordinates and volQForm:
                #import pydb
                #pydb.debugger()
                #coord = np.dot(volQForm, summary_['maxcoord']+[1])[:3]
                coord = volQForm[summary_['maxcoord']]
                output.write(" %r" % (tuple(coord),))

        if args.createSummary>3 and summary_.has_key('distances'):
            # if we got statistics over referenced voxels
            Nr, mean, std, minv, maxv, ssummary = \
                statistics(summary_['distances'])
            Nr = len(summary_['distances'])
            # print "N=", N, " Nr=", Nr
            output.write(" Referenced: %d/%d%% Distances: %s" \
                         % (Nr, int(Nr*100.0 / N), \
                            ssummary))
        output.write("\n")
        # output might fail to flush, like in the case with broken pipe
        # -- imho that is not a big deal, ie not worth scaring the user
        try:
            output.flush()
        except IOError:
            pass
    output.write("-----\n")
    output.write("TOTAL: %d items" % count_reported)
    if total != count_reported:
        output.write(" (out of %i, %i were excluded)" % (total, total-count_reported))
    output.write("\n")


parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

def setup_parser(parser):
    parser.add_argument("-a", "--atlas-file",
                      action="store", dest="atlasFile",
                      default=None,
                      help="Atlas file to use. Overrides --atlas-path and --atlas")

    parser.add_argument("--atlas-path",
                      action="store", dest="atlasPath",
                      default=None,
                      help=r"Path to the atlas files. '%(name)s' will be replaced"
                           " with the atlas name. See -A. Defaults depend on the"
                           " atlas family.")

    parser.add_argument("-A", "--atlas",
                      action="store", dest="atlasName",
                      default="talairach", choices=KNOWN_ATLASES.keys(),
                      help="Atlas to use. Choices: %s"
                           % ', '.join(KNOWN_ATLASES.keys()))

    parser.add_argument("-f", "--atlas-image-file",
                      action="store", dest="atlasImageFile",
                      default=None,
                      help=r"Path to the data image for the corresponding atlas. "
                           " Can be used to override default image if it was "
                           " already resliced into a corresponding space (e.g."
                           " subject)")

    parser.add_argument("-i", "--input-coordinates-file",
                      action="store", dest="inputCoordFile",
                      default=None,
                      help="Fetch coordinates from ASCII file")

    parser.add_argument("-v", "--input-volume-file",
                      action="store", dest="inputVolFile",
                      default=None,
                      help="Fetch coordinates from volumetric file")

    parser.add_argument("-o", "--output-file",
                      action="store", dest="outputFile",
                      default=None,
                      help="Output file. Otherwise standard output")

    parser.add_argument("-d", "--max-distance",
                      action="store", type=float, dest="maxDistance",
                      default=0,
                      help="When working with reference/distance atlases, what"
                      " maximal distance to use to look for the voxel of interest")

    parser.add_argument("-T", "--transformation-file",
                      dest="transformationFile",
                      help="First transformation to apply to the data. Usually"+
                      " should be subject -> standard(MNI) transformation")

    parser.add_argument("-s", "--summary",
                      action="count", dest="createSummary", default=0,
                      help="Either to create a summary instead of dumping voxels."
                      " Use multiple -s for greater verbose summary")


    parser.add_argument("--ss", "--sort-summary-by",
                       dest="sortSummaryBy", default="name",
                      choices=['name', 'count', 'a-p'],
                      help="How to sort summary entries. "
                      " a-p sorts anterior-posterior order")

    parser.add_argument("--dumpmap-file",
                      action="store", dest="dumpmapFile", default=None,
                      help="If original data is given as image file, dump indexes"
                      " per each treholded voxels into provided here output file")

    parser.add_argument("-l", "--levels",
                      dest="levels", default=None,
                      help="Indexes of levels which to print, or based on which "
                      "to create a summary (for a summary levels=4 is default). "
                      "To get listing of known for the atlas levels, use '-l list'")

    parser.add_argument("--mni2tal",
                      choices=["matthewbrett", "lancaster07fsl",
                               "lancaster07pooled", "meyerlindenberg98"],
                      dest="MNI2TalTransformation", default="matthewbrett",
                      help="Choose between available transformations from mni "
                      "2 talairach space")

    parser.add_argument("--thr", "--lthr", "--lower-threshold",
                      action="store", type=float, dest="lowerThreshold",
                      default=None,
                      help="Lower threshold for voxels to output")

    parser.add_argument("--uthr", "--upper-threshold",
                      action="store", type=float, dest="upperThreshold",
                      default=None,
                      help="Upper threshold for voxels to output")

    parser.add_argument("--count-thr", "--cthr",
                      action="store", type=int, dest="countThreshold",
                      default=1,
                      help="Lowest number of voxels for area to be reported in summary")

    parser.add_argument("--count-pthr", "--pthr",
                      action="store", type=float, dest="countPercentThreshold",
                      default=0.00001,
                      help="Lowest percentage of voxels within an area to be reported in summary")

    parser.add_argument("--suppress-none", "--sn",
                      action="store_true", dest="suppressNone",
                      help="Suppress reporting of voxels which found no labels (reported as None)")

    parser.add_argument("--abbr", "--abbreviated-labels",
                      action="store_true", dest="abbreviatedLabels",
                      help="Manipulate with abbreviations for labels instead of"
                      " full names, if the atlas has such")

    # Parameters to be inline with older talairachlabel

    parser.add_argument("-c", "--tc", "--show-target-coord",
                      action="store_true", dest="showTargetCoordinates",
                      help="Show target coordinates")


    parser.add_argument("--tv", "--show-target-voxel",
                      action="store_true", dest="showTargetVoxel",
                      help="Show target coordinates")

    parser.add_argument("--rc", "--show-referenced-coord",
                      action="store_true", dest="showReferencedCoordinates",
                      help="Show referenced coordinates/distance in case if we are"
                      " working with reference atlas")

    parser.add_argument("-C", "--oc", "--show-orig-coord",
                      action="store_true", dest="showOriginalCoordinates",
                      help="Show original coordinates")

    parser.add_argument("-V", "--show-values",
                      action="store_true", dest="showValues",
                      help="Show values")


    parser.add_argument("-I", "--input-space",
                      action="store", dest="inputSpace",
                      default="MNI",
                      help="Space in which input volume/coordinates provided in. For instance Talairach/MNI")

    parser.add_argument("-F", "--forbid-direct-mapping",
                      action="store_true", dest="forbidDirectMapping",
                      default=False,
                      help="If volume is provided it first tries to do direct "
                      "mapping voxel-2-voxel if there is no transformation file "
                      "given. This option forbids such behavior and does "
                      "coordinates mapping anyway.")

    parser.add_argument("-t", "--talairach",
                      action="store_true", dest="coordInTalairachSpace",
                      default=False,
                      help="Coordinates are in talairach space (1x1x1mm)," +
                      " otherwise assumes in mni space (2x2x2mm)."
                      " Shortcut for '-I Talairach'")

    parser.add_argument("-H", "--half-voxel-correction",
                      action="store_true", dest="halfVoxelCorrection",
                      default=False,
                      help="Adjust coord by 0.5mm after transformation to "
                      "Tal space.")

    parser.add_argument("-r", "--relative-to-origin",
                      action="store_true", dest="coordRelativeToOrigin",
                      help="Coords are relative to the origin standard form" +
                      " ie in spatial units (mm), otherwise the default assumes" +
                      " raw voxel dimensions")

    parser.add_argument("--input-line-format",
                      action="store", dest="inputLineFormat",
                      default=r"^\s*(?P<x>\S+)[ \t,]+(?P<y>\S+)[ \t,]+(?P<z>\S+)\s*$",
                      help="Format of the input lines (if ASCII input is provided)")

    parser.add_argument("--iv", "--input-voxels",
                      action="store_true", dest="input_voxels",
                      default=False,
                      help="Input lines carry voxel indices (int), not coordinates.")

    # Specific atlas options
    # TODO : group into options groups

    # Reference atlas
    parser.add_argument("-R", "--reference",
                      action="store", dest="referenceLevel",
                      default="Closest Gray",
                      help="Which level to reference in the case of reference"
                      " atlas")

    # Probabilistic atlases
    parser.add_argument("--prob-thr",
                      action="store", type=float, dest="probThr",
                      default=25.0,
                      help="At what probability (in %) to threshold in "
                      "probabilistic atlases (e.g. FSL)")

    parser.add_argument("--prob-strategy",
                      action="store", dest="probStrategy",
                      choices=['all', 'max'], default='max',
                      help="What strategy to use for reporting. 'max' would report"
                      " single area (above threshold) with maximal probabilitity")


def run(args):
    #atlas.relativeToOrigin = args.coordRelativeToOrigin

    fileIn = None
    coordT = None
    niftiInput = None
    # define data type for coordinates
    if args.input_voxels:
        ctype = int
        query_voxel = True
    else:
        ctype = float
        query_voxel = False

    # Setup coordinates read-in
    volQForm = None

    #
    # compatibility with older talairachlabel
    if args.inputCoordFile:
        fileIn = open(args.inputCoordFile)
        coordsIterator = parsed_coordinates_iterator(
            args.inputLineFormat, fileIn, ctype=ctype)
    if args.inputVolFile:
        infile = args.inputVolFile
        # got a volume/file to process
        if __debug__:
            debug('ATL', "Testing if 0th element in the list a volume")
        niftiInput = None
        try:
            niftiInput = nb.load(infile)
            if __debug__:
                debug('ATL', "Yes it is")
        except Exception, e:
            if __debug__:
                debug('ATL', "No it is not due to %s. Trying to parse the file" % e)

        if niftiInput:
            # if we got here -- it is a proper volume
            # XXX ask Michael to remove nasty warning message
            coordsIterator = select_from_volume_iterator(
                infile, args.lowerThreshold, args.upperThreshold)
            assert(coordT is None)
            coordT = Linear(niftiInput.get_header().get_qform())
            # lets store volumeQForm for possible conversion of voxels into coordinates
            volQForm = coordT
            # previous iterator returns space coordinates
            args.coordRelativeToOrigin = True
        else:
            raise ValueError('could not open volumetric input file')
    # input is stdin
    else:
        coordsIterator = parsed_coordinates_iterator(
            args.inputLineFormat, ctype=ctype)

    # Open and initialize atlas lookup
    if args.atlasFile is None:
        if args.atlasPath is None:
            args.atlasPath = KNOWN_ATLASES[args.atlasName]
        args.atlasFile = args.atlasPath % ( {'name': args.atlasName} )

    akwargs_common = {}
    if args.atlasImageFile:
        akwargs_common['image_file'] = args.atlasImageFile

    if not args.forbidDirectMapping \
           and niftiInput is not None and not args.transformationFile:
        akwargs = {'resolution': niftiInput.get_header().get_zooms()[0]}
        query_voxel = True   # if we can query directly by voxel, do so

        akwargs.update(akwargs_common)
        verbose(1, "Will attempt direct mapping from input voxels into atlas "
                   "voxels at resolution %.2f" % akwargs['resolution'])

        atlas = Atlas(args.atlasFile, **akwargs)

        # verify that we got the same qforms in atlas and in the data file
        if atlas.space != args.inputSpace:
            verbose(0,
                "Cannot do direct mapping between input image in %s space and"
                " atlas in %s space. Use -I switch to override input space if"
                " it misspecified, or use -T to provide transformation. Trying"
                " to proceed" %(args.inputSpace, atlas.space))
            query_voxel = False
        elif not (niftiInput.get_header().get_qform() == atlas._image.get_header().get_qform()).all():
            if args.atlasImageFile is None:
                warning(
                    "Cannot do direct mapping between files with different qforms."
                    " Please provide original transformation (-T)."
                    "\n Input qform:\n%s\n Atlas qform: \n%s"
                    %(niftiInput.get_header().get_qform(), atlas._image.get_header().get_qform), 1)
                # reset ability to query by voxels
                query_voxel = False
            else:
                warning(
                    "QForms are different between input image and "
                    "provided atlas image."
                    "\n Input qform of %s:\n%s\n Atlas qform of %s:\n%s"
                    %(infile, niftiInput.get_header().get_qform(),
                      args.atlasImageFile, atlas._image.get_header().get_qform()), 1)
        else:
            coordT = None
    else:
        atlas = Atlas(args.atlasFile, **akwargs_common)


    if isinstance(atlas, ReferencesAtlas):
        args.referenceLevel = args.referenceLevel.replace('/', ' ')
        atlas.set_reference_level(args.referenceLevel)
        atlas.distance = args.maxDistance
    else:
        args.showReferencedCoordinates = False

    if isinstance(atlas, FSLProbabilisticAtlas):
        atlas.strategy = args.probStrategy
        atlas.thr = args.probThr

    ## If not in Talairach -- in MNI with voxel size 2x2x2
    # Original talairachlabel assumed that if respective to origin -- voxels were
    # scaled already.
    #if args.coordInTalairachSpace:
    #   voxelSizeOriginal = np.array([1, 1, 1])
    #else:
    #   voxelSizeOriginal = np.array([2, 2, 2])

    if args.coordInTalairachSpace:
            args.inputSpace = "Talairach"

    if not (args.inputSpace == atlas.space or
            (args.inputSpace in ["MNI", "Talairach"] and
             atlas.space == "Talairach")):
        raise XMLAtlasException("Unknown space '%s' which is not the same as atlas "
                                "space '%s' either" % ( args.inputSpace, atlas.space ))

    if query_voxel:
        # we do direct mapping
        coordT = None
    else:
        verbose(2, "Chaining needed transformations")
        # by default -- no transformation
        if args.transformationFile:
            #externals.exists('scipy', raise_=True)
            # scipy.io.read_array was deprecated a while back (around 0.8.0)
            from numpy import loadtxt

            transfMatrix = loadtxt(args.transformationFile)
            coordT = Linear(transfMatrix, previous=coordT)
            verbose(2, "coordT got linear transformation from file %s" %
                       args.transformationFile)

        voxelOriginOriginal = None
        voxelSizeOriginal = None

        if not args.coordRelativeToOrigin:
            if args.inputSpace == "Talairach":
                # assume that atlas is in Talairach space already
                voxelOriginOriginal = atlas.origin
                voxelSizeOriginal = np.array([1, 1, 1])
            elif args.inputSpace == "MNI":
                # need to adjust for MNI origin as it was thought to be at
                # in terms of voxels
                #voxelOriginOriginal = np.array([46, 64, 37])
                voxelOriginOriginal = np.array([45, 63, 36])
                voxelSizeOriginal = np.array([2.0, 2.0, 2.0])
                warning("Assuming elderly sizes for MNI volumes with"
                           " origin %s and sizes %s" %\
                           ( `voxelOriginOriginal`, `voxelSizeOriginal`))


        if not (voxelOriginOriginal is None and voxelSizeOriginal is None):
            verbose(2, "Assigning origin adjusting transformation with"+\
                    " origin=%s and voxelSize=%s" %\
                    ( `voxelOriginOriginal`, `voxelSizeOriginal`))

            coordT = SpaceTransformation(origin=voxelOriginOriginal,
                                         voxelSize=voxelSizeOriginal,
                                         to_real_space=True, previous=coordT)

        # besides adjusting for different origin we need to transform into
        # Talairach space
        if args.inputSpace == "MNI" and atlas.space == "Talairach":
            verbose(2, "Assigning transformation %s" %
                       args.MNI2TalTransformation)
            # What transformation to use
            coordT = {"matthewbrett": MNI2Tal_MatthewBrett,
                      "lancaster07fsl":  mni_to_tal_lancaster07_fsl,
                      "lancaster07pooled":  mni_to_tal_lancaster07pooled,
                      "meyerlindenberg98":  mni_to_tal_meyer_lindenberg98,
                      "yohflirt": mni_to_tal_yohflirt
                      }\
                      [args.MNI2TalTransformation](previous=coordT)

        if args.inputSpace == "MNI" and args.halfVoxelCorrection:
            originCorrection = np.array([0.5, 0.5, 0.5])
        else:
            # perform transformation any way to convert to voxel space (integers)
            originCorrection = None

        # To be closer to what original talairachlabel did -- add 0.5 to each coord
        coordT = SpaceTransformation(origin=originCorrection, voxelSize=None,
                                         to_real_space=False, previous = coordT)

    if args.createSummary:
        summary = {}
        if args.levels is None:
            args.levels = str(min(4, atlas.nlevels-1))
    if args.levels is None:
        args.levels = range(atlas.nlevels)
    elif isinstance(args.levels, basestring):
        if args.levels == 'list':
            print "Known levels and their indicies:\n" + atlas.levels_listing()
            sys.exit(0)
        slevels = args.levels.split(',')
        args.levels = []
        for level in slevels:
            try:
                int_level = int(level)
            except:
                if atlas.levels.has_key(level):
                    int_level = atlas.levels[level].index
                else:
                    raise RuntimeError(
                        "Unknown level '%s'. " % level +
                        "Known levels and their indicies:\n"
                        + atlas.levels_listing())
            args.levels += [int_level]
    else:
        raise ValueError("Don't know how to handle list of levels %s."
                         "Example is '1,2,3'" % (args.levels,))

    verbose(3, "Operating on following levels: %s" % args.levels)
    # assign levels to the atlas
    atlas.default_levels = args.levels

    if args.outputFile:
        output = open(args.outputFile, 'w')
    else:
        output = sys.stdout

    # validity check
    if args.dumpmapFile:
        if niftiInput is None:
            raise RuntimeError, "You asked to dump indexes into the volume, " \
                  "but input wasn't a volume"
            sys.exit(1)
        ni_dump = nb.load(infile)
        ni_dump_data = np.zeros(ni_dump.get_header().get_data_shape()[:3] + (len(args.levels),))

    # Also check if we have provided voxels but not querying by voxels
    if args.input_voxels:
        if coordT is not None:
            raise NotImplementedError, \
                  "Cannot perform voxels querying having coordT defined"
        if not query_voxel:
            raise NotImplementedError, \
                  "query_voxel was reset to False, can't do queries by voxel"

    # Read coordinates
    numVoxels = 0
    for c in coordsIterator:

        value, coord_orig, t = c[0], c[1:4], c[4]
        if __debug__:
            debug('ATL', "Obtained coord_orig=%s with value %s"
                  % (repr(coord_orig), value))

        lt, ut = args.lowerThreshold, args.upperThreshold
        if lt is not None and value < lt:
            verbose(5, "Value %s is less than lower threshold %s, thus voxel "
                    "is skipped" % (value, args.lowerThreshold))
            continue
        if ut is not None and value > ut:
            verbose(5, "Value %s is greater than upper threshold %s, thus voxel "
                    "is skipped" % (value, args.upperThreshold))
            continue

        numVoxels += 1

        # Apply necessary transformations
        coord = coord_orig = np.array(coord_orig)

        if coordT:
            coord = coordT[ coord_orig ]

        # Query label
        if query_voxel:
            voxel = atlas[coord]
        else:
            voxel = atlas(coord)
        voxel['coord_orig'] = coord_orig
        voxel['value'] = value
        voxel['t'] = t
        if args.createSummary:
            summaryIndex = ""
            voxel_labels = voxel["labels"]
            for i,ind in enumerate(args.levels):
                voxel_label = voxel_labels[i]
                text = present_labels(args, voxel_label)
                #if len(voxel_label):
                #   assert(voxel_label['index'] == ind)
                summaryIndex += text + " / "
            if not summary.has_key(summaryIndex):
                summary[summaryIndex] = {'values':[], 'max':value,
                                         'maxcoord':coord_orig}
                if voxel.has_key('voxel_referenced'):
                    summary[summaryIndex]['distances'] = []
            summary_ = summary[summaryIndex]
            summary_['values'].append(value)
            if summary_['max'] < value:
                summary_['max'] = value
                summary_['maxcoord'] = coord_orig
            if voxel.has_key('voxel_referenced'):
                if voxel['voxel_referenced'] and voxel['distance']>=1e-3:
                    verbose(5, 'Appending distance %e for voxel at %s'
                            % (voxel['distance'], voxel['coord_orig']))
                    summary_['distances'].append(voxel['distance'])
        else:
            # Display while reading/processing
            first, out = True, ""

            if args.showValues:
                out += "%(value)5.2f "
            if args.showOriginalCoordinates:
                out += "%(coord_orig)s ->"
            if args.showReferencedCoordinates:
                out += " %(voxel_referenced)s=>%(distance).2f=>%(voxel_queried)s ->"
            if args.showTargetCoordinates:
                out += " %(coord_queried)s: "
                #out += "(%d,%d,%d): " % tuple(map(lambda x:int(round(x)),coord))
            if args.showTargetVoxel:
                out += " %(voxel_queried)s ->"

            if args.levels is None:
                args.levels = range(len(voxel['labels']))

            labels = [present_labels(args, voxel['labels'][i]) for i in args.levels]
            out += ','.join(labels)
            #if args.abbreviatedLabels:
            #   out += ','.join([l.abbr for l in labels])
            #else:
            #   out += ','.join([l.text for l in labels])
            #try:
            output.write(out % voxel + "\n")
            #except:
            #    import pydb
            #    pydb.debugger()

        if args.dumpmapFile:
            try:
                ni_dump_data[coord_orig[0], coord_orig[1], coord_orig[2]] = \
                  [voxel['labels'][i]['label'].index
                   for i,ind in enumerate(args.levels)]
            except Exception, e:
                import pydb
                pydb.debugger()

    # if we opened any file -- close it
    if fileIn:
        fileIn.close()

    if args.dumpmapFile:
        ni_dump = nb.Nifti1Image(ni_dump_data, None, ni_dump.get_header())
        ni_dump.to_filename(args.dumpmapFile)

    if args.createSummary:
        if numVoxels == 0:
            verbose(1, "No matching voxels were found.")
        else:
            get_summary(args, summary, output)

    if args.outputFile:
        output.close()
