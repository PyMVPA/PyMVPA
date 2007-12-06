#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
NOTEs: sbj5 had no labels for Rrun9 (ie chunk 8)
"""

__docformat__ = 'restructuredtext'



import numpy as N
import re, glob, os

from nifti import NiftiImage
from optparse import OptionParser

from mvpa.misc import verbose
from mvpa.misc.cmdline import optVerbose


#sbjs={1:'1_ii', 2:'2_ss', 3:'3_js'

usage="topdir output_prefix"

parser = OptionParser(usage=usage, option_list=[optVerbose])

parser.add_option("-r", "--resurrect",
                  action="store", type="int", dest="resurrect", default=1,
                  help="How many volumes lost their labels due to the beginning of the block 'discard'")

parser.add_option("-a", "--resurrect-after",
                  action="store", type="int", dest="resurrectafter", default=1,
                  help="How many volumes lost their labels due to the end of the block 'discard'")

parser.add_option("-b", "--base-category",
                  action="store", type="string", dest="base_category", default="rest",
                  help="What is base category -- ie for whenever sample has no label")

parser.add_option("-c", "--categories",
                  action="store", type="string", dest="categories",
                  default="bottle,cat,chair,face,house,scissors,scrambledpix,shoe",
                  help="Comma separated list of categories")

parser.add_option("-l", "--just-labels",
                  action="store_true",  dest="just_labels", default=False,
                  help="If just to process labels and do not bother about files")

parser.add_option("-n", "--dry-run",
                  action="store_true",  dest="dry_run", default=False,
                  help="If just to print commands to run")


(options, files) = parser.parse_args()

if len(files)!=2:
    print "Please provide both original directory and output file prefix"

basedir, outprefix = files

verbose(1, "Basedir: '%s' outprefix: '%s'" % (basedir, outprefix))
#create  directory if outprefix ends with /
if outprefix.endswith("/"):
    try:
        os.mkdir(outprefix)
        verbose(1, "Created directory %s" % outprefix)
    except:
        pass

filenames = glob.glob("%s/Rrun*.img" % basedir)

# lets don't be smartasses
startsample = 5
nvolumesinchunk = 121

def parse_name(fname):
    res = re.match("Rrun(?P<run>[0-9]+)(?P<sample>[0-9]{3}).img$", fname)
    if not res:
        raise ValueError, "Filename %s didn't match our pattern" % fname
    resd = res.groupdict()
    return (int(resd['run'])-1, int(resd['sample'])-startsample)

def sample_offset (chunk, sample):
    if chunk >= nchunks:
        raise ValueError, "Yarik go to sleep -- wrong chunk %d" % chunk
    return chunk*nvolumesinchunk + sample

def fileExists(filename, warn=True):
    res = len(glob.glob(filename)) > 0
    if res and warn:
        verbose(2, "File %s exists already" % filename)
    return res

def ossystem(cmd):
    if options.dry_run:
        print "RUN: ", cmd
        return 0
    else:
        return os.system(cmd)

cats = options.categories.split(",")

nvolumes = len(filenames)
#if nvolumes != nchunks * nvolumesinchunk:
#    raise ValueError, "We got total %s volumes whenever must be %s" % (nvolumes, nchunks * nvolumesinchunk)
nchunks = nvolumes / nvolumesinchunk
if nvolumes % nvolumesinchunk != 0:
    raise ValueError, "It seems that number of the volumes %d is not even to number of volumes within a chunk" % nvolumes

verbose(1, "Number of chunks = %d" % nchunks)
# create chunks indexes
chunks = [ [i]*nvolumesinchunk for i in xrange(nchunks) ]
chunks = reduce(lambda x,y: x+y, chunks, []) # flatten the beast
labels = [ options.base_category ] * nvolumes

# read labels from files
for cat in cats:
    listfilename = "%s/filelists/files.%s" % (basedir, cat)
    files = open(listfilename, 'r').readlines()
    files = [ x.strip() for x in files ]
    prevsample = 0
    for fname in files:
        verbose(3, "File: %s\t" % fname, lf=False) 
        run, sample = parse_name(fname)
        # I am still smart ass... why why why
        if run*1000+sample - 1 != prevsample:
            prevsample = run*1000 + sample
            #verbose(3, "Detected beginning of the block")
            for o in range(options.resurrect): # 
                index = sample_offset(run, sample-o-1)
                verbose(4, "resurrected at offset %d category %s " %
                        (index, cat))
                labels[index] = cat

        index = sample_offset(run, sample)
        verbose(4, "offset %d, category %s " %
                (index, cat))
        labels[index] = cat
        for o in range(options.resurrectafter+1):
            labels[index+o] = cat

labelsfile = '%slabels.txt' % outprefix
if not fileExists(labelsfile):
    fout = open(labelsfile, 'w')
    print >>fout, "labels chunks"
    for i in xrange(len(labels)):
        print >>fout, labels[i], chunks[i]
    fout.close()

if options.just_labels:
    verbose(1, 'done')
    sys.exit(0)
# now we have to go through the files and suck them up into a single volume.
# and since we are at it -- correct everything which needs to be corrected

# too late at night, so lets just assure the right order and feed it into
# avwmerge with consecutive avwswapdim
sorted_fnames = []
for filename in filenames:
    fname = os.path.basename(filename)
    chunk, volume = parse_name(fname)
    sorted_fnames.append( (chunk*1000+volume, fname) )
sorted_fnames.sort()
flist = reduce(lambda x,y: x+ " " + y[1], sorted_fnames, "")

if not fileExists("%sbold.nii.gz" % outprefix):
    verbose(1, "Merging all volumes into a single chunk")
    verbose(5, "Merging volumes in the following order: %s" % `flist`)
    ossystem("cd %s; avwmerge -t %sbold-preswap.nii.gz %s" % (basedir, outprefix, flist))
    verbose(2, "Swapping dimensions of the volumes")
    ossystem("avwswapdim %sbold-preswap.nii.gz z y -x %sbold.nii.gz" % (outprefix, outprefix))
    # obtained bold is flipped in comparison to michael's

if not fileExists("%sanat.nii.gz" % outprefix):
    verbose(2, "Copying anatomical")
    ossystem("avwchfiletype NIFTI_GZ %s/anat.img %sanat-preswap.nii.gz" % ( basedir, outprefix) )
    ossystem("avwswapdim %sanat-preswap.nii.gz z -x -y %sanat.nii.gz" % (outprefix, outprefix))

if not fileExists("%smask4_vt.nii.gz" % outprefix):
    verbose(2, "Copying masks, reorienting them the same way as bolds")
    for fnamefull in glob.glob("%s/../../masks/mask*+orig_0000.img" % basedir):
        fname = os.path.basename(fnamefull)
        fnamebase = os.path.splitext(fname)[0]
        fnameout = fnamebase.replace('+orig_0000', '')
        ossystem("avwchfiletype NIFTI_GZ %s %s%s-preswap.nii.gz" % ( fnamefull, outprefix, fnameout) )
        ossystem("avwswapdim %s%s-preswap.nii.gz z y -x %s%s.nii.gz" % (outprefix, fnameout, outprefix, fnameout))

verbose(2, "Removing preswap files")
ossystem("rm -f %s/*preswap.nii.gz" % outprefix)

# for now do not do MC at all
if False: # not fileExists("%sbold_mcf.nii.gz" % outprefix):
    verbose(2, "Doing motion correction")
    try:
        os.mkdir("%smc" % outprefix)
        verbose(1, "Created directory %smc" % outprefix)
    except:
        pass

    ossystem("mcflirt -in %s/bold.nii.gz -out %smc/bold_mcf -mats -plots -rmsrel -rmsabs" %
              (outprefix, outprefix))


    verbose(2, "Moving bold_mcf upstairs")
    ossystem("mv %smc/bold_mcf.nii.gz %sbold_mcf.nii.gz" % (outprefix, outprefix))
