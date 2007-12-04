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
import re, glob

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
                  action="store_true",  dest="just_labels",
                  help="If just to process labels and do not bother about files")


(options, files) = parser.parse_args()

if len(files)!=2:
    print "Please provide both original directory and output file prefix"

basedir, outprefix = files


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
        verbose(2, "File: %s\t" % fname, lf=False) 
        run, sample = parse_name(fname)
        # I am still smart ass... why why why
        if run*1000+sample - 1 != prevsample:
            prevsample = run*1000 + sample
            #verbose(3, "Detected beginning of the block")
            for o in range(options.resurrect): # 
                index = sample_offset(run, sample-o-1)
                verbose(3, "resurrected at offset %d category %s " %
                        (index, cat))
                labels[index] = cat

        index = sample_offset(run, sample)
        verbose(4, "offset %d, category %s " %
                (index, cat))
        labels[index] = cat
        for o in range(options.resurrectafter+1):
            labels[index+o] = cat


fout = open('%s-labels.txt' % outprefix, 'w')
print >>fout, "labels chunks"
for i in xrange(len(labels)):
    print >>fout, labels[i], chunks[i]
fout.close()

if options.just_labels:
    verbose(1, 'done')

# now we have to go through the files and suck them up into a single volume.
# and since we are at it -- correct everything which needs to be corrected

