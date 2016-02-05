#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Parameterizable Python Scripting: Searchlight Example.
======================================================

Example demonstrating composition of analysis script with optional
command line parameters and arguments to make the computation easily
parameterizable.  That would allow you to process multiple datasets
and vary classifiers and/or parameters of the algorithm within some
batch system scheduler.  Searchlight analysis on an fMRI dataset is
taken for the example of actual computation to be done.  Run
`searchlight.py --help` to see the list of available command line
options.
"""

from mvpa2.suite import *


def main():
    """ Wrapped into a function call for easy profiling later on
    """

    parser.usage = """\
    %s [options] <NIfTI samples> <targets+blocks> <NIfTI mask> [<output>]

    where targets+blocks is a text file that lists the class label and the
    associated block of each data sample/volume as a tuple of two integer
    values (separated by a single space). -- one tuple per line.""" \
    % sys.argv[0]

    parser.option_groups = [opts.SVM, opts.KNN, opts.general, opts.common]

    # Set a set of available classifiers for this example
    opt.clf.choices=['knn', 'lin_nu_svmc', 'rbf_nu_svmc']
    opt.clf.default='lin_nu_svmc'

    parser.add_options([opt.clf, opt.zscore])

    (options, files) = parser.parse_args()

    if not len(files) in [3, 4]:
        parser.error("Please provide 3 or 4 files in the command line")
        sys.exit(1)

    verbose(1, "Loading data")

    # data filename
    dfile = files[0]
    # text file with targets and block definitions (chunks)
    cfile = files[1]
    # mask volume filename
    mfile = files[2]

    ofile = None
    if len(files)>=4:
        # outfile name
        ofile = files[3]

    # read conditions into an array (assumed to be two columns of integers)
    # TODO: We need some generic helper to read conditions stored in some
    #       common formats
    verbose(2, "Reading conditions from file %s" % cfile)
    attrs = SampleAttributes(cfile, literallabels=True)

    verbose(2, "Loading volume file %s" % dfile)
    data = fmri_dataset(dfile,
                         targets=attrs.targets,
                         chunks=attrs.chunks,
                         mask=mfile)

    # do not try to classify baseline condition
    # XXX this is only valid for our haxby8 example dataset and should
    # probably be turned into a proper --baselinelabel option that can
    # be used for zscoring as well.
    data = data[data.targets != 'rest']

    if options.zscore:
        verbose(1, "Zscoring data samples")
        zscore(data, chunks_attr='chunks', dtype='float32')

    if options.clf == 'knn':
        clf = kNN(k=options.knearestdegree)
    elif options.clf == 'lin_nu_svmc':
        clf = LinearNuSVMC(nu=options.svm_nu)
    elif options.clf == 'rbf_nu_svmc':
        clf = RbfNuSVMC(nu=options.svm_nu)
    else:
        raise ValueError, 'Unknown classifier type: %s' % `options.clf`
    verbose(3, "Using '%s' classifier" % options.clf)

    verbose(1, "Computing")

    verbose(3, "Assigning a measure to be CrossValidation")
    # compute N-1 cross-validation with the selected classifier in each sphere
    cv = CrossValidation(clf, NFoldPartitioner(cvtype=options.crossfolddegree))

    verbose(3, "Generating Searchlight instance")
    # contruct searchlight with 5mm radius
    # this assumes that the spatial pixdim values in the source NIfTI file
    # are specified in mm
    sl = sphere_searchlight(cv, radius=options.radius)

    # run searchlight
    verbose(3, "Running searchlight on loaded data")
    results = sl(data)

    if ofile is not None:
        # map the result vector back into a nifti image
        rimg = map2nifti(data, results)

        # save to file
        rimg.save(ofile)
    else:
        print results

if __name__ == "__main__":
    main()
