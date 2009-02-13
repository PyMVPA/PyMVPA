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
Searchlight Analysis on an fMRI Dataset
=======================================

Example demonstrating a searchlight analysis on an fMRI dataset.
"""

from mvpa.suite import *


def main():
    """ Wrapped into a function call for easy profiling later on
    """

    parser.usage = """\
    %s [options] <NIfTI samples> <labels+blocks> <NIfTI mask> [<output>]

    where labels+blocks is a text file that lists the class label and the
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
    # text file with labels and block definitions (chunks)
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
    attrs = SampleAttributes(cfile)

    verbose(2, "Loading volume file %s" % dfile)
    data = NiftiDataset(samples=dfile,
                        labels=attrs.labels,
                        chunks=attrs.chunks,
                        mask=mfile,
                        dtype=N.float32)

    # do not try to classify baseline condition
    # XXX this is only valid for our haxby8 example dataset and should
    # probably be turned into a proper --baselinelabel option that can
    # be used for zscoring as well.
    data = data.selectSamples(data.labels != 0)

    if options.zscore:
        verbose(1, "Zscoring data samples")
        zscore(data, perchunk=True)

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
    cv = CrossValidatedTransferError(TransferError(clf),
                            NFoldSplitter(cvtype=options.crossfolddegree))

    verbose(3, "Generating Searchlight instance")
    # contruct searchlight with 5mm radius
    # this assumes that the spatial pixdim values in the source NIfTI file
    # are specified in mm
    sl = Searchlight(cv, radius=options.radius)

    # run searchlight
    verbose(3, "Running searchlight on loaded data")
    results = sl(data)

    if not ofile is None:
        # map the result vector back into a nifti image
        rimg = data.map2Nifti(results)

        # save to file
        rimg.save(ofile)
    else:
        print results

if __name__ == "__main__":
    main()
