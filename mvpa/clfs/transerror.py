# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utility class to compute the transfer error of classifiers."""

__docformat__ = 'restructuredtext'

import mvpa.support.copy as copy

import numpy as np

from sets import Set
from StringIO import StringIO
from math import log10, ceil

from mvpa.base import externals

from mvpa.misc.errorfx import mean_power_fx, root_mean_power_fx, RMSErrorFx, \
     CorrErrorFx, CorrErrorPFx, RelativeRMSErrorFx, MeanMismatchErrorFx, \
     AUCErrorFx
from mvpa.base import warning
from mvpa.base.collections import Collectable
from mvpa.base.state import ConditionalAttribute, ClassWithCollections
from mvpa.base.dochelpers import enhanced_doc_string, table2string
from mvpa.clfs.stats import auto_null_dist

if __debug__:
    from mvpa.base import debug

if externals.exists('scipy'):
    from scipy.stats.stats import nanmean
    from mvpa.misc.stats import chisquare
else:
    from mvpa.clfs.stats import nanmean
    chisquare = None

def _p2(x, prec=2):
    """Helper to print depending on the type nicely. For some
    reason %.2g for 100 prints exponential form which is ugly
    """
    if isinstance(x, int):
        return "%d" % x
    elif isinstance(x, float):
        s = ("%%.%df" % prec % x).rstrip('0').rstrip('.').lstrip()
        if s == '':
            s = '0'
        return s
    else:
        return "%s" % x



class SummaryStatistics(object):
    """Basic class to collect targets/predictions and report summary statistics

    It takes care about collecting the sets, which are just tuples
    (targets, predictions, estimates). While 'computing' the matrix, all
    sets are considered together.  Children of the class are
    responsible for computation and display.
    """

    _STATS_DESCRIPTION = (
        ('# of sets',
         'number of target/prediction sets which were provided',
         None), )


    def __init__(self, targets=None, predictions=None, estimates=None, sets=None):
        """Initialize SummaryStatistics

        targets or predictions cannot be provided alone (ie targets
        without predictions)

        Parameters
        ----------
        targets
         Optional set of targets
        predictions
         Optional set of predictions
        estimates
         Optional set of estimates (which served for prediction)
        sets
         Optional list of sets
        """
        self._computed = False
        """Flag either it was computed for a given set of data"""

        self.__sets = (sets, [])[int(sets is None)]
        """Datasets (target, prediction) to compute confusion matrix on"""

        self._stats = {}
        """Dictionary to keep statistics. Initialized here to please pylint"""

        if not targets is None or not predictions is None:
            if not targets is None and not predictions is None:
                self.add(targets=targets, predictions=predictions,
                         estimates=estimates)
            else:
                raise ValueError, \
                      "Please provide none or both targets and predictions"


    def add(self, targets, predictions, estimates=None):
        """Add new results to the set of known results"""
        if len(targets) != len(predictions):
            raise ValueError, \
                  "Targets[%d] and predictions[%d]" % (len(targets),
                                                       len(predictions)) + \
                  " have different number of samples"

        # extract value if necessary
        if isinstance(estimates, Collectable):
            estimates = estimates.value

        if estimates is not None and len(targets) != len(estimates):
            raise ValueError, \
                  "Targets[%d] and estimates[%d]" % (len(targets),
                                                  len(estimates)) + \
                  " have different number of samples"

        # enforce labels in predictions to be of the same datatype as in
        # targets, since otherwise we are getting doubles for unknown at a
        # given moment labels
        nonetype = type(None)
        for i in xrange(len(targets)):
            t1, t2 = type(targets[i]), type(predictions[i])
            # if there were no prediction made - leave None, otherwise
            # convert to appropriate type
            if t1 != t2 and t2 != nonetype:
                #warning("Obtained target %s and prediction %s are of " %
                #       (t1, t2) + "different datatypes.")
                if isinstance(predictions, tuple):
                    predictions = list(predictions)
                predictions[i] = t1(predictions[i])

        if estimates is not None:
            # assure that we have a copy, or otherwise further in-place
            # modifications might screw things up (some classifiers share
            # estimates and spit out results)
            estimates = copy.deepcopy(estimates)

        self.__sets.append( (targets, predictions, estimates) )
        self._computed = False


    ##REF: Name was automagically refactored
    def as_string(self, short=False, header=True, summary=True,
                 description=False):
        """'Pretty print' the matrix

        Parameters
        ----------
        short : bool
          if True, ignores the rest of the parameters and provides consise
          1 line summary
        header : bool
          print header of the table
        summary : bool
          print summary (accuracy)
        description : bool
          print verbose description of presented statistics
        """
        raise NotImplementedError


    def __str__(self):
        """String summary over the `SummaryStatistics`

        It would print description of the summary statistics if 'CM'
        debug target is active
        """
        if __debug__:
            description = ('CM' in debug.active)
        else:
            description = False
        return self.as_string(short=False, header=True, summary=True,
                             description=description)


    def __iadd__(self, other):
        """Add the sets from `other` s `SummaryStatistics` to current one
        """
        #print "adding ", other, " to ", self
        # need to do shallow copy, or otherwise smth like "cm += cm"
        # would loop forever and exhaust memory eventually
        othersets = copy.copy(other.__sets)
        for set in othersets:
            self.add(*set)#[0], set[1])
        return self


    def __add__(self, other):
        """Add two `SummaryStatistics`s
        """
        result = copy.copy(self)
        result += other
        return result


    def compute(self):
        """Actually compute the confusion matrix based on all the sets"""
        if self._computed:
            return

        self._compute()
        self._computed = True


    def _compute(self):
        """Compute basic statistics
        """
        self._stats = {'# of sets' : len(self.sets)}


    @property
    def summaries(self):
        """Return a list of separate summaries per each stored set"""
        return [ self.__class__(sets=[x]) for x in self.sets ]


    @property
    def error(self):
        raise NotImplementedError


    @property
    def stats(self):
        self.compute()
        return self._stats


    def reset(self):
        """Cleans summary -- all data/sets are wiped out
        """
        self.__sets = []
        self._computed = False


    sets = property(lambda self:self.__sets)


class ROCCurve(object):
    """Generic class for ROC curve computation and plotting
    """

    def __init__(self, labels, sets=None):
        """
        Parameters
        ----------
        labels : list
          labels which were used (in order of estimates if multiclass,
          or 1 per class for binary problems (e.g. in SMLR))
        sets : list of tuples
          list of sets for the analysis
        """
        self._labels = labels
        self._sets = sets
        self.__computed = False


    def _compute(self):
        """Lazy computation if needed
        """
        if self.__computed:
            return
        # local bindings
        labels = self._labels
        Nlabels = len(labels)
        sets = self._sets

        # Handle degenerate cases politely
        if Nlabels < 2:
            warning("ROC was asked to be evaluated on data with %i"
                    " labels which is a degenerate case." % Nlabels)
            self._ROCs = []
            self._aucs = []
            return

        # take sets which have values in the shape we can handle
        ##REF: Name was automagically refactored
        def _check_values(set_):
            """Check if values are 'acceptable'"""
            if len(set_)<3: return False
            x = set_[2]
            # TODO: OPT: need optimization
            if (x is None) or len(x) == 0: return False          # undefined
            for v in x:
                try:
                    if Nlabels <= 2 and np.isscalar(v):
                        continue
                    if (isinstance(v, dict) or # not dict for pairs
                        ((Nlabels>=2) and len(v)!=Nlabels) # 1 per each label for multiclass
                        ): return False
                except Exception, e:
                    # Something else which is not supported, like
                    # in shogun interface we don't yet extract values per each label or
                    # in pairs in the case of built-in multiclass
                    if __debug__:
                        debug('ROC', "Exception %s while checking "
                              "either %s are valid labels" % (str(e), x))
                    return False
            return True

        sets_wv = filter(_check_values, sets)
        # check if all had values, if not -- complain
        Nsets_wv = len(sets_wv)
        if Nsets_wv > 0 and len(sets) != Nsets_wv:
            warning("Only %d sets have values assigned from %d sets. "
                    "ROC estimates might be incorrect." %
                    (Nsets_wv, len(sets)))
        # bring all values to the same 'shape':
        #  1 value per each label. In binary classifier, if only a single
        #  value is provided, add '0' for 0th label 'value'... it should
        #  work taking drunk Yarik logic ;-)
        # yoh: apparently it caused problems whenever we had just a single
        #      unique label in the sets. Introduced handling for
        #      NLabels == 1
        for iset,s in enumerate(sets_wv):
            # we will do inplace modification, thus go by index
            estimates = s[2]
            # we would need it to be a list to reassign element with a list
            if isinstance(estimates, np.ndarray) and len(estimates.shape)==1:
                # XXX ??? so we are going away from inplace modifications?
                estimates = list(estimates)
            rangev = None
            for i in xrange(len(estimates)):
                v = estimates[i]
                if np.isscalar(v):
                    if Nlabels == 1:
                        # ensure the right dimensionality
                        estimates[i] = np.array(v, ndmin=2)
                    elif Nlabels == 2:
                        def last_el(x):
                            """Helper function. Returns x if x is scalar, and
                            last element if x is not (ie list/tuple)"""
                            if np.isscalar(x): return x
                            else:             return x[-1]
                        if rangev is None:
                            # we need to figure out min/max estimates
                            # to invert for the 0th label
                            estimates_ = [last_el(x) for x in estimates]
                            rangev = np.min(estimates_) + np.max(estimates_)
                        estimates[i] = [rangev - v, v]
                    else:
                        raise ValueError, \
                              "Cannot have a single 'value' for multiclass" \
                              " classification. Got %s" % (v)
                elif len(v) != Nlabels:
                    raise ValueError, \
                          "Got %d estimates whenever there is %d labels" % \
                          (len(v), Nlabels)
            # reassign possibly adjusted estimates
            sets_wv[iset] = (s[0], s[1], np.asarray(estimates))


        # we need to estimate ROC per each label
        # XXX order of labels might not correspond to the one among 'estimates'
        #     which were used to make a decision... check
        rocs, aucs = [], []             # 1 per label
        for i,label in enumerate(labels):
            aucs_pl = []
            ROCs_pl = []
            for s in sets_wv:
                targets_pl = (np.asanyarray(s[0]) == label).astype(int)
                # XXX we might unify naming between AUC/ROC
                ROC = AUCErrorFx()
                aucs_pl += [ROC([np.asanyarray(x)[i] for x in s[2]], targets_pl)]
                ROCs_pl.append(ROC)
            if len(aucs_pl)>0:
                rocs += [ROCs_pl]
                aucs += [nanmean(aucs_pl)]
                #aucs += [np.mean(aucs_pl)]

        # store results within the object
        self._ROCs =  rocs
        self._aucs = aucs
        self.__computed = True


    @property
    def aucs(self):
        """Compute and return set of AUC values 1 per label
        """
        self._compute()
        return self._aucs


    @property
    ##REF: Name was automagically refactored
    def rocs(self):
        self._compute()
        return self._ROCs


    def plot(self, label_index=0):
        """

        TODO: make it friendly to labels given by values?
              should we also treat labels_map?
        """
        externals.exists("pylab", raise_=True)
        import pylab as pl

        self._compute()

        labels = self._labels
        # select only rocs for the given label
        rocs = self.rocs[label_index]

        fig = pl.gcf()
        ax = pl.gca()

        pl.plot([0, 1], [0, 1], 'k:')

        for ROC in rocs:
            pl.plot(ROC.fp, ROC.tp, linewidth=1)

        pl.axis((0.0, 1.0, 0.0, 1.0))
        pl.axis('scaled')
        pl.title('Label %s. Mean AUC=%.2f' % (label_index, self.aucs[label_index]))

        pl.xlabel('False positive rate')
        pl.ylabel('True positive rate')


class ConfusionMatrix(SummaryStatistics):
    """Class to contain information and display confusion matrix.

    Implementation of the `SummaryStatistics` in the case of
    classification problem. Actual computation of confusion matrix is
    delayed until all data is acquired (to figure out complete set of
    labels). If testing data doesn't have a complete set of labels,
    but you like to include all labels, provide them as a parameter to
    the constructor.

    Confusion matrix provides a set of performance statistics (use
    as_string(description=True) for the description of abbreviations),
    as well ROC curve (http://en.wikipedia.org/wiki/ROC_curve)
    plotting and analysis (AUC) in the limited set of problems:
    binary, multiclass 1-vs-all.
    """

    _STATS_DESCRIPTION = (
        ('TP', 'true positive (AKA hit)', None),
        ('TN', 'true negative (AKA correct rejection)', None),
        ('FP', 'false positive (AKA false alarm, Type I error)', None),
        ('FN', 'false negative (AKA miss, Type II error)', None),
        ('TPR', 'true positive rate (AKA hit rate, recall, sensitivity)',
                'TPR = TP / P = TP / (TP + FN)'),
        ('FPR', 'false positive rate (AKA false alarm rate, fall-out)',
                'FPR = FP / N = FP / (FP + TN)'),
        ('ACC', 'accuracy', 'ACC = (TP + TN) / (P + N)'),
        ('SPC', 'specificity', 'SPC = TN / (FP + TN) = 1 - FPR'),
        ('PPV', 'positive predictive value (AKA precision)',
                'PPV = TP / (TP + FP)'),
        ('NPV', 'negative predictive value', 'NPV = TN / (TN + FN)'),
        ('FDR', 'false discovery rate', 'FDR = FP / (FP + TP)'),
        ('MCC', "Matthews Correlation Coefficient",
                "MCC = (TP*TN - FP*FN)/sqrt(P N P' N')"),
        ('AUC', "Area under (AUC) curve", None),
        ('CHI^2', "Chi-square of confusion matrix", None),
        ) + SummaryStatistics._STATS_DESCRIPTION


    def __init__(self, labels=None, labels_map=None, **kwargs):
        """Initialize ConfusionMatrix with optional list of `labels`

        Parameters
        ----------
        labels : list
         Optional set of labels to include in the matrix
        labels_map : None or dict
         Dictionary from original dataset to show mapping into
         numerical labels
        targets
         Optional set of targets
        predictions
         Optional set of predictions
         """

        SummaryStatistics.__init__(self, **kwargs)

        if labels == None:
            labels = []
        self.__labels = labels
        """List of known labels"""
        self.__labels_map = labels_map
        """Mapping from original into given labels"""
        self.__matrix = None
        """Resultant confusion matrix"""


    # XXX might want to remove since summaries does the same, just without
    #     supplying labels
    @property
    def matrices(self):
        """Return a list of separate confusion matrix per each stored set"""
        return [ self.__class__(labels=self.labels,
                                labels_map=self.labels_map,
                                sets=[x]) for x in self.sets]


    def _compute(self):
        """Actually compute the confusion matrix based on all the sets"""

        super(ConfusionMatrix, self)._compute()

        if __debug__:
            if not self.__matrix is None:
                debug("LAZY",
                      "Have to recompute %s#%s" \
                        % (self.__class__.__name__, id(self)))


        # TODO: BinaryClassifier might spit out a list of predictions for each
        # value need to handle it... for now just keep original labels
        try:
            # figure out what labels we have
            labels = \
                list(reduce(lambda x, y: x.union(Set(y[0]).union(Set(y[1]))),
                            self.sets,
                            Set(self.__labels)))
        except:
            labels = self.__labels

        # Check labels_map if it was provided if it covers all the labels
        labels_map = self.__labels_map
        if labels_map is not None:
            labels_set = Set(labels)
            map_labels_set = Set(labels_map.values())

            if not map_labels_set.issuperset(labels_set):
                warning("Provided labels_map %s is not coherent with labels "
                        "provided to ConfusionMatrix. No reverse mapping "
                        "will be provided" % labels_map)
                labels_map = None

        # Create reverse map
        labels_map_rev = None
        if labels_map is not None:
            labels_map_rev = {}
            for k,v in labels_map.iteritems():
                v_mapping = labels_map_rev.get(v, [])
                v_mapping.append(k)
                labels_map_rev[v] = v_mapping
        self.__labels_map_rev = labels_map_rev

        labels.sort()
        self.__labels = labels          # store the recomputed labels

        Nlabels, Nsets = len(labels), len(self.sets)

        if __debug__:
            debug("CM", "Got labels %s" % labels)

        # Create a matrix for all votes
        mat_all = np.zeros( (Nsets, Nlabels, Nlabels), dtype=int )

        # create total number of samples of each label counts
        # just for convinience I guess since it can always be
        # computed from mat_all
        counts_all = np.zeros( (Nsets, Nlabels) )

        # reverse mapping from label into index in the list of labels
        rev_map = dict([ (x[1], x[0]) for x in enumerate(labels)])
        for iset, set_ in enumerate(self.sets):
            for t,p in zip(*set_[:2]):
                mat_all[iset, rev_map[p], rev_map[t]] += 1


        # for now simply compute a sum of votes across different sets
        # we might do something more sophisticated later on, and this setup
        # should easily allow it
        self.__matrix = np.sum(mat_all, axis=0)
        self.__Nsamples = np.sum(self.__matrix, axis=0)
        self.__Ncorrect = sum(np.diag(self.__matrix))

        TP = np.diag(self.__matrix)
        offdiag = self.__matrix - np.diag(TP)
        stats = {
            '# of labels' : Nlabels,
            'TP' : TP,
            'FP' : np.sum(offdiag, axis=1),
            'FN' : np.sum(offdiag, axis=0)}

        stats['CORR']  = np.sum(TP)
        stats['TN']  = stats['CORR'] - stats['TP']
        stats['P']  = stats['TP'] + stats['FN']
        stats['N']  = np.sum(stats['P']) - stats['P']
        stats["P'"] = stats['TP'] + stats['FP']
        stats["N'"] = stats['TN'] + stats['FN']
        stats['TPR'] = stats['TP'] / (1.0*stats['P'])
        # reset nans in TPRs to 0s whenever there is no entries
        # for those labels among the targets
        stats['TPR'][stats['P'] == 0] = 0
        stats['PPV'] = stats['TP'] / (1.0*stats["P'"])
        stats['NPV'] = stats['TN'] / (1.0*stats["N'"])
        stats['FDR'] = stats['FP'] / (1.0*stats["P'"])
        stats['SPC'] = (stats['TN']) / (1.0*stats['FP'] + stats['TN'])

        MCC_denom = np.sqrt(1.0*stats['P']*stats['N']*stats["P'"]*stats["N'"])
        nz = MCC_denom!=0.0
        stats['MCC'] = np.zeros(stats['TP'].shape)
        stats['MCC'][nz] = \
                 (stats['TP'] * stats['TN'] - stats['FP'] * stats['FN'])[nz] \
                  / MCC_denom[nz]

        stats['ACC'] = np.sum(TP)/(1.0*np.sum(stats['P']))
        stats['ACC%'] = stats['ACC'] * 100.0
        if chisquare:
            # indep_rows to assure reasonable handling of disbalanced
            # cases
            stats['CHI^2'] = chisquare(self.__matrix, exp='indep_rows')
        #
        # ROC computation if available
        ROC = ROCCurve(labels=labels, sets=self.sets)
        aucs = ROC.aucs
        if len(aucs)>0:
            stats['AUC'] = aucs
            if len(aucs) != Nlabels:
                raise RuntimeError, \
                      "We must got a AUC per label. Got %d instead of %d" % \
                      (len(aucs), Nlabels)
            self.ROC = ROC
        else:
            # we don't want to provide ROC if it is bogus
            stats['AUC'] = [np.nan] * Nlabels
            self.ROC = None


        # compute mean stats
        for k,v in stats.items():
            stats['mean(%s)' % k] = np.mean(v)

        self._stats.update(stats)


    ##REF: Name was automagically refactored
    def as_string(self, short=False, header=True, summary=True,
                 description=False):
        """'Pretty print' the matrix

        Parameters
        ----------
        short : bool
          if True, ignores the rest of the parameters and provides consise
          1 line summary
        header : bool
          print header of the table
        summary : bool
          print summary (accuracy)
        description : bool
          print verbose description of presented statistics
        """
        if len(self.sets) == 0:
            return "Empty"

        self.compute()

        # some shortcuts
        labels = self.__labels
        labels_map_rev = self.__labels_map_rev
        matrix = self.__matrix

        labels_rev = []
        if labels_map_rev is not None:
            labels_rev = [','.join([str(x) for x in labels_map_rev[l]])
                                   for l in labels]

        out = StringIO()
        # numbers of different entries
        Nlabels = len(labels)
        Nsamples = self.__Nsamples.astype(int)

        stats = self._stats
        if short:
            return "%(# of sets)d sets %(# of labels)d labels " \
                   " ACC:%(ACC).2f" \
                   % stats

        Ndigitsmax = int(ceil(log10(max(Nsamples))))
        Nlabelsmax = max( [len(str(x)) for x in labels] )

        # length of a single label/value
        L = max(Ndigitsmax+2, Nlabelsmax) #, len("100.00%"))
        res = ""

        stats_perpredict = ["P'", "N'", 'FP', 'FN', 'PPV', 'NPV', 'TPR',
                            'SPC', 'FDR', 'MCC']
        # print AUC only if ROC was computed
        if self.ROC is not None: stats_perpredict += [ 'AUC' ]
        stats_pertarget = ['P', 'N', 'TP', 'TN']
        stats_summary = ['ACC', 'ACC%', '# of sets']


        #prefixlen = Nlabelsmax + 2 + Ndigitsmax + 1
        prefixlen = Nlabelsmax + 1
        pref = ' '*(prefixlen) # empty prefix

        if matrix.shape != (Nlabels, Nlabels):
            raise ValueError, \
                  "Number of labels %d doesn't correspond the size" + \
                  " of a confusion matrix %s" % (Nlabels, matrix.shape)

        # list of lists of what is printed
        printed = []
        underscores = [" %s" % ("-" * L)] * Nlabels
        if header:
            # labels
            printed.append(['@l----------.        '] + labels_rev)
            printed.append(['@lpredictions\\targets'] + labels)
            # underscores
            printed.append(['@l            `------'] \
                           + underscores + stats_perpredict)

        # matrix itself
        for i, line in enumerate(matrix):
            l = labels[i]
            if labels_rev != []:
                l = '@r%10s / %s' % (labels_rev[i], l)
            printed.append(
                [l] +
                [ str(x) for x in line ] +
                [ _p2(stats[x][i]) for x in stats_perpredict])

        if summary:
            ## Various alternative schemes ;-)
            # printed.append([''] + underscores)
            # printed.append(['@lPer target \ Means:'] + underscores + \
            #               [_p2(x) for x in mean_stats])
            # printed.append(['Means:'] + [''] * len(labels)
            #                + [_p2(x) for x in mean_stats])
            printed.append(['@lPer target:'] + underscores)
            for stat in stats_pertarget:
                printed.append([stat] + [
                    _p2(stats[stat][i]) for i in xrange(Nlabels)])

            # compute mean stats
            # XXX refactor to expose them in stats as well, as
            #     mean(FCC)
            mean_stats = np.mean(np.array([stats[k] for k in stats_perpredict]),
                                axis=1)
            printed.append(['@lSummary \ Means:'] + underscores
                           + [_p2(stats['mean(%s)' % x])
                              for x in stats_perpredict])

            if 'CHI^2' in self.stats:
                chi2t = stats['CHI^2']
                printed.append(['CHI^2'] + [_p2(chi2t[0])]
                               + ['p:'] + ['%.2g' % chi2t[1]])

            for stat in stats_summary:
                printed.append([stat] + [_p2(stats[stat])])

        table2string(printed, out)

        if description:
            out.write("\nStatistics computed in 1-vs-rest fashion per each " \
                      "target.\n")
            out.write("Abbreviations (for details see " \
                      "http://en.wikipedia.org/wiki/ROC_curve):\n")
            for d, val, eq in self._STATS_DESCRIPTION:
                out.write(" %-3s: %s\n" % (d, val))
                if eq is not None:
                    out.write("      " + eq + "\n")

        #out.write("%s" % printed)
        result = out.getvalue()
        out.close()
        return result


    def plot(self, labels=None, numbers=False, origin='upper',
             numbers_alpha=None, xlabels_vertical=True, numbers_kwargs={},
             **kwargs):
        """Provide presentation of confusion matrix in image

        Parameters
        ----------
        labels : list of int or str
          Optionally provided labels guarantee the order of
          presentation. Also value of None places empty column/row,
          thus provides visual groupping of labels (Thanks Ingo)
        numbers : bool
          Place values inside of confusion matrix elements
        numbers_alpha : None or float
          Controls textual output of numbers. If None -- all numbers
          are plotted in the same intensity. If some float -- it controls
          alpha level -- higher value would give higher contrast. (good
          value is 2)
        origin : str
          Which left corner diagonal should start
        xlabels_vertical : bool
          Either to plot xlabels vertical (benefitial if number of labels
          is large)
        numbers_kwargs : dict
          Additional keyword parameters to be added to numbers (if numbers
          is True)
        **kwargs
          Additional arguments given to imshow (\eg me cmap)

        Returns
        -------
         (fig, im, cb) -- figure, imshow, colorbar
        """

        externals.exists("pylab", raise_=True)
        import pylab as pl

        self.compute()
        labels_order = labels

        # some shortcuts
        labels = self.__labels
        labels_map = self.__labels_map
        labels_map_rev = self.__labels_map_rev
        matrix = self.__matrix

        # craft original mapping from a label into index in the matrix
        labels_indexes = dict([(x,i) for i,x in enumerate(labels)])

        labels_rev = []
        if labels_map_rev is not None:
            labels_rev = [','.join([str(x) for x in labels_map_rev[l]])
                                   for l in labels]
            labels_map_full = dict(zip(labels_rev, labels))

        if labels_order is not None:
            labels_order_filtered = filter(lambda x:x is not None, labels_order)
            labels_order_filtered_set = Set(labels_order_filtered)
            # Verify if all labels provided in labels
            if Set(labels) == labels_order_filtered_set:
                # We were provided numerical (most probably) set
                labels_plot = labels_order
            elif len(labels_rev) \
                     and Set(labels_rev) == labels_order_filtered_set:
                # not clear if right whenever there were multiple labels
                # mapped into the same
                labels_plot = []
                for l in labels_order:
                    v = None
                    if l is not None: v = labels_map_full[l]
                    labels_plot += [v]
            else:
                raise ValueError, \
                      "Provided labels %s do not match set of known " \
                      "original labels (%s) or mapped labels (%s)" % \
                      (labels_order, labels, labels_rev)
        else:
            labels_plot = labels

        # where we have Nones?
        isempty = np.array([l is None for l in labels_plot])
        non_empty = np.where(np.logical_not(isempty))[0]
        # numbers of different entries
        NlabelsNN = len(non_empty)
        Nlabels = len(labels_plot)

        if matrix.shape != (NlabelsNN, NlabelsNN):
            raise ValueError, \
                  "Number of labels %d doesn't correspond the size" + \
                  " of a confusion matrix %s" % (NlabelsNN, matrix.shape)

        confusionmatrix = np.zeros((Nlabels, Nlabels))
        mask = confusionmatrix.copy()
        ticks = []
        tick_labels = []
        # populate in a silly way
        reordered_indexes = [labels_indexes[i] for i in labels_plot
                             if i is not None]
        for i, l in enumerate(labels_plot):
            if l is not None:
                j = labels_indexes[l]
                confusionmatrix[i, non_empty] = matrix[j, reordered_indexes]
                confusionmatrix[non_empty, i] = matrix[reordered_indexes, j]
                ticks += [i + 0.5]
                if labels_map_rev is not None:
                    tick_labels += ['/'.join(labels_map_rev[l])]
                else:
                    tick_labels += [str(l)]
            else:
                mask[i, :] = mask[:, i] = 1

        confusionmatrix = np.ma.MaskedArray(confusionmatrix, mask=mask)

        # turn off automatic update if interactive
        if pl.matplotlib.get_backend() == 'TkAgg':
            pl.ioff()

        fig = pl.gcf()
        ax = pl.gca()
        ax.axis('off')

        # some customization depending on the origin
        xticks_position, yticks, ybottom = {
            'upper': ('top', [Nlabels-x for x in ticks], 0.1),
            'lower': ('bottom', ticks, 0.2)
            }[origin]


        # Plot
        axi = fig.add_axes([0.15, ybottom, 0.7, 0.7])
        im = axi.imshow(confusionmatrix, interpolation="nearest", origin=origin,
                        aspect='equal', extent=(0, Nlabels, 0, Nlabels),
                        **kwargs)

        # plot numbers
        if numbers:
            numbers_kwargs_ = {'fontsize': 10,
                               'horizontalalignment': 'center',
                               'verticalalignment': 'center'}
            maxv = float(np.max(confusionmatrix))
            colors = [im.to_rgba(0), im.to_rgba(maxv)]
            for i,j in zip(*np.logical_not(mask).nonzero()):
                v = confusionmatrix[j, i]
                # scale alpha non-linearly
                if numbers_alpha is None:
                    alpha = 1.0
                else:
                    # scale according to value
                    alpha = 1 - np.array(1 - v / maxv) ** numbers_alpha
                y = {'lower':j, 'upper':Nlabels-j-1}[origin]
                numbers_kwargs_['color'] = colors[int(v<maxv/2)]
                numbers_kwargs_.update(numbers_kwargs)
                pl.text(i+0.5, y+0.5, '%d' % v, alpha=alpha, **numbers_kwargs_)

        maxv = np.max(confusionmatrix)
        boundaries = np.linspace(0, maxv, np.min((maxv, 10)), True)

        # Label axes
        pl.xlabel("targets")
        pl.ylabel("predictions")

        pl.setp(axi, xticks=ticks, yticks=yticks,
               xticklabels=tick_labels, yticklabels=tick_labels)

        axi.xaxis.set_ticks_position(xticks_position)
        axi.xaxis.set_label_position(xticks_position)

        if xlabels_vertical:
            pl.setp(pl.getp(axi, 'xticklabels'), rotation='vertical')

        axcb = fig.add_axes([0.8, ybottom, 0.02, 0.7])
        cb = pl.colorbar(im, cax=axcb, format='%d', ticks = boundaries)

        if pl.matplotlib.get_backend() == 'TkAgg':
            pl.ion()
        pl.draw()
        # Store it primarily for testing
        self._plotted_confusionmatrix = confusionmatrix
        return fig, im, cb


    @property
    def error(self):
        self.compute()
        return 1.0-self.__Ncorrect*1.0/sum(self.__Nsamples)


    @property
    def labels(self):
        self.compute()
        return self.__labels


    ##REF: Name was automagically refactored
    def get_labels_map(self):
        return self.__labels_map


    ##REF: Name was automagically refactored
    def set_labels_map(self, val):
        if val is None or isinstance(val, dict):
            self.__labels_map = val
        else:
            raise ValueError, "Cannot set labels_map to %s" % val
        # reset it just in case
        self.__labels_map_rev = None
        self._computed = False


    @property
    def matrix(self):
        self.compute()
        return self.__matrix


    @property
    ##REF: Name was automagically refactored
    def percent_correct(self):
        self.compute()
        return 100.0*self.__Ncorrect/sum(self.__Nsamples)

    labels_map = property(fget=get_labels_map, fset=set_labels_map)


class RegressionStatistics(SummaryStatistics):
    """Class to contain information and display on regression results.

    """

    _STATS_DESCRIPTION = (
        ('CCe', 'Error based on correlation coefficient',
         '1 - corr_coef'),
        ('CCp', 'Correlation coefficient (p-value)', None),
        ('RMSE', 'Root mean squared error', None),
        ('STD', 'Standard deviation', None),
        ('RMP', 'Root mean power (compare to RMSE of results)',
         'sqrt(mean( data**2 ))'),
        ) + SummaryStatistics._STATS_DESCRIPTION


    def __init__(self, **kwargs):
        """Initialize RegressionStatistics

        Parameters
        ----------
        targets
         Optional set of targets
        predictions
         Optional set of predictions
         """

        SummaryStatistics.__init__(self, **kwargs)


    def _compute(self):
        """Actually compute the confusion matrix based on all the sets"""

        super(RegressionStatistics, self)._compute()
        sets = self.sets
        Nsets = len(sets)

        stats = {}

        funcs = {
            'RMP_t': lambda p,t:root_mean_power_fx(t),
            'STD_t': lambda p,t:np.std(t),
            'RMP_p': lambda p,t:root_mean_power_fx(p),
            'STD_p': lambda p,t:np.std(p),
            'CCe': CorrErrorFx(),
            'CCp': CorrErrorPFx(),
            'RMSE': RMSErrorFx(),
            'RMSE/RMP_t': RelativeRMSErrorFx()
            }

        for funcname, func in funcs.iteritems():
            funcname_all = funcname + '_all'
            stats[funcname_all] = []
            for i, (targets, predictions, estimates) in enumerate(sets):
                stats[funcname_all] += [func(predictions, targets)]
            stats[funcname_all] = np.array(stats[funcname_all])
            stats[funcname] = np.mean(stats[funcname_all])
            stats[funcname+'_std'] = np.std(stats[funcname_all])
            stats[funcname+'_max'] = np.max(stats[funcname_all])
            stats[funcname+'_min'] = np.min(stats[funcname_all])

        # create ``summary`` statistics, since some per-set statistics
        # might be uncomputable if a set contains just a single number
        # (like in the case of correlation coefficient)
        targets, predictions = [], []
        for i, (targets_, predictions_, estimates_) in enumerate(sets):
            targets += list(targets_)
            predictions += list(predictions_)

        for funcname, func in funcs.iteritems():
            funcname_all = 'Summary ' + funcname
            stats[funcname_all] = func(predictions, targets)

        self._stats.update(stats)


    def plot(self,
             plot=True, plot_stats=True,
             splot=True
             #labels=None, numbers=False, origin='upper',
             #numbers_alpha=None, xlabels_vertical=True,
             #numbers_kwargs={},
             #**kwargs
             ):
        """Provide presentation of regression performance in image

        Parameters
        ----------
        plot : bool
          Plot regular plot of values (targets/predictions)
        plot_stats : bool
          Print basic statistics in the title
        splot : bool
          Plot scatter plot

        Returns
        -------
         (fig, im, cb) -- figure, imshow, colorbar
        """
        externals.exists("pylab", raise_=True)
        import pylab as pl

        self.compute()
        # total number of plots
        nplots = plot + splot

        # turn off automatic update if interactive
        if pl.matplotlib.get_backend() == 'TkAgg':
            pl.ioff()

        fig = pl.gcf()
        pl.clf()
        sps = []                        # subplots

        nplot = 0
        if plot:
            nplot += 1
            sps.append(pl.subplot(nplots, 1, nplot))
            xstart = 0
            lines = []
            for s in self.sets:
                nsamples = len(s[0])
                xend = xstart+nsamples
                xs = xrange(xstart, xend)
                lines += [pl.plot(xs, s[0], 'b')]
                lines += [pl.plot(xs, s[1], 'r')]
                # vertical line
                pl.plot([xend, xend], [np.min(s[0]), np.max(s[0])], 'k--')
                xstart = xend
            if len(lines)>1:
                pl.legend(lines[:2], ('Target', 'Prediction'))
            if plot_stats:
                pl.title(self.as_string(short='very'))

        if splot:
            nplot += 1
            sps.append(pl.subplot(nplots, 1, nplot))
            for s in self.sets:
                pl.plot(s[0], s[1], 'o',
                       markeredgewidth=0.2,
                       markersize=2)
            pl.gca().set_aspect('equal')

        if pl.matplotlib.get_backend() == 'TkAgg':
            pl.ion()
        pl.draw()

        return fig, sps

    ##REF: Name was automagically refactored
    def as_string(self, short=False, header=True,  summary=True,
                 description=False):
        """'Pretty print' the statistics"""

        if len(self.sets) == 0:
            return "Empty"

        self.compute()

        stats = self.stats

        if short:
            if short == 'very':
                # " RMSE/RMP_t:%(RMSE/RMP_t).2f" \
                return "%(# of sets)d sets CCe=%(CCe).2f p=%(CCp).2g" \
                       " RMSE:%(RMSE).2f" \
                       " Summary (stacked data): " \
                       "CCe=%(Summary CCe).2f p=%(Summary CCp).2g" \
                       % stats
            else:
                return "%(# of sets)d sets CCe=%(CCe).2f+-%(CCe_std).3f" \
                       " RMSE=%(RMSE).2f+-%(RMSE_std).3f" \
                       " RMSE/RMP_t=%(RMSE/RMP_t).2f+-%(RMSE/RMP_t_std).3f" \
                       % stats

        stats_data = ['RMP_t', 'STD_t', 'RMP_p', 'STD_p']
        # CCp needs tune up of format so excluded
        stats_ = ['CCe', 'RMSE', 'RMSE/RMP_t']
        stats_summary = ['# of sets']

        out = StringIO()

        printed = []
        if header:
            # labels
            printed.append(['Statistics', 'Mean', 'Std', 'Min', 'Max'])
            # underscores
            printed.append(['----------', '-----', '-----', '-----', '-----'])

        def print_stats(printed, stats_):
            # Statistics itself
            for stat in stats_:
                s = [stat]
                for suffix in ['', '_std', '_min', '_max']:
                    s += [ _p2(stats[stat+suffix], 3) ]
                printed.append(s)

        printed.append(["Data:     "])
        print_stats(printed, stats_data)
        printed.append(["Results:  "])
        print_stats(printed, stats_)
        printed.append(["Summary:  "])
        printed.append(["CCe", _p2(stats['Summary CCe']), "", "p=", '%g' % stats['Summary CCp']])
        printed.append(["RMSE", _p2(stats['Summary RMSE'])])
        printed.append(["RMSE/RMP_t", _p2(stats['Summary RMSE/RMP_t'])])

        if summary:
            for stat in stats_summary:
                printed.append([stat] + [_p2(stats[stat])])

        table2string(printed, out)

        if description:
            out.write("\nDescription of printed statistics.\n"
                      " Suffixes: _t - targets, _p - predictions\n")

            for d, val, eq in self._STATS_DESCRIPTION:
                out.write(" %-3s: %s\n" % (d, val))
                if eq is not None:
                    out.write("      " + eq + "\n")

        result = out.getvalue()
        out.close()
        return result


    @property
    def error(self):
        self.compute()
        return self.stats['RMSE']



class ClassifierError(ClassWithCollections):
    """Compute (or return) some error of a (trained) classifier on a dataset.
    """

    confusion = ConditionalAttribute(enabled=False)
    """TODO Think that labels might be also symbolic thus can't directly
       be indicies of the array
    """

    training_confusion = ConditionalAttribute(enabled=False,
        doc="Proxy training_confusion from underlying classifier.")


    def __init__(self, clf, labels=None, train=True, **kwargs):
        """Initialization.

        Parameters
        ----------
        clf : Classifier
          Either trained or untrained classifier
        labels : list
          if provided, should be a set of labels to add on top of the
          ones present in testdata
        train : bool
          unless train=False, classifier gets trained if
          trainingdata provided to __call__
        """
        ClassWithCollections.__init__(self, **kwargs)
        self.__clf = clf

        self._labels = labels
        """Labels to add on top to existing in testing data"""

        self.__train = train
        """Either to train classifier if trainingdata is provided"""


    __doc__ = enhanced_doc_string('ClassifierError', locals(), ClassWithCollections)


    def __copy__(self):
        """TODO: think... may be we need to copy self.clf"""
        out = ClassifierError.__new__(TransferError)
        ClassifierError.__init__(out, self.clf)
        return out


    def _precall(self, testdataset, trainingdataset=None):
        """Generic part which trains the classifier if necessary
        """
        if not trainingdataset is None:
            if self.__train:
                # XXX can be pretty annoying if triggered inside an algorithm
                # where it cannot be switched of, but retraining might be
                # intended or at least not avoidable.
                # Additonally is_trained docs say:
                #   MUST BE USED WITH CARE IF EVER
                #
                # switching it off for now
                #if self.__clf.is_trained(trainingdataset):
                #    warning('It seems that classifier %s was already trained' %
                #            self.__clf + ' on dataset %s. Please inspect' \
                #                % trainingdataset)
                if self.ca.is_enabled('training_confusion'):
                    self.__clf.ca.change_temporarily(
                        enable_ca=['training_confusion'])
                self.__clf.train(trainingdataset)
                if self.ca.is_enabled('training_confusion'):
                    self.ca.training_confusion = \
                        self.__clf.ca.training_confusion
                    self.__clf.ca.reset_changed_temporarily()

        if self.__clf.ca.is_enabled('trained_targets') \
               and not self.__clf.__is_regression__ \
               and not testdataset is None:
            newlabels = Set(testdataset.sa[self.clf.params.targets_attr].unique) \
                        - Set(self.__clf.ca.trained_targets)
            if len(newlabels)>0:
                warning("Classifier %s wasn't trained to classify labels %s" %
                        (self.__clf, newlabels) +
                        " present in testing dataset. Make sure that you have" +
                        " not mixed order/names of the arguments anywhere")

        ### Here checking for if it was trained... might be a cause of trouble
        # XXX disabled since it is unreliable.. just rely on explicit
        # self.__train
        #    if  not self.__clf.is_trained(trainingdataset):
        #        self.__clf.train(trainingdataset)
        #    elif __debug__:
        #        debug('CERR',
        #              'Not training classifier %s since it was ' % `self.__clf`
        #              + ' already trained on dataset %s' % `trainingdataset`)


    def _call(self, testdataset, trainingdataset=None):
        raise NotImplementedError


    def _postcall(self, testdataset, trainingdataset=None, error=None):
        pass


    def __call__(self, testdataset, trainingdataset=None):
        """Compute the transfer error for a certain test dataset.

        If `trainingdataset` is not `None` the classifier is trained using the
        provided dataset before computing the transfer error. Otherwise the
        classifier is used in it's current state to make the predictions on
        the test dataset.

        Returns a scalar value of the transfer error.
        """
        self._precall(testdataset, trainingdataset)
        error = self._call(testdataset, trainingdataset)
        self._postcall(testdataset, trainingdataset, error)
        if __debug__:
            debug('CERR', 'Classifier error on %s: %.2f'
                  % (testdataset, error))
        return error


    def untrain(self):
        """Untrain the `*Error` which relies on the classifier
        """
        self.clf.untrain()


    @property
    def clf(self):
        return self.__clf


    @property
    def labels(self):
        return self._labels



class TransferError(ClassifierError):
    """Compute the transfer error of a (trained) classifier on a dataset.

    The actual error value is computed using a customizable error function.
    Optionally the classifier can be trained by passing an additional
    training dataset to the __call__() method.
    """

    null_prob = ConditionalAttribute(enabled=True,
                    doc="Stores the probability of an error result under "
                         "the NULL hypothesis")
    samples_error = ConditionalAttribute(enabled=False,
                        doc="Per sample errors computed by invoking the "
                            "error function for each sample individually. "
                            "Errors are available in a dictionary with each "
                            "samples origid as key.")

    def __init__(self, clf, errorfx=None, labels=None,
                 null_dist=None, samples_idattr='origids', **kwargs):
        """Initialization.

        Parameters
        ----------
        clf : Classifier
          Either trained or untrained classifier
        errorfx: func, optional
          Functor that computes a scalar error value from the vectors of
          desired and predicted values (e.g. subclass of `ErrorFunction`).
          If None, then MeanMismatchErrorFx is chosen for classifiers and
          CorrErrorFx for regressions
        labels : list, optional
          If provided, should be a set of labels to add on top of the
          ones present in testdata
        null_dist : instance of distribution estimator, optional
        samples_idattr : str, optional
          What samples attribute to use to identify and store samples_errors
          conditional attribute
        """
        ClassifierError.__init__(self, clf, labels, **kwargs)
        if errorfx is None:
            errorfx = {False: MeanMismatchErrorFx,
                       True: CorrErrorFx}[clf.__is_regression__]()
        self.__errorfx = errorfx
        self.__null_dist = auto_null_dist(null_dist)
        self.__samples_idattr = samples_idattr


    __doc__ = enhanced_doc_string('TransferError', locals(), ClassifierError)


    def __copy__(self):
        """Performs deepcopying of the classifier."""
        # TODO -- use ClassifierError.__copy__
        out = TransferError.__new__(TransferError)
        TransferError.__init__(out, self.clf.clone(),
                               self.errorfx, self._labels)

        return out

    # XXX: TODO: unify naming? test/train or with ing both
    def _call(self, testdataset, trainingdataset=None):
        """Compute the transfer error for a certain test dataset.

        If `trainingdataset` is not `None` the classifier is trained using the
        provided dataset before computing the transfer error. Otherwise the
        classifier is used in it's current state to make the predictions on
        the test dataset.

        Returns a scalar value of the transfer error.
        """
        testtargets = testdataset.sa[self.clf.params.targets_attr].value
        # OPT: local binding
        clf = self.clf
        if testdataset is None:
            # We cannot do anythin, but we can try to figure out WTF and
            # warn the user accordingly in some usecases
            import traceback as tb
            filenames = [x[0] for x in tb.extract_stack(limit=100)]
            rfe_matches = [f for f in filenames if f.endswith('/rfe.py')]
            cv_matches = [f for f in filenames if
                          f.endswith('cvtranserror.py')]
            msg = ""
            if len(rfe_matches) > 0 and len(cv_matches):
                msg = " It is possible that you used RFE with stopping " \
                      "criterion based on the TransferError and directly" \
                      " from CrossValidatedTransferError, such approach" \
                      " would require exposing testing dataset " \
                      " to the classifier which might heavily bias " \
                      " generalization performance estimate. If you are " \
                      " sure to use it that way, create CVTE with " \
                      " parameter expose_testdataset=True"
            raise ValueError, "Transfer error call obtained None " \
                  "as a dataset for testing.%s" % msg
        #clf should handle dataset or samples
        predictions = clf.predict(testdataset)
        # compute confusion matrix
        # Should it migrate into ClassifierError.__postcall?
        # -> Probably not because other childs could estimate it
        #  not from test/train datasets explicitely, see
        #  `ConfusionBasedError`, whereca. confusion is simply
        #  bound to classifiers confusion matrix
        ca = self.ca
        if ca.is_enabled('confusion'):
            confusion = clf.__summary_class__(
                #labels = self.targets,
                targets = testtargets,
                predictions = predictions,
                estimates = clf.ca.get('estimates', None))
            ca.confusion = confusion

        if ca.is_enabled('samples_error'):
            samples_error = []
            for i, p in enumerate(predictions):
                samples_error.append(
                    self.__errorfx([p], testtargets[i:i+1]))
            testdataset.init_origids(
                'samples', attr=self.__samples_idattr, mode='existing')
            ca.samples_error = dict(
                zip(testdataset.sa[self.__samples_idattr].value,
                    samples_error))

        # compute error from desired and predicted values
        error = self.__errorfx(predictions, testtargets)

        return error


    def _postcall(self, vdata, wdata=None, error=None):
        """
        """
        # estimate the NULL distribution when functor and training data is
        # given
        if not self.__null_dist is None and not wdata is None:
            # we need a matching transfer error instances (e.g. same error
            # function), but we have to disable the estimation of the null
            # distribution in that child to prevent infinite looping.
            null_terr = copy.copy(self)
            null_terr.__null_dist = None
            self.__null_dist.fit(null_terr, wdata, vdata)


        # get probability of error under NULL hypothesis if available
        if not error is None and not self.__null_dist is None:
            self.ca.null_prob = self.__null_dist.p(error)


    @property
    def errorfx(self):
        return self.__errorfx

    @property
    def null_dist(self):
        return self.__null_dist



class ConfusionBasedError(ClassifierError):
    """For a given classifier report an error based on internally
    computed error measure (given by some `ConfusionMatrix` stored in
    some conditional attribute of `Classifier`).

    This way we can perform feature selection taking as the error
    criterion either learning error, or transfer to splits error in
    the case of SplitClassifier
    """

    def __init__(self, clf, labels=None, confusion_state="training_confusion",
                 **kwargs):
        """Initialization.

        Parameters
        ----------
        clf : Classifier
          Either trained or untrained classifier
        confusion_state
          Id of the conditional attribute which stores `ConfusionMatrix`
        labels : list
          if provided, should be a set of labels to add on top of the
          ones present in testdata
        """
        ClassifierError.__init__(self, clf, labels, **kwargs)

        self.__confusion_state = confusion_state
        """What state to extract from"""

        if not clf.ca.has_key(confusion_state):
            raise ValueError, \
                  "Conditional attribute %s is not defined for classifier %r" % \
                  (confusion_state, clf)
        if not clf.ca.is_enabled(confusion_state):
            if __debug__:
                debug('CERR', "Forcing state %s to be enabled for %r" %
                      (confusion_state, clf))
            clf.ca.enable(confusion_state)


    __doc__ = enhanced_doc_string('ConfusionBasedError', locals(),
                                ClassifierError)


    def _call(self, testdata, trainingdata=None):
        """Extract transfer error. Nor testdata, neither trainingdata is used
        """
        confusion = self.clf.ca[self.__confusion_state].value
        self.ca.confusion = confusion
        return confusion.error
