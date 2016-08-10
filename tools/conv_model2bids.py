# convert our old OpenFMRI model info into BIDS events.tsv files
import numpy as np

conds = {int(d['cond'][-1]): d['name']
         for d in np.recfromcsv(
             'mvpa2/data/haxby2001/models/model001/condition_key.txt',
             delimiter=' ',
             names=('task', 'cond', 'name'))}

for run in range(1, 13):
    evs = []
    for c in conds.keys():
        onsets = [l.split()[:2]
                  for l in open('mvpa2/data/haxby2001/sub001/model/model001/onsets/task001_run%.3i/cond%.3i.txt' % (run, c))]
        for o in onsets:
            evs.append((float(o[0]), float(o[1]), conds[c]))
    evs = sorted(evs, cmp=lambda x, y: cmp(x[0], y[0]))
    with open('mvpa2/data/haxby2001/sub001/BOLD/task001_run%.3i/bold_events.tsv' % run, 'w') as evfile:
        evfile.write('onset\tduration\ttrial_type\n')
        for ev in evs:
            evfile.write('%.1f\t%.1f\t%s\n' % ev)
