#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: t -*- 
#ex: set sts=4 ts=4 sw=4 noet:
#------------------------- =+- Python script -+= -------------------------
"""
 LICENSE:

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the 
  Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
  MA 02110-1301, USA.

 On Debian system see /usr/share/common-licenses/GPL for the full license.
"""
#-----------------\____________________________________/------------------


from mvpa.maskeddataset import MaskedDataset
from mvpa.splitter import NFoldSplitter
from mvpa.crossval import CrossValidation
from mvpa import libsvm
from mvpa.svm import SVM

# fmri    is an input 4d ndarray  TxZxYxX
# labels  is                      Tx1
# runs    is                      Tx1
# brainmask                       ZxYxX
data = Dataset( fmri, labels, None, brainmask)

splitter = NFoldSplitter()

classifer = SVM(kernel_type=libsvm.LINEAR, svm_type=libsvm.NU_SVC)

postError = RMSESplitProcessing()

crossval = CrossValidation(splitter, classifer, [postError])

result = crossval(data)
