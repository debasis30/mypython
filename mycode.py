#!/usr/bin/env python
#PYTHON_ARGCOMPLETE_OK

""" classify_image_signatures.py
    
    The one script to evaluate all signature computation schemes.

    ./classify_image_signatures.py -s $templateCsv $templateLabels -t $testCsv
    $testLabels -o $outputFile -m cos -n
"""

import os
import sys
import argparse
import argcomplete
import datetime
import time
import itertools

import numpy as np

DELIMITER = ','
benchmark_dir = './benchmark'

def match_signatures(signatures, test_fvs, metric):
    signature_fvs, signature_labels = signatures
    N = len(signature_labels)

    if metric == 'cos':
        norms_train = np.linalg.norm(signature_fvs, axis=1)
        norms_test = np.linalg.norm(test_fvs, axis=1)
    
    N_test = test_fvs.shape[0]
    for i in range(N_test):
        fv = test_fvs[i,:]

        if metric == 'l1':
            d = np.sum(np.abs(fv - signature_fvs), axis=1)
        elif metric == 'l2':
            d = np.sqrt(np.sum((fv - signature_fvs) ** 2, axis=1))
        elif metric.startswith('l'):
            # TODO: improve this parsing with regex etc.
            frac_p = float(metric[1:])
            d = np.sum(np.abs(fv - signature_fvs) ** frac_p, axis=1)
            d = d ** (1.0 / frac_p)
        elif metric == 'cos':
            fv_norm = norms_test[i]
            c = np.sum(fv * signature_fvs, axis=1) / (fv_norm * norms_train)
            d = 1 - c
        else:
            raise NotImplementedError
        
        # Free up some memory...
        # np.delete(test_fvs, 0, axis=0)

        nearest = np.argmin(d)
        min_dist = d[nearest]
        yield (signature_labels[nearest], min_dist)
