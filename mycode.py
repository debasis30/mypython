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

def main():
    # Help texts.
    description = ('Evaluate the given feature representation for license '
                    'plate recognition using signature matching. Signature '
                    'matching proceeds via a nearest-neighbour assignment '
                    'of a test image signature to one of the training set '
                    'signatures, along with a confidence value.')
    train_help = ('Files containing the signatures of the training set '
                    '(CSV files), along with their corresponding labels.')
    test_help = ('Files containing the signatures of the test set '
                    '(CSV files), along with their corresponding labels.')
    output_help = ('The name to give to the output file.')
    metric_help = ('The metric to apply for computing distances.'
                    'Can be one of : cos, l1, l2, l<p> where p is a float.')
    normalise_help = ('Whether to mean-centre the data and normalise all '
                        'feature variances to 1.')

    # Creating the command-line parser.
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--signatures', nargs=2, metavar=('SIG','LABELS'),
                        help=train_help)
    parser.add_argument('-t', '--test', nargs=2, metavar=('SIG', 'LABELS'),
                        help=test_help)
    parser.add_argument('-o', '--output', nargs=1, help=output_help)
    parser.add_argument('-m', '--metric', nargs=1, default=('cos',),
                        help=metric_help)
    parser.add_argument('-n', '--normalise-features', action='store_true',
                        help=normalise_help)

    # Parsing the command-line arguments.
    # print 'Parsing the command-line arguments...'
    # IMPORTANT : do not place code with side effects before the following line.
    # Code up to here executes every time we try to autocomplete.
    # https://pypi.python.org/pypi/argcomplete/1.1.1 
    #   See 'Side effects'
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    signature_fv_file = args.signatures[0]
    signature_labels_file = args.signatures[1]
    test_fv_file = args.test[0]
    test_labels_file = args.test[1]
    output_name, = args.output
    metric, = args.metric
    normalise_features = args.normalise_features

    # Read in the signatures.
    print 'Reading in the signatures...'
    signature_fvs = np.loadtxt(signature_fv_file, delimiter=DELIMITER)
    with open(signature_labels_file) as labels_file:
        signature_labels = [l.strip() for l in labels_file.readlines() 
                                      if len(l.strip()) > 0]
    assert len(signature_labels) == signature_fvs.shape[0]
    
    # Read in the test data.
    print 'Reading in the test data...'
    test_fvs = np.loadtxt(test_fv_file, delimiter=DELIMITER)
    with open(test_labels_file) as labels_file:
        test_labels = [l.strip() for l in labels_file.readlines() 
                                 if len(l.strip()) > 0]
    assert len(test_labels) == test_fvs.shape[0]

    assert test_fvs.shape[1] == signature_fvs.shape[1]
    n_dims = signature_fvs.shape[1]
    n_lps = signature_fvs.shape[0]
    n_test = test_fvs.shape[0]

    # Mean-centre and normalise the variance of each feature vector to 1.
    if normalise_features:
        print 'Normalising feature variances to 1...'
        mean_signature = np.mean(signature_fvs, axis=0).reshape((1, n_dims))
        signature_fvs -= mean_signature

        # training_examples are already mean-centred now. So all values are
        # deviations.
        stdev_signature = np.linalg.norm(signature_fvs, axis=0)\
                            .reshape((1, n_dims))
        stdev_signature /= np.sqrt(n_lps)

        # should force bad components to 0 rather than blowing up to infinity.
        stdev_signature[stdev_signature == 0.0] = 1.0e15
        signature_fvs /= stdev_signature

        # Do the same normalisation for the test set as well.
        test_fvs -= mean_signature
        test_fvs /= stdev_signature

    # Run the nearest-neighbour matching.
    print 'Running nearest-neighbour matching with %s metric...' % metric

    matches = match_signatures((signature_fvs, signature_labels),
                               test_fvs, metric)

    # Set up the output
    if os.path.isabs(output_name):
        output_filename = output_name
    else:
        output_dir = os.path.join(benchmark_dir, 'classify_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir,
                                       '%s_%d.csv' % (output_name,
                                                      int(time.time())))
    
    # Look at the matching.
    print 'Outputting matching results...'
    start = datetime.datetime.now()
    log_interval = n_test // 10 # Print every 10% of the way.
    with open(output_filename, 'w') as output:
        n_processed = 0
        n_correct = 0
        output.write('assigned,min_dist,true_label\n')
        for match, true_label in itertools.izip(matches, test_labels):
            assigned, min_dist = match
            output.write(','.join([assigned, str(min_dist), true_label]))
            output.write('\n')

            n_processed += 1
            if assigned == true_label:
                n_correct += 1

            if n_processed % log_interval == 0:
                now = datetime.datetime.now()
                print '%s %d/%d (%.2f%%) processed. Overall accuracy %.2f%%' \
                        % (str(now - start),
                           n_processed, n_test,
                           (100. * n_processed) / n_test,
                           (100. * n_correct) / n_processed)
    print 'Done.'

    # Accuracy output.
    accuracy = float(n_correct) / n_test
    print 'Accuracy : %.4f' % accuracy

if __name__ == '__main__':
    main()
