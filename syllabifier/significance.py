# coding=utf-8

import re

try:
    import cpickle as pickle

except ImportError:
    import pickle

from compound import CONSTRAINTS, FinnSeg
from itertools import combinations
from maxent import MaxentInput
from numpy import mean
from os import sys, path
from scipy.spatial import KDTree
from scipy.stats import ttest_ind

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import finnsyll as finn

SEGMENTERS = [
    None,
    'OT',
    'Unviolable',
    'Maxent',
    'Maxent-exclLoans',
    ]

METRICS = [
    'P', 'R', 'F1', 'Acc',  # segmentation metrics
    'SyP', 'SyR', 'SyF1', 'Acc-S', 'Acc-C', 'Acc-T',  # syllabification metrics
    ]


class SignificancePartOne:

    def __init__(self):
        filename = 'data/test/sig/results-f10.pickle'

        # load results, or run the segmenters if the results are nonexistent
        try:
            self.results = pickle.load(open(filename))

        except IOError:
            self.results = {a: {m: [] for m in METRICS} for a in SEGMENTERS}
            self.run_segmenters()

            # pickle the results to file
            pickle.dump(
                self.results,
                open(filename, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL,
                )

    # Language modeling alone, Optimality Theory, and unviolable segmenters
    def run_segmenters(self):
        for fold in range(1, 11):
            print 'Fold 1.%s' % fold

            TRAINING = finn.exclude_fold(fold)
            VALIDATION = finn.get_fold(fold)

            filename = 'data/test/sig/morfessor-f' + str(fold)

            for approach in SEGMENTERS[:3]:
                F = FinnSeg(
                    training=TRAINING,
                    validation=VALIDATION,
                    filename=filename,
                    Print=False,
                    approach=approach,
                    )

                # segmentation performance
                self.results[approach]['P'].append(F.precision)
                self.results[approach]['R'].append(F.recall)
                self.results[approach]['F1'].append(F.f1)
                self.results[approach]['Acc'].append(F.accuracy)

                # syllabification performance
                self.results[approach]['SyP'].append(F.syll_precision)
                self.results[approach]['SyR'].append(F.syll_recall)
                self.results[approach]['SyF1'].append(F.syll_f1)
                self.results[approach]['Acc-S'].append(F.syll_simplex_acc)
                self.results[approach]['Acc-C'].append(F.syll_complex_acc)
                self.results[approach]['Acc-T'].append(F.syll_acc)

            # Maxent inputs
            for excl_loans in [True, False]:
                MaxentInput(
                    training=TRAINING,
                    validation=None,
                    excl_loans=excl_loans,
                    filename='f' + str(fold),
                    )


class SignificancePartTwo:

    def __init__(self):
        ttests_filename = 'data/test/sig/ttest-f10.txt'
        results_filename = 'data/test/sig/results-f10.pickle'

        # load the ttest results, or generate them if they are nonexistent
        try:
            with open(ttests_filename) as f:
                self.ttests = f.read()

            print self.ttests

        except IOError:
            # load the evaluation results, are throw an errow
            try:
                self.results = pickle.load(open(results_filename))

            except IOError:
                raise ValueError('Need to run Part I first and handle Maxent.')

            # finish curating the evaluation results
            if self.results['Maxent']['Acc-T'] == []:
                self.run_maxent_segmenters(results_filename)
            self.do_ttests(ttests_filename)

        # Maxent segmenters
    def run_maxent_segmenters(self, results_filename):
        for fold in range(1, 11):
            print 'Fold 1.%s' % fold

            TRAINING = finn.exclude_fold(fold)
            VALIDATION = finn.get_fold(fold)

            filename = 'data/test/sig/morfessor-f' + str(fold)

            for excl_loans in [True, False]:
                output = 'data/test/sig/maxent-f%s-4' % str(fold)
                output += '-exclLoans' if excl_loans else ''
                output += '-output.txt'

                with open(output) as f:
                    output = f.read()

                # extract weights: MnWrd, SonSeq, WrdFinal, V-Harmony, Ngram
                pattern = r'after optimization:\n(.+)\nInput'
                weights = re.search(pattern, output, flags=re.S)
                weights = weights.group(1).split('\n')
                weights = [float(w.split('\t')[-1]) for w in weights]

                Sum = sum(weights)
                weights = [w / Sum for w in weights]

                for i in xrange(len(CONSTRAINTS)):
                    CONSTRAINTS[i].weight = weights[i]

                F = FinnSeg(
                    training=TRAINING,
                    validation=VALIDATION,
                    filename=filename,
                    Print=False,
                    approach='Maxent',
                    constraints=CONSTRAINTS,
                    )

                approach = 'Maxent-exclLoans'if excl_loans else 'Maxent'

                # segmentation performance
                self.results[approach]['P'].append(F.precision)
                self.results[approach]['R'].append(F.recall)
                self.results[approach]['F1'].append(F.f1)
                self.results[approach]['Acc'].append(F.accuracy)

                # syllabification performance
                self.results[approach]['SyP'].append(F.syll_precision)
                self.results[approach]['SyR'].append(F.syll_recall)
                self.results[approach]['SyF1'].append(F.syll_f1)
                self.results[approach]['Acc-S'].append(F.syll_simplex_acc)
                self.results[approach]['Acc-C'].append(F.syll_complex_acc)
                self.results[approach]['Acc-T'].append(F.syll_acc)

        # pickle the updated results to file
        pickle.dump(
            self.results,
            open(results_filename, 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL,
            )

    def do_ttests(self, n, ttests_filename):
        titles = {None: 'LMA', }
        means = []
        ttests = []

        # get the mean performance of each segmenter on each metric
        for a in SEGMENTERS:
            performance = titles.get(a, a) + '\n'

            for m in METRICS:
                performance += '\t%s:\t\t%s\n' % (m, mean(self.results[a][m]))

            performance += '\n'

            means.append(performance)

        # compare each segmenter against every other segmenter on each metric
        for (a1, a2) in combinations(SEGMENTERS, 2):
            ttest = titles.get(a1, a1) + ' ~ ' + titles.get(a2, a2) + '\n'

            # Welch’s t-tests
            for m in METRICS:
                m1 = self.results[a1][m]
                m2 = self.results[a2][m]
                ttest += '\t%s:\t\t' % m
                ttest += 't = %6.4f\t\tp = %6.6f\n' % \
                    ttest_ind(m1, m2, equal_var=False)

            ttest += '\n'

            ttests.append(ttest)

        # alphabetize t-tests
        ttests.sort()

        # compose ttests text file
        header = '---- t-tests n%s ' % str(n) + '-' * 40
        self.ttests = header + '\n\n'
        self.ttests += ''.join(means)
        self.ttests += ''.join(ttests)
        self.ttests += '-' * len(header)

        with open(ttests_filename, 'w') as f:
            f.write(self.ttests)

        print self.ttests


def find_vector_nearest_to_mean(filename='data/test/sig/results-n50.pickle'):
    results = pickle.load(open(filename))
    acc_vectors = [[results[s]['Acc'][i] for s in SEGMENTERS] for i in range(50)]  # noqa
    mean_acc_vector = [mean(results[s]['Acc']) for s in SEGMENTERS]

    _, iteration = KDTree(acc_vectors).query(mean_acc_vector)
    iteration = int(iteration) + 1

    print 'Iteration:', iteration

    return iteration


def load_nearest_to_mean_segmenters():
    iteration = find_vector_nearest_to_mean()
    filename = 'data/test/sig/morfessor-n' + str(iteration)

    FULL_TRAINING = finn.full_training_set()
    TEST = finn.test_set()

    # language modeling alone
    FinnSeg(training=FULL_TRAINING, validation=TEST, filename=filename)

    # optimality theory
    FinnSeg(
        training=FULL_TRAINING,
        validation=TEST,
        filename=filename,
        approach='OT',
        )

    # unviolable
    FinnSeg(
        training=FULL_TRAINING,
        validation=TEST,
        filename=filename,
        approach='Unviolable',
        )

    # maxent
    for excl_loans in [False, True]:
        output = 'data/test/sig/maxent-n%s-4' % str(iteration)
        output += '-exclLoans' if excl_loans else ''
        output += '-output.txt'

        with open(output) as f:
            output = f.read()

        # extract weights: MnWrd, SonSeq, WrdFinal, V-Harmony, Ngram
        pattern = r'after optimization:\n(.+)\nInput'
        weights = re.search(pattern, output, flags=re.S)
        weights = weights.group(1).split('\n')
        weights = [float(w.split('\t')[-1]) for w in weights]

        Sum = sum(weights)
        weights = [w / Sum for w in weights]

        for i in xrange(len(CONSTRAINTS)):
            CONSTRAINTS[i].weight = weights[i]

        FinnSeg(
            training=FULL_TRAINING,
            validation=TEST,
            filename=filename,
            approach='Maxent',
            constraints=CONSTRAINTS,
            )


if __name__ == '__main__':
    SignificancePartOne()
    # SignificancePartTwo()

    # find_vector_nearest_to_mean()
    # load_nearest_to_mean_segmenters()

    finn.db.session.rollback()
