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
from scipy.stats import ttest_ind

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import finnsyll as finn

FULL_TRAINING = finn.full_training_set()
TEST = finn.test_set()

SEGMENTERS = [
    None,
    'OT',
    'Unviolable',
    'Maxent',
    'Maxent-exclLoans',
    ]

MEASURES = ['P', 'R', 'F1', 'Acc']


class SignificancePartOne:

    def __init__(self, n=1):
        filename = 'data/test/sig/results-n%s.pickle' % n

        # load results, or run the segmenters if the results are nonexistent
        try:
            self.results = pickle.load(open(filename))

        except IOError:
            self.results = {a: {m: [] for m in MEASURES} for a in SEGMENTERS}
            self.run_segmenters(n)

            # pickle the results to file
            pickle.dump(
                self.results,
                open(filename, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL,
                )

    # Language modeling alone, Optimality Theory, and unviolable segmenters
    def run_segmenters(self, n):
        for n in range(1, n + 1):
            print 'Run 1.%s' % n

            filename = 'data/test/sig/morfessor-n' + str(n)

            for approach in SEGMENTERS[:3]:
                F = FinnSeg(
                    training=FULL_TRAINING,
                    validation=TEST,
                    filename=filename,
                    Print=False,
                    approach=approach,
                    )

                self.results[approach]['P'].append(F.precision)
                self.results[approach]['R'].append(F.recall)
                self.results[approach]['F1'].append(F.f1)
                self.results[approach]['Acc'].append(F.accuracy)

            # Maxent inputs
            for excl_loans in [True, False]:
                MaxentInput(
                    training=FULL_TRAINING,
                    validation=None,
                    excl_loans=excl_loans,
                    filename='n' + str(n),
                    )


class SignificancePartTwo:

    def __init__(self, n=1):
        ttests_filename = 'data/test/sig/ttest-n%s.txt' % n
        results_filename = 'data/test/sig/results-n%s.pickle' % n

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
            if self.results['Maxent']['Acc'] == []:
                self.run_maxent_segmenters(n)
            self.do_ttests(n, ttests_filename)

    # Maxent segmenters
    def run_maxent_segmenters(self, n):
        for n in range(1, n + 1):
            print 'Run 2.%s' % n

            filename = 'data/test/sig/morfessor-n' + str(n)

            for excl_loans in [True, False]:
                output = 'data/test/sig/maxent-n%s-4' % str(n)
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
                    training=FULL_TRAINING,
                    validation=TEST,
                    filename=filename,
                    Print=False,
                    approach='Maxent',
                    constraints=CONSTRAINTS,
                    )

                approach = 'Maxent-exclLoans'if excl_loans else 'Maxent'

                self.results[approach]['P'].append(F.precision)
                self.results[approach]['R'].append(F.recall)
                self.results[approach]['F1'].append(F.f1)
                self.results[approach]['Acc'].append(F.accuracy)

        # pickle the updated results to file
        pickle.dump(
            self.results,
            open(filename, 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL,
            )

    def do_ttests(self, n, ttests_filename):
        titles = {None: 'LMA', }
        means = []
        ttests = []

        # get the mean performance of each segmenter on each metric
        for a in SEGMENTERS:
            performance = titles.get(a, a) + '\n'

            for m in MEASURES:
                performance += '\t%s:\t\t%s\n' % (m, mean(self.results[a][m]))

            performance += '\n'

            means.append(performance)

        # compare each segmenter against every other segmenter on each metric
        for (a1, a2) in combinations(SEGMENTERS, 2):
            ttest = titles.get(a1, a1) + ' ~ ' + titles.get(a2, a2) + '\n'

            for m in MEASURES:
                m1 = self.results[a1][m]
                m2 = self.results[a2][m]
                ttest += '\t%s:\t\t' % m
                ttest += 't = %6.3f\tp = %6.4f\n' % ttest_ind(m1, m2)

            ttest += '\n'

            ttests.append(ttest)

        # alphabetize t-tests
        ttests.sort()

        # compose ttests text file
        header = '---- t-tests n%s ' % str(n) + '-' * 40
        self.ttests = header + '\n\n'
        self.ttests += ''.join(means)
        self.ttests += ''.join(ttests)
        self.ttests += '\n' + '-' * len(header)

        with open(ttests_filename, 'w') as f:
                f.write(self.ttests)

        print self.ttests

if __name__ == '__main__':
    SignificancePartOne(n=20)
    # SignificancePartTwo(n=20)
