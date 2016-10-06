# coding=utf-8

try:
    import cpickle as pickle

except ImportError:
    import pickle

import os
import sys

from collections import defaultdict
from datetime import datetime as dt
from nltk.stem.snowball import FinnishStemmer

from phonology import replace_umlauts

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finnsyll as finn

FS = FinnishStemmer()


class Koehn:

    def __init__(self, validation=None, evaluate=True):
        self.validation_tokens = validation
        self.filename = './data/koehn/corpus.pickle'

        # gather corpus frequencies
        print 'Gathering corpus frequencies... ' + dt.utcnow().strftime('%I:%M')  # noqa
        self.corpus = self.get_corpus()

        # segment and evaluate
        if evaluate:
            print 'Segmenting... ' + dt.utcnow().strftime('%I:%M')
            self.evaluate()

        print 'Complete. ' + dt.utcnow().strftime('%I:%M')

    def get_corpus(self):
        try:
            with open(self.filename) as f:
                corpus = pickle.load(f)

        except IOError:
            corpus = defaultdict(int)
            count = finn.Token.query.count()
            start = 0
            end = x = 1000

            while start + x < count:

                for t in finn.Token.query.order_by(finn.Token.id).slice(start, end):  # noqa
                    corpus[t.orth.lower()] += t.freq

                start = end
                end += x

            for t in finn.Token.query.order_by(finn.Token.id).slice(start, count):  # noqa
                corpus[t.orth.lower()] += t.freq

            with open(self.filename, 'wb') as f:
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

        return corpus

    def segment(self, word):
        word = word.lower()  # normalize
        candidates = self._get_candidates(word)
        scored_candidates = [self._score_candidate(c) for c in candidates]
        best = max(scored_candidates)  # [1]

        scores = [sc for sc in scored_candidates if sc[0] == best[0]]
        if len(scores) > 1:
            print scores, best[1]

        return best[1]

    def _get_candidates(self, word):

        # stackoverflow.com/questions/3577362
        def split(s):
            result = [s, ]

            for i in range(1, len(s)):
                result.extend(
                    '%s=%s' % (s[:i], x) for x in split(s[i:])
                    if len(s[:i]) > 2 and len(x) > 2
                    )

            return result

        # get all splits of minimum length 3
        return split(word)

    def _score_candidate(self, candidate):
        constituents = candidate.split('=')

        score = 1

        for cand in constituents:
            freq = self.corpus[cand]

            if freq == 0:
                score = 0
                break

            score *= freq

        score *= (1.0 / len(constituents))

        return (score, candidate)

    def evaluate(self):
        # results include true positives, false positives, true negatives,
        # false negatives, and accurately identified compounds with 'bad'
        # segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}

        for i, t in enumerate(self.validation_tokens, start=1):
            word = self.segment(t.orth)
            gold = replace_umlauts(t.gold_base, put_back=True)
            label = self._label(word, gold, t.is_complex)
            results[label].append((word, gold))

            sys.stdout.write(
                'TP: %s  FP: %s  TN: %s  FN: %s  Bad: %s  #%s\r' % (
                    len(results['TP']),
                    len(results['FP']),
                    len(results['TN']),
                    len(results['FN']),
                    len(results['bad']),
                    i,
                    )
                )
            sys.stdout.flush()

        TP = len(results['TP'])
        FP = len(results['FP'])
        TN = len(results['TN'])
        FN = len(results['FN'])
        bad = len(results['bad'])

        # calculate precision, recall, and F-measures on a word-by-word basis
        try:
            P = float(TP) / (TP + FP + bad)
            R = float(TP) / (TP + FN + bad)
            F1 = (2.0 * P * R) / (P + R)
            F05 = (float(0.5**2 + 1) * P * R) / ((0.5**2 * P) + R)

        except ZeroDivisionError:
            P, R, F1, F05 = 0.0, 0.0, 0.0, 0.0

        ACCURACY = float(TP + TN) / (TP + TN + FP + FN + bad)

        print (
            '\n\n'
            '---- Evaluation: Koehn & Knight 2003 ----------------------------'
            '\n\nFalse negatives:\n\t%s'
            '\n\nFalse positives:\n\t%s'
            '\n\nBad segmentations:\n\t%s'
            '\n\n'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tTN:\t%s\n\tFN:\t%s\n\tBad:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s\n\tAcc.:\t%s'
            '\n\n'
            '-----------------------------------------------------------------'
            '\n\n'
            ) % (
                '\n\t'.join(['%s (%s)' % t for t in results['FN']]),
                '\n\t'.join(['%s (%s)' % t for t in results['FP']]),
                '\n\t'.join(['%s (%s)' % t for t in results['bad']]),
                TP, FP, TN, FN, bad, P, R, F1, F05, ACCURACY,
                )

    def _label(self, word, gold, is_complex):
        # true positive or true negative
        if word == gold:
            label = 'TP' if is_complex and '=' in gold else 'TN'

        # bad segmentation or false positive
        elif '=' in word:
            label = 'bad' if is_complex else 'FP'

        # false negative
        else:
            label = 'FN'

        return label


if __name__ == '__main__':
    # VALIDATION = finn.dev_set()
    VALIDATION = finn.test_set()
    Koehn(validation=VALIDATION)
