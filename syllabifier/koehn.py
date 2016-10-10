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
from itertools import product

from phonology import replace_umlauts

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finnsyll as finn

FS = FinnishStemmer()


class Koehn:

    def __init__(self, validation=None, pos_filtering=True, transform=False):
        self.validation_tokens = validation

        if pos_filtering:
            self.filename = './data/koehn/corpus-filtered-no_proper.pickle'

        else:
            self.filename = './data/koehn/corpus.pickle'

        # gather corpus frequencies
        print 'Gathering corpus frequencies... ' + dt.utcnow().strftime('%I:%M')  # noqa
        self.corpus = self.get_corpus(pos_filtering)

        if transform:
            # get words ending in "-nen"
            print 'Gathering nens... ' + dt.utcnow().strftime('%I:%M')
            self.nens = self.get_nens(pos_filtering)

            self.get_candidates = self._get_stripped_candidates
            self.score_candidate = self._score_stripped_candidate

        else:
            self.get_candidates = self._get_candidates
            self.score_candidate = self._score_candidate

        # segment and evaluate
        print 'Segmenting... ' + dt.utcnow().strftime('%I:%M')
        self.evaluate()

        print 'Complete. ' + dt.utcnow().strftime('%I:%M')

    def get_corpus(self, pos_filtering):
        # If pos_filtering, exclude the types Conjunction, Interjection,
        # Numeral, Preposition, and Proper. Include Adjective, Adjective-Noun,
        # Adverb, CompPart, Noun, Noun-Noun, Pronoun, and Verb.
        try:
            with open(self.filename) as f:
                corpus = pickle.load(f)

        except IOError:

            if pos_filtering:
                query = finn.Token.query.filter(finn.and_(
                    finn.Token.pos != 'Conjunction',
                    finn.Token.pos != 'Interjection',
                    finn.Token.pos != 'Numeral',
                    finn.Token.pos != 'Preposition',
                    finn.Token.pos != 'Proper',
                    ))
            else:
                query = finn.Token.query

            corpus = defaultdict(int)
            count = query.count()
            start = 0
            end = x = 1000

            while start + x < count:

                for t in query.order_by(finn.Token.id).slice(start, end):
                    corpus[t.orth.lower()] += t.freq

                start = end
                end += x

            for t in query.order_by(finn.Token.id).slice(start, count):
                corpus[t.orth.lower()] += t.freq

            with open(self.filename, 'wb') as f:
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

        return corpus

    def get_nens(self, pos_filtering):
        filename = self.filename.replace('corpus', 'nens')

        try:
            with open(filename) as f:
                nens = pickle.load(f)

        except IOError:

            if pos_filtering:
                query = finn.Token.query.filter(finn.and_(
                    finn.Token.pos != 'Conjunction',
                    finn.Token.pos != 'Interjection',
                    finn.Token.pos != 'Numeral',
                    finn.Token.pos != 'Preposition',
                    finn.Token.pos != 'Proper',
                    ))
            else:
                query = finn.Token.query

            # nens = defaultdict(int)
            nens = []
            count = query.count()
            start = 0
            end = x = 1000

            while start + x < count:

                for t in query.order_by(finn.Token.id).slice(start, end):
                    word = t.orth.lower()

                    if word.endswith('nen'):
                        nens.append(word)

                start = end
                end += x

            for t in query.order_by(finn.Token.id).slice(start, count):
                word = t.orth.lower()

                if word.endswith('nen'):
                    nens.append(word)

            with open(filename, 'wb') as f:
                pickle.dump(nens, f, protocol=pickle.HIGHEST_PROTOCOL)

        return nens

    def segment(self, word):
        word = word.lower()  # normalize
        candidates = self.get_candidates(word)

        try:
            scored_candidates = [self.score_candidate(k, v) for k, v in candidates.items()]  # noqa

        except AttributeError:
            scored_candidates = [self.score_candidate(c) for c in candidates]

        best = max(scored_candidates)[1]

        return best

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
        candidates = split(word)

        return candidates

    def _get_stripped_candidates(self, word):

        def swap(seg):
            return seg[:-1] + 'nen'

        def derivable(seg):
            return seg.endswith('s') and swap(seg) in self.nens

        candidates = self._get_candidates(word)
        candidate_dict = {c: [] for c in candidates}

        for cand in list(candidates):
            # 2 ^ n - 1 new candidates
            parts = [[s, swap(s)] if derivable(s) else [s, ] for s in cand.split('=')]  # noqa
            new = ['='.join(constituents) for constituents in product(*parts)]
            candidate_dict[cand].extend(new)

        candidate_dict = {k: list(set(v)) for k, v in candidate_dict.items()}

        return candidate_dict

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

        return score, candidate

    def _score_stripped_candidate(self, candidate, options):
        scored_options = [self._score_candidate(c) for c in options]
        score = max(scored_options)[0]

        return score, candidate

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
            '\n'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tTN:\t%s\n\tFN:\t%s\n\tBad:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s\n\tAcc.:\t%s'
            '\n\n'
            '-----------------------------------------------------------------'
            '\n'
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
    Koehn(validation=VALIDATION, pos_filtering=True)
