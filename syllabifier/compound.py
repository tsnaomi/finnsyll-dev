# coding=utf-8

import math
import morfessor
import re

# from datetime import datetime as dt
from itertools import izip_longest as izip, product
from os import sys, path
from phonology import (
    is_cluster,
    is_consonant,
    is_coronal,
    is_harmonic,
    is_vowel,
    replace_umlauts,
    )
# from pprint import pprint

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


# Morfessor -------------------------------------------------------------------

# In morphological segmentation, compounds are word forms, constructions are
# morphs, and atoms are characters. In chunking, compounds are sentences,
# constructions are phrases, and atoms are words.

class StupidMorfessorClassifier(object):

    def __init__(self, filename='data/morfessor-training', test=False, mx='P'):
        from finnsyll import training_set, dev_set, test_set

        self.training_tokens = training_set()
        self.validation_tokens = test_set() if test else dev_set()

        # Morfessor model
        self.model = None

        # filename of Training text and Morfessor model binary file
        self.filename = filename

        # ngram containers
        self.unigrams = {'UNK': 0, 'X': 1, '#': 1}
        self.bigrams = {'UNK': 0, }
        self.trigrams = {'UNK': 0, }
        self.total = 0

        # evaluation measure to maximize
        self.maximize = {'P': 0, 'R': 1, 'F1': 2}[mx]

        # coefficients
        self.a = 0.0  # ngram
        self.b = 0.0  # nuclei
        self.d = 0.0  # coronal
        self.g = 0.0  # harmonic
        self.p = 0.0  # sonseq

        # Evaluation report
        self.report = None

        self.train()
        self.evaluate()

    def train(self):
        self._train_morfessor()
        self._train_ngrams()
        self._train_coefficients()

    def _train_morfessor(self):
        io = morfessor.MorfessorIO()

        # load training data, or create training data if it is nonexistent
        try:
            train_data = list(io.read_corpus_file(self.filename + '.txt'))

        except IOError:
            print 'Creating training data...'

            tokens = [t.gold_base for t in self.training_tokens]
            tokens = ' '.join(tokens).replace('-', ' ').replace('=', ' ')
            tokens = replace_umlauts(tokens, put_back=True)
            tokens = tokens.lower().encode('utf-8')

            with open(self.filename + '.txt', 'w') as f:
                f.write(tokens)

            train_data = list(io.read_corpus_file(self.filename + '.txt'))

        # load model, or train and save model if it is nonexistent
        try:
            self.model = io.read_binary_model_file(self.filename + '.bin')

        except IOError:
            print 'Training Morfessor model...'

            model = morfessor.BaselineModel()
            model.load_data(train_data)
            model.train_batch()
            io.write_binary_model_file(self.filename + '.bin', model)

    def _train_ngrams(self):
        print 'Training ngrams...'

        for t in self.training_tokens:
            stems = re.split(r'=|-| ', t.gold_base)
            morphemes = map(
                lambda m: m.replace(' ', '').replace('-', ''),
                self.model.viterbi_segment(t.orth.lower())[0],
                )

            if stems == morphemes:

                for morpheme in morphemes:
                    self.unigrams.setdefault(morpheme, 0)
                    self.unigrams[morpheme] += 1
                    self.total += 1

                    bigram = '#' + morpheme
                    self.bigrams.setdefault(bigram, 0)
                    self.bigrams[bigram] += 1

                    bigram = morpheme + '#'
                    self.bigrams.setdefault(bigram, 0)
                    self.bigrams[bigram] += 1

                    trigram = '#' + morpheme + '#'
                    self.trigrams.setdefault(trigram, 0)
                    self.trigrams[trigram] += 1

            else:
                index = -1
                indices = [-1, ]

                for stem in stems:
                    index += len(stem)
                    indices.append(index)

                index = -1

                for morpheme in morphemes:
                    L = '#' if index in indices else 'X'
                    index += len(morpheme)
                    R = '#' if index in indices else 'X'

                    self.unigrams.setdefault(morpheme, 0)
                    self.unigrams[morpheme] += 1
                    self.total += 1

                    bigram = L + morpheme
                    self.bigrams.setdefault(bigram, 0)
                    self.bigrams[bigram] += 1

                    bigram = morpheme + R
                    self.bigrams.setdefault(bigram, 0)
                    self.bigrams[bigram] += 1

                    trigram = L + morpheme + R
                    self.trigrams.setdefault(trigram, 0)
                    self.trigrams[trigram] += 1

    def _train_coefficients(self):
        # generate all possible sets of coefficients
        coefficients = [float(i) / 100 for i in range(0, 101, 5)]
        coefficients = product(coefficients, repeat=5)
        coefficients = filter(lambda x: sum(x) == 1.0, coefficients)

        # further filter coefficients so that ngrams cannot receive a weight
        # less than 0.5
        # coefficients = filter(lambda x: x[0] > 0.5, coefficients)

        scored_coefficients = []

        # evaluate segmenter performance with each set of coefficients
        for i, co in enumerate(coefficients):

            # # write progress to terminal
            # message = 'Training coefficients... %s / %s                   \r'
            # sys.stdout.write(message % (i, len(coefficients)))
            # sys.stdout.flush()

            self.a, self.b, self.d, self.g, self.p = co
            score = self.evaluate(Eval=False)[self.maximize]
            scored_coefficients.append((score, coefficients))

        # set coefficients that maximizes the f1 score of the training data
        coefficients = max(scored_coefficients)[1]
        self.a, self.b, self.d, self.g, self.p = coefficients

        print 'a=%s\nb=%s\nd=%s\ng=%s\np=%s\n' % coefficients

    def evaluate(self, Eval=True):
        if Eval:
            print 'Evaluating...'

        # true positives, false positives, true negatives, false negatives, and
        # accurately identifies compounds with 'bad' segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}

        # set validation or training tokens
        tokens = self.validation_tokens if Eval else self.training_tokens

        # evaluate tokens
        for t in tokens:
            word = self.segment(t.orth)
            # print word, '(%s)' % t.gold_base

            if replace_umlauts(word) == t.gold_base:

                # true positives
                if t.is_complex:
                    results['TP'].append((word, t.gold_base))

                # true negatives
                else:
                    results['TN'].append((word, t.gold_base))

            # bad segmentations
            elif '=' in word and '=' in t.gold_base:
                results['bad'].append((word, t.gold_base))

            # false positives
            elif '=' in word:
                results['FP'].append((word, t.gold_base))

            # false negatives
            else:
                results['FN'].append((word, t.gold_base))

        TP = len(results['TP'])    # 372
        FP = len(results['FP'])    # 173
        TN = len(results['TN'])    # 1523
        FN = len(results['FN'])    # 5
        bad = len(results['bad'])  # 27

        # calculate precision, recall, and F1
        P = (TP * 1.0) / (TP + FP + bad)  # 0.6503
        R = (TP * 1.0) / (TP + FN)        # 0.9867
        F1 = (2.0 * P * R) / (P + R)      # 0.7840

        if not Eval:
            return P, R, F1

        # generate evaluation report
        self.report = (
            '\n'
            '---- Evaluation -------------------------------------------------'
            # '\n\nTrue negatives:\n\t%s'
            # '\n\nTrue positives:\n\t%s'
            '\n\nFalse negatives:\n\t%s'
            '\n\nFalse positives:\n\t%s'
            '\n\nBad segmentations:\n\t%s'
            '\n\nMaximizing: %s (0=P, 1=R, 2=F1)'
            '\n\nTP: %s\nFP: %s\nTN: %s\nFN: %s\nBad: %s'
            '\n\nP / R (F1): %s / %s (%s)\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TN']]),
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TP']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FN']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FP']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['bad']]),
                self.maximize,
                TP,
                FP,
                TN,
                FN,
                bad,
                P,
                R,
                F1,
                )

        print self.report

    def segment(self, word):
        token = []

        for comp in re.split(r'(-| )', word):

            if len(comp) > 1:
                morphemes = map(
                    lambda m: m.replace(' ', '').replace('-', ''),
                    self.model.viterbi_segment(comp.lower())[0],
                    )
                token.append(self._segment(morphemes))
                continue

            token.append(comp)

        token = ''.join(token)

        return token

    def _segment(self, morphemes):
        scored_candidates = []
        delimiter_sets = product(['#', 'X'], repeat=len(morphemes) - 1)

        for delimiters in delimiter_sets:
            candidate = [x for y in izip(morphemes, delimiters) for x in y]
            candidate = ['#', ] + filter(None, candidate) + ['#', ]
            scored_candidates.append(self.score(candidate))

        # pprint(scored_candidates)

        morphemes = max(scored_candidates)[1]

        return morphemes

    def score(self, candidate):
        score, candidate = self._score_ngrams(candidate)
        score, candidate = self._score_phonotactics(score, candidate)

        return score, candidate

    def _score_ngrams(self, candidate):
        score = 0

        for i, morpheme in enumerate(candidate):
            C = morpheme

            if i > 0:
                B = candidate[i-1]

                if i > 1:
                    A = candidate[i-2]
                    ABC = A + B + C
                    ABC_count = self.trigrams.get(ABC, 0)

                    if ABC_count:
                        AB = A + B
                        AB_count = self.bigrams[AB]
                        score += math.log(ABC_count)
                        score -= math.log(AB_count)
                        continue

                BC = B + C
                BC_count = self.bigrams.get(BC, 0)

                if BC_count:
                    B_count = self.unigrams[B]
                    score += math.log(BC_count * 0.4)
                    score -= math.log(B_count)
                    continue

            C_count = self.unigrams.get(C, 0) + 1
            score += math.log(C_count * 0.4)
            score -= math.log(self.total + len(self.unigrams))

        return score, candidate

    def _score_phonotactics(self, score, candidate):

        def _nuclei(segment):
            return len(filter(is_vowel, segment)) > 1

        def _coronal(segment):
            return is_vowel(segment[-1]) or is_coronal(segment[-1])

        def _harmonic(segment):
            return is_harmonic(segment)

        def _sonseq(segment):
            onset = re.split(r'(i|e|A|y|O|a|u|o)', segment)[0]

            if onset and is_consonant(onset[0]):
                return is_consonant(onset) or is_cluster(onset)

            return True

        # convert score from negative to positive
        score = 100.0 - (score * -1.0)
        score /= 100.0

        # convert candidate from list to string to normalize it
        del candidate[0]
        del candidate[-1]
        candidate = ''.join(candidate).replace('#', '=').replace('X', '')

        # get suggested segmentations as list
        segments = replace_umlauts(candidate).split('=')

        if len(segments) > 1:

            # score phonotactic features
            nuclei = 1.0 if all(_nuclei(seg) for seg in segments) else 0.0
            coronal = 1.0 if all(_coronal(seg) for seg in segments) else 0.0
            harmonic = 1.0 if all(_harmonic(seg) for seg in segments) else 0.0
            sonseq = 1.0 if all(_sonseq(seg) for seg in segments) else 0.0

            score *= self.a
            score += nuclei * self.b
            score += coronal * self.d
            score += harmonic * self.g
            score += sonseq * self.p

        return score, candidate


# -----------------------------------------------------------------------------

def delimit(word):
    '''Insert syllable breaks at non-delimited compound boundaries.'''
    return word


if __name__ == '__main__':

    StupidMorfessorClassifier()
    StupidMorfessorClassifier(mx='R')
    StupidMorfessorClassifier(mx='F1')

    # words = [
    #     'pian',             # pian
    #     'talous',           # talous
    #     'erinomaisesti',    # erinomaisesti
    #     'jääkiekkoilu',     # jää=kiekkoilu
    #     'asianajaja',       # asian=ajaja
    #     'rahoituserien',    # rahoitus=erien
    #     'vastikään',        # vast=ikään
    #     'kansanäänestys',   # kansan=äänestys
    #     ]

    # for word in words:
    #     split = model.viterbi_segment(word)
    #     print '='.join(split[0]), split[1]
