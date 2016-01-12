# coding=utf-8

# A LOT OF COUNTING AND DIVIDING.
# CHECKOUT: KenLM, Google Ngram Viewer

import math
import morfessor
import re

from itertools import izip_longest as izip, product
from os import sys, path
from phonology import (
    check_nuclei as nuclei,
    check_sonseq as sonseq,
    check_word_final as word_final,
    is_harmonic as harmonic,
    replace_umlauts,
    )

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # UH OH

import finnsyll as finn

TRAINING = finn.training_set()
VALIDATION = finn.dev_set()


# Morfessor: "In morphological segmentation, compounds are word forms,
# constructions are morphs, and atoms are characters. In chunking, compounds
# are sentences, constructions are phrases, and atoms are words."

class DilettanteSplitter(object):

    def __init__(self, training=TRAINING, validation=VALIDATION,
                 filename='data/morfessor-training', test=False,
                 maximize='P', Eval=True, train_coefficients=True,
                 a=1.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0):
        self.training_tokens = training
        self.validation_tokens = validation

        # Morfessor model, or some other morphological analyzer
        self.model = None

        # filename of training text and morfessor model binary file
        prefix = 'syllabifier/' if __name__ != '__main__' else ''  # UH OH
        self.filename = prefix + filename

        # ngram containers
        self.unigrams = {}  # UNK?
        self.bigrams = {}   # UNK?
        self.trigrams = {}  # UNK?
        self.total = 0

        # evaluation measure to maximize
        self.maximize = {'P': 0, 'R': 1, 'F1': 2, 'F05': 3}[maximize]

        # coefficients
        self.a = a  # ngram
        self.b = b  # nuclei
        self.c = d  # coronal
        self.d = d  # harmonic
        self.e = e  # sonseq
        self.f = f  # breaks

        # evaluation report
        self.report = None

        # train segmenter
        self.train_coefficients = False if a < 1.0 else train_coefficients
        self.train()

        # evaluate segmenter
        if Eval:
            self.evaluate()

    def train(self, train_coefficients=True):
        self._train_morfessor()
        self._train_ngrams()

        if self.train_coefficients:
            self._train_coefficients()

    def _train_morfessor(self):
        io = morfessor.MorfessorIO()

        # load model, or train and save model if it is nonexistent
        try:
            self.model = io.read_binary_model_file(self.filename + '.bin')

        except IOError:

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

            print 'Training Morfessor model...'

            self.model = morfessor.BaselineModel()
            self.model.load_data(train_data)
            self.model.train_batch()
            io.write_binary_model_file(self.filename + '.bin', self.model)

    def _train_ngrams(self):
        print 'Training ngrams...'

        for t in self.training_tokens:
            stems = re.split(r'=|-| ', t.gold_base)
            morphemes = map(
                lambda m: m.replace(' ', '').replace('-', ''),
                self.model.viterbi_segment(t.orth.lower())[0],
                )

            # create word of form [#, morpheme1, X, morpheme2, #]
            word = []
            index = -1
            indices = [-1, ]

            for stem in stems:
                index += len(stem)
                indices.append(index)

            index = -1

            for morpheme in morphemes:
                word.append('#' if index in indices else 'X')
                word.append(morpheme)
                index += len(morpheme)

            word.append('#')

            # get unigram, bigram, and trigram counts
            for i, morpheme in enumerate(word):
                self.unigrams.setdefault(morpheme, 0)
                self.unigrams[morpheme] += 1
                self.total += 1

                if i > 0:
                    bigram = word[i-1] + morpheme
                    self.bigrams.setdefault(bigram, 0)
                    self.bigrams[bigram] += 1

                if i > 1:
                    trigram = word[i-2] + bigram
                    self.trigrams.setdefault(trigram, 0)
                    self.trigrams[trigram] += 1

    def _train_ngrams_old(self):  # out of commission
        print 'Training ngrams...'

        self.unigrams['X'] = 1
        self.unigrams['#'] = 1

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
        filename = '%s-%s-coefficients.txt' % (self.filename, self.maximize)

        # load coefficients, or train coefficients if they are non-existent
        try:
            with open(filename, 'r') as f:
                coefficients = [float(i) for i in f.read().split()]
                self.a, self.b, self.d, self.g, self.p, self.k = coefficients

        except IOError:
            # generate all possible sets of coefficients
            coefficients = [float(i) / 100 for i in range(0, 101, 5)]
            coefficients = product(coefficients, repeat=6)
            coefficients = filter(lambda x: sum(x) == 1.0, coefficients)

            # further filter coefficients so that ngram scores cannot receive a
            # weight less than 0
            coefficients = filter(lambda x: x[0] > 0, coefficients)

            # number of sets of coefficients
            count = len(coefficients)

            scored_coefficients = []

            # evaluate segmenter performance with each set of coefficients
            for i, co in enumerate(coefficients):

                # write progress to terminal
                message = 'Training coefficients... %s / %s               \r'
                sys.stdout.write(message % (i, count))
                sys.stdout.flush()

                self.a, self.b, self.d, self.g, self.p, self.k = co
                score = self.evaluate(Eval=False)[self.maximize]
                scored_coefficients.append((score, co))

            # write final progress to terminal (e.g., 604/604)
            print 'Training coefficients... %s / %s' % (count, count)

            # set coefficients that maximizes the f1 score of the training data
            coefficients = max(scored_coefficients)[1]
            self.a, self.b, self.d, self.g, self.p, self.k = coefficients

            print (
                '\na=%s\tngram\nb=%s\tnuclei\nd=%s\tcoronal\ng=%s\tharmonic'
                '\np=%s\tsonseq\n'
                ) % tuple([format(c, '.2f') for c in coefficients])

            with open(filename, 'w') as f:
                f.write('%s\n%s\n%s\n%s\n%s' % coefficients)

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
        P = (TP * 1.0) / (TP + FP + bad)  # 0.6514
        R = (TP * 1.0) / (TP + FN)        # 0.9867
        F1 = (2.0 * P * R) / (P + R)      # 0.7848
        F05 = ((0.5**2 + 1.0) * P * R) / ((0.5**2 * P) + R)  # 0.6989

        # TODO: perplexity measure

        if not Eval:
            return P, R, F1, F05

        # generate evaluation report
        self.report = (
            '\n'
            '---- Evaluation: DilettanteSplitter -----------------------------'
            # '\n\nTrue negatives:\n\t%s'
            # '\n\nTrue positives:\n\t%s'
            '\n\nFalse negatives:\n\t%s'
            '\n\nFalse positives:\n\t%s'
            '\n\nBad segmentations:\n\t%s'
            '\n\nMaximizing: %s (0=P, 1=R, 2=F1, 3=F0.5)'
            '\n\nTP:\t%s\nFP:\t%s\nTN:\t%s\nFN:\t%s\nBad:\t%s'
            '\n\nP/R:\t%s / %s\nF1:\t%s\nF0.5:\t%s\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TN']]),
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TP']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FN']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FP']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['bad']]),
                'N/A' if not self.train_coefficients else self.maximize,
                TP,
                FP,
                TN,
                FN,
                bad,
                P,
                R,
                F1,
                F05,
                )

        print self.report

    def segment(self, word, inform_base=False):
        if inform_base:
            word = replace_umlauts(word, put_back=False)

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

        if inform_base:
            token = replace_umlauts(token)

        return token

    def _segment(self, morphemes):
        scored_candidates = []
        delimiter_sets = product(['#', 'X'], repeat=len(morphemes) - 1)

        for delimiters in delimiter_sets:
            candidate = [x for y in izip(morphemes, delimiters) for x in y]
            candidate = ['#', ] + filter(None, candidate) + ['#', ]
            scored_candidates.append(self.score(candidate))

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
            score -= math.log(self.total + len(self.unigrams))  # UNK?

        # convert score from negative to positive
        score = 100.0 - (score * -1.0)
        score /= 100.0

        # convert candidate from list to string
        del candidate[0]
        del candidate[-1]
        candidate = ''.join(candidate).replace('#', '=').replace('X', '')

        return score, candidate

    def _score_phonotactics(self, score, candidate):
        if self.a < 1:

            # get suggested segmentations as list
            segments = replace_umlauts(candidate).split('=')

            if len(segments) > 1:

                # score phonotactic features
                nuclei = 1 if all(nuclei(seg) for seg in segments) else 0
                coronal = 1 if all(word_final(seg) for seg in segments) else 0
                harmonic = 1 if all(harmonic(seg) for seg in segments) else 0
                sonseq = 1 if all(sonseq(seg) for seg in segments) else 0
                breaks = 1.0 / len(segments)

                score *= self.a
                score += nuclei * self.b
                score += coronal * self.c
                score += harmonic * self.d
                score += sonseq * self.e
                score += breaks * self.f

        return score, candidate


def delimit(word):
    '''Insert syllable breaks at non-delimited compound boundaries.'''
    return word


if __name__ == '__main__':
    # DilettanteSplitter(maximize='P')
    # DilettanteSplitter(maximize='R')
    # DilettanteSplitter(maximize='F1')
    # DilettanteSplitter(maximize='F05')
    # DilettanteSplitter(train_coefficients=False)
    DilettanteSplitter(a=0.8, b=0.1, c=0.0, d=0.0, e=0.0, f=0.1)
