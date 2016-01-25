# coding=utf-8

# A LOT OF COUNTING AND DIVIDING.

try:
    import cpickle as pickle

except ImportError:
    import pickle

import csv
import math
import morfessor
import re

from collections import Counter, defaultdict
from itertools import izip_longest as izip, product
from os import sys, path
from phonology import (
    check_nuclei as _nuclei,
    check_sonseq as _sonseq,
    check_word_final as _word_final,
    is_harmonic as _harmonic,
    replace_umlauts,
    )

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # UH OH

import finnsyll as finn

TRAINING = finn.training_set()
VALIDATION = finn.dev_set()
TEST = finn.test_set()


# Morfessor: "In morphological segmentation, compounds are word forms,
# constructions are morphs, and atoms are characters. In chunking, compounds
# are sentences, constructions are phrases, and atoms are words."

class FinnSeg(object):

    def __init__(self, training=TRAINING, validation=VALIDATION, Eval=True,
                 filename='data/morfessor', train_coefficients=True,
                 smoothing='stupid', absolute=False, UNK=False,
                 a=1.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0):

        # if coefficients do not sum to 1, throw an error
        if round(sum([a, b, c, d, e, f]), 4) != 1:
            raise ValueError('Coefficients need to sum to 1.')

        self.training_tokens = training
        self.validation_tokens = validation

        # Morfessor model, or some other morphological analyzer
        self.model = None

        # filename of training text and morfessor model binary file
        prefix = 'syllabifier/' if __name__ != '__main__' else ''  # UH OH
        self.filename = prefix + filename

        # ngram and open vocabulary containers
        self.UNK = UNK
        self.ngrams = {}
        self.vocab = set()
        self.total = 0

        # select smoothing algorithm
        self.smoothing = smoothing

        # Stupid Backoff (smoothed scoring method)
        if smoothing.lower() == 'stupid':
            self.ngram_score = self._stupid_backoff_score

        # Interpolated Modified Kneser-Ney (smoothed scoring method)
        elif smoothing.lower() == 'mkn':
            self.ngram_score = self._interpolated_modified_kneser_ney_score
            self.discounts = None
            self.alphas = {}

        else:
            raise ValueError('Invalid smoothing algorithm specified.')

        # coefficients
        self.a = a  # ngram
        self.b = b  # nuclei
        self.c = d  # word-final
        self.d = d  # harmonic
        self.e = e  # sonseq
        self.f = f  # breaks

        # train segmenter
        self.train_coefficients = False if a < 1.0 else train_coefficients
        self.train()

        # if absolute is specified, treat nuclei and sonority sequencing tests
        # as unviolable constraints
        self.absolute = absolute

        if self.absolute:
            self.a += self.b + self.e
            self.b = 0
            self.e = 0

        # evaluation report
        self.report = None

        # evaluate segmenter
        if Eval:
            self.evaluate()

    # Training ----------------------------------------------------------------

    def train(self, train_coefficients=True):
        self._train_morfessor()
        self._train_ngrams()

        if self.smoothing == 'mkn':
            self._train_modified_kneser_ney_parameters()

        if self.train_coefficients:
            pass

    def _train_morfessor(self):
        io = morfessor.MorfessorIO()
        filename = self.filename + '-training'

        # load model, or train and save model if it is nonexistent
        try:
            self.model = io.read_binary_model_file(filename + '.bin')

        except IOError:

            # load training data, or create training data if it is nonexistent
            try:
                train_data = list(io.read_corpus_file(filename + '.txt'))

            except IOError:
                print 'Creating training data...'

                tokens = [t.gold_base for t in self.training_tokens]
                tokens = ' '.join(tokens).replace('-', ' ').replace('=', ' ')
                tokens = replace_umlauts(tokens, put_back=True)
                tokens = tokens.lower().encode('utf-8')

                with open(self.filename + '.txt', 'w') as f:
                    f.write(tokens)

                train_data = list(io.read_corpus_file(filename + '.txt'))

            print 'Training Morfessor model...'

            self.model = morfessor.BaselineModel()
            self.model.load_data(train_data)
            self.model.train_batch()
            io.write_binary_model_file(filename + '.bin', self.model)

    # Language modeling -------------------------------------------------------

    def _train_ngrams(self):
        filename = self.filename + ('-ngrams-UNK' if self.UNK else '-ngrams')

        # load ngrams, or train and save ngrams if they are nonexistent
        try:
            self.ngrams, self.vocab, self.total = \
                pickle.load(open(filename + '.pickle'))

        except IOError:
            print 'Training ngrams...'

            for t in self.training_tokens:
                stems = re.split(r'=|-| ', t.gold_base)
                morphemes = filter(None, map(
                    lambda m: m.replace(' ', '').replace('-', ''),
                    self.model.viterbi_segment(t.orth.lower())[0],
                    ))

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

                # if self.UNK, model out of vocabulary (OOV) words with <UNK>
                # (this is done by replacing every first instance of a morpheme
                # with <UNK>)
                WORD = [m if m in self.vocab else '<UNK>' for m in word] \
                    if self.UNK else word
                self.vocab.update(word)

                # get unigram, bigram, and trigram counts
                for i, morpheme in enumerate(WORD):
                    self.ngrams.setdefault(morpheme, 0)
                    self.ngrams[morpheme] += 1
                    self.total += 1

                    if i > 0:
                        bigram = WORD[i-1] + ' ' + morpheme
                        self.ngrams.setdefault(bigram, 0)
                        self.ngrams[bigram] += 1

                    if i > 1:
                        trigram = WORD[i-2] + ' ' + bigram
                        self.ngrams.setdefault(trigram, 0)
                        self.ngrams[trigram] += 1

            # remove any morphemes that were only seen once (since these were
            # were previously converted into <UNK>)
            self.vocab = filter(lambda w: w in self.ngrams.keys(), self.vocab)

            # pickle ngrams to file
            pickle.dump(
                [self.ngrams, self.vocab, self.total],
                open(filename + '.pickle', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL,
                )

    # Smoothing ---------------------------------------------------------------

    def _train_modified_kneser_ney_parameters(self):
        ngrams = Counter(self.ngrams.values())

        # N# is the number of ngrams with exactly count #
        N1 = float(ngrams[1])
        N2 = float(ngrams[2])
        N3 = float(ngrams[3])
        N4 = float(ngrams[4])
        Y = N1 / (N1 + 2 * N2)

        D1 = 1 - 2 * Y * (N2 / N1)
        D2 = 2 - 3 * Y * (N3 / N2)
        D3 = 3 - 4 * Y * (N4 / N3)

        # add discounts to dict
        self.discounts = defaultdict(lambda: D3)
        self.discounts[0] = 0
        self.discounts[1] = D1
        self.discounts[2] = D2
        self.discounts[3] = D3

        continuations = defaultdict(lambda: {1: set(), 2: set(), 3: set()})

        # get unique continuations for each history
        for ngram, counts in self.ngrams.iteritems():
            morphemes = ngram.split(' ')

            if len(morphemes) > 1:
                history, word = ' '.join(morphemes[:-1]), morphemes[-1]
                counts = counts if counts < 4 else 3
                continuations[history][counts].add(word)

        # set alpha parameters for each history
        for history, v in continuations.iteritems():
            alpha = (D1 * len(v[1])) + (D2 * len(v[2])) + (D3 * len(v[3]))
            alpha /= self.ngrams[history]
            self.alphas[history] = alpha

    def _interpolated_modified_kneser_ney_score(self, candidate):

        def interpolate(ngram):

            # if unigram
            if len(ngram) == 1:
                return 1.0 / self.total

            # if bigram or trigram
            else:
                history = ' '.join(ngram[:-1])
                count = self.ngrams.get(' '.join(ngram), 0)
                score = float(max(count - self.discounts[count], 0))
                score /= self.ngrams.get(history, 1)
                score += self.alphas.get(history, 0) * interpolate(ngram[1:])

            return score

        score = 0

        if self.UNK:
            candidate = [m if m in self.vocab else '<UNK>' for m in candidate]

        for i, morpheme in enumerate(candidate):
            ngram = candidate[i-2:i+1] or candidate[i-1:i+1] or [morpheme, ]
            score += math.log(interpolate(ngram) or 1)

        return score

    def _stupid_backoff_score(self, candidate):
        score = 0

        if self.UNK:
            candidate = [m if m in self.vocab else '<UNK>' for m in candidate]

        for i, morpheme in enumerate(candidate):
            C = morpheme

            if i > 0:
                B = candidate[i-1]

                if i > 1:
                    A = candidate[i-2]
                    ABC = A + ' ' + B + ' ' + C
                    ABC_count = self.ngrams.get(ABC, 0)

                    if ABC_count:
                        AB = A + ' ' + B
                        AB_count = self.ngrams[AB]
                        score += math.log(ABC_count)
                        score -= math.log(AB_count)
                        continue

                BC = B + ' ' + C
                BC_count = self.ngrams.get(BC, 0)

                if BC_count:
                    B_count = self.ngrams[B]
                    score += math.log(BC_count * 0.4)
                    score -= math.log(B_count)
                    continue

            C_count = self.ngrams.get(C, 1)  # Laplace smoothed unigram
            score += math.log(C_count * 0.4)
            score -= math.log(self.total + len(self.vocab) + 1)

        return score

    # Scoring -----------------------------------------------------------------

    def score(self, candidate):
        # return the candidate's smoothed language model score
        score = self.ngram_score(candidate)

        # convert candidate from list to string
        del candidate[0]
        del candidate[-1]
        candidate = ''.join(candidate).replace('#', '=').replace('X', '')

        # note that, if self.a is equal to 1, then the candidate's score is
        # equal to the score returned by self._score_ngrams()
        if self.a < 1 or self.absolute:

            # convert score from negative to positive
            score = 100.0 - (score * -1.0)
            score /= 100.0

            # get segmentation as list: e.g., 'book=worm' > ['book', 'worm']
            segments = replace_umlauts(candidate).split('=')

            # score phonotactic features
            nuclei = 1 if all(_nuclei(seg) for seg in segments) else 0
            word_final = 1 if all(_word_final(seg) for seg in segments) else 0
            harmonic = 1 if all(_harmonic(seg) for seg in segments) else 0
            sonseq = 1 if all(_sonseq(seg) for seg in segments) else 0
            breaks = 1.0 / len(segments)

            # automatically disqualify candidates that fail nuclei and
            # sonority sequencing tests, if absolute is specified
            if self.absolute and (not nuclei or not sonseq):
                return 0, candidate

            # calculate composite score
            score *= self.a
            score += nuclei * self.b
            score += word_final * self.c
            score += harmonic * self.d
            score += sonseq * self.e
            score += breaks * self.f

        return score, candidate

    # Segmentation ------------------------------------------------------------

    def segment(self, word):
        token = []

        # split the word along any overt delimiters and iterate across the
        # components
        for comp in re.split(r'(-| )', word):

            if len(comp) > 1:

                # use the language model to obtain the component's morphemes
                morphemes = self.model.viterbi_segment(comp.lower())[0]

                scored_candidates = []
                delimiter_sets = product(['#', 'X'], repeat=len(morphemes) - 1)

                # produce and score each candidate segmentation
                for d in delimiter_sets:
                    candidate = [x for y in izip(morphemes, d) for x in y]
                    candidate = ['#', ] + filter(None, candidate) + ['#', ]
                    scored_candidates.append(self.score(candidate))

                # select the best-scoring segmentation
                morphemes = max(scored_candidates)[1]
                token.append(morphemes)

            else:
                token.append(comp)

        # return the segmentation in string form
        return ''.join(token)

    def get_morphemes(self, word):
        morphemes = []

        # split the word along any overt delimiters and iterate across the
        # components
        for comp in re.split(r'(-| )', word):

            if len(comp) > 1:

                # use the language model to obtain the component's morphemes
                comp = map(
                    lambda m: m.replace(' ', '').replace('-', ''),
                    self.model.viterbi_segment(comp.lower())[0],
                    )
                morphemes.extend(comp)

            else:
                morphemes.append(comp)

        return morphemes

    # Evalutation -------------------------------------------------------------

    def evaluate(self):

        # results include true positives, false positives, true negatives,
        # false negatives, and accurately identified compounds with 'bad'
        # segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}

        # evaluate tokens
        for t in self.validation_tokens:
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

        TP = len(results['TP'])
        FP = len(results['FP'])
        TN = len(results['TN'])
        FN = len(results['FN'])
        bad = len(results['bad'])

        # TODO: average precision, recall, and F1 compound by compound

        # calculate precision, recall, and F1
        P = (TP * 1.0) / (TP + FP + bad)
        R = (TP * 1.0) / (TP + FN)
        F1 = (2.0 * P * R) / (P + R)
        F05 = ((0.5**2 + 1.0) * P * R) / ((0.5**2 * P) + R)

        # TODO: calculate perplexity

        # generate an evaluation report
        self.report = (
            '\n'
            '---- Evaluation: FinnSeg ----------------------------------------'
            # '\n\nTrue negatives:\n\t%s'
            # '\n\nTrue positives:\n\t%s'
            # '\n\nFalse negatives:\n\t%s'
            # '\n\nFalse positives:\n\t%s'
            # '\n\nBad segmentations:\n\t%s'
            '\n\nWeights: a=%s, b=%s, c=%s, d=%s, e=%s, f=%s'
            '\n\t ngram, nuclei, word-final, harmony, sonseq, boundaries%s'
            '\n\nTP:\t%s\nFP:\t%s\nTN:\t%s\nFN:\t%s\nBad:\t%s'
            '\n\nP/R:\t%s / %s\nF1:\t%s\nF0.5:\t%s\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TN']]),
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TP']]),
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FN']]),
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FP']]),
                # '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['bad']]),
                self.a, self.b, self.c, self.d, self.e, self.f,
                '\n\t Absolute nuclei and sonseq.' if self.absolute else '',
                TP, FP, TN, FN, bad, P, R, F1, F05,
                )

        print self.report

    def trial_evaluate(self):

        # results include true positives, false positives, true negatives,
        # false negatives, and accurately identified compounds with 'bad'
        # segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}
        measures = {'p': [], 'r': [], 'f1': [], 'f05': []}

        evaluations = (self._evaluate(t) for t in self.validation_token)

        for label, word, gold, p, r, f1, f05 in evaluations:
            results[label].append((word, gold))
            measures['p'].append(p)
            measures['r'].append(r)
            measures['f1'].append(f1)
            measures['f05'].append(f05)

        TP = len(results['TP'])
        FP = len(results['FP'])
        TN = len(results['TN'])
        FN = len(results['FN'])
        bad = len(results['bad'])

        # calculate precision, recall, and F1 (the original way)
        P = (TP * 1.0) / (TP + FP + bad)
        R = (TP * 1.0) / (TP + FN)
        F1 = (2.0 * P * R) / (P + R)
        F05 = ((0.5**2 + 1.0) * P * R) / ((0.5**2 * P) + R)

        # calculate precision, recall, F1, and F0.5 by averaging the
        # boundary-by-boundary measures of each token
        length = len(measures['p'])
        p = float(sum(measures['p'])) / length
        r = float(sum(measures['r'])) / length
        f1 = float(sum(measures['f1'])) / length
        f05 = float(sum(measures['f05'])) / length

        # TODO: calculate perplexity

        # generate an evaluation report
        self.report = (
            '\n'
            '---- Evaluation: FinnSeg ----------------------------------------'
            '\n\nTrue negatives:\n\t%s'
            '\n\nTrue positives:\n\t%s'
            '\n\nFalse negatives:\n\t%s'
            '\n\nFalse positives:\n\t%s'
            '\n\nBad segmentations:\n\t%s'
            '\n\nWeights: a=%s, b=%s, c=%s, d=%s, e=%s, f=%s'
            '\n\t ngram, nuclei, word-final, harmony, sonseq, breaks%s'
            '\n\nTP:\t%s\nFP:\t%s\nTN:\t%s\nFN:\t%s\nBad:\t%s'
            '\n\nWord Level:\nP/R:\t%s / %s\nF1:\t%s\nF0.5:\t%s'
            '\n\nBoundary Level:\nP/R:\t%s / %s\nF1:\t%s\nF0.5:\t%s\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TN']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['TP']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FN']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['FP']]),
                '\n\t'.join(['%s (%s)' % (w, t) for w, t in results['bad']]),
                self.a, self.b, self.c, self.d, self.e, self.f,
                '\n\t Absolute nuclei and sonseq.' if self.absolute else '',
                TP, FP, TN, FN, bad, P, R, F1, F05, p, r, f1, f05,
                )

        print self.report

    def _evaluate(self, token):

        # calculate precision and recall on a boundary-by-boundary basis
        def pr(word, token):
            w = [i for i, j in enumerate(word)]
            t = [i for i, j in enumerate(token.gold_base)]

            tp = float(len([i for i in w if i in t]))
            fp = len([i for i in w if i not in t])
            fn = len([i for i in t if i not in w])

            p = tp / (tp + fp)
            r = tp / (tp + fn)

            return p, r

        word = self.segment(token.orth)

        if replace_umlauts(word) == token.gold_base:

            # true positive
            if token.is_complex:
                label, p, r = 'TP', 1, 1

            # true negative
            else:
                label, p, r = 'TN', 1, 1  # ASK ARTO

        elif '=' in word:

            # bad segmentation
            if token.is_complex:
                label = 'bad'
                p, r = pr(word, token)

            # false positive
            else:
                label, p, r = 'FP', 0, 0

        else:
            # false negative
            label, p, r = 'FN', 0, 0

        try:
            f1 = (2.0 * p * r) / (p + r)
            f05 = ((0.5**2 + 1.0) * p * r) / ((0.5**2 * p) + r)

        except ZeroDivisionError:
            f1 = 0
            f05 = 0

        return label, word, token.gold_base, p, r, f1, f05

    # -------------------------------------------------------------------------


class MaxEntInput(object):

    def __init__(self):
        self.FinnSeg = FinnSeg(train_coefficients=False, Eval=False)
        self.create_maxent_input()
        self.tableaux = None

    def create_maxent_input(self):
        try:
            open('data/MaxEntInput.csv', 'rb')  # TODO

        except IOError:
            print 'Generating tableaux...'

            tableaux = [
                ['', '', '', 'Nuclei', 'Word-final', 'Harmonic', 'SonSeq', 'Boundaries'],  # noqa
                ['', '', '', 'C1',     'C2',         'C3',       'C4',     'C5'],  # noqa
                ]

            for t in self.FinnSeg.training_tokens:
                Input = t.orth.lower()
                candidates = self._get_candidates(Input)

                try:
                    candidates.remove(t.gold_base)

                except ValueError:
                    pass

                outputs = [t.gold_base, ] + candidates

                # if there are no losing candidates, exclude the token from the
                # tableaux
                if len(outputs) == 1:
                    continue

                violations = self._get_constraint_violations(outputs)

                # append the winner to the tableaux
                winner = [Input.encode('utf-8'), outputs[0], 1] + violations[0]
                tableaux.append(winner)

                # append the losers to the tableaux
                for o, v in zip(outputs, violations)[1:]:
                    tableaux.append(['', o, 0] + v)

            # write tableaux to a csv file
            with open('data/MaxEntInput.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(tableaux)

            self.tableaux = tableaux

    def _get_candidates(self, word):
        candidates = []

        # split the word along any overt delimiters and iterate across the
        # components
        for comp in re.split(r'(-| )', word):

            if len(comp) > 1:

                # use the language model to obtain the component's morphemes
                morphemes = self.FinnSeg.model.viterbi_segment(comp)[0]

                comp_candidates = []
                delimiter_sets = product(['#', 'X'], repeat=len(morphemes) - 1)

                # produce and score each candidate segmentation
                for d in delimiter_sets:
                    candidate = [x for y in izip(morphemes, d) for x in y]
                    candidate = ['#', ] + filter(None, candidate) + ['#', ]
                    comp_candidates.append(self.FinnSeg.score(candidate)[1])

                candidates.append(comp_candidates)

            else:
                candidates.append(comp)

        # convert candidates into string form
        candidates = [''.join(c) for c in product(*candidates)]

        # convert candidates into gold_base form
        for i, c in enumerate(candidates):
            cand = c.replace('#', '=').replace('X', '')
            cand = replace_umlauts(cand)
            candidates[i] = cand

        return candidates

    def _get_constraint_violations(self, outputs):
        # constraints: b  c  d  e  f
        violations = [[0, 0, 0, 0, 0] for i in xrange(len(outputs))]

        for i, output in enumerate(outputs):

            for seg in re.split(r'=|-| ', output):

                if '=' in seg or '-' in seg or ' ' in seg:
                    raise ValueError('Abort!')

                violations[i][0] += 0 if _nuclei(seg) else 1
                violations[i][1] += 0 if _word_final(seg) else 1
                violations[i][2] += 0 if _harmonic(seg) else 1
                violations[i][3] += 0 if _sonseq(seg) else 1

            violations[i][4] += output.count('=')  # asserted boundaries
            violations[i] = map(lambda n: '' if n == 0 else n, violations[i])

        return violations

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # MaxEntInput()

    print 'train_coefficients=False'
    FinnSeg(train_coefficients=False)

    print 'train_coefficients=False, UNK=True'
    FinnSeg(train_coefficients=False, UNK=True)

    print 'train_coefficients=False, absolute=True'
    FinnSeg(train_coefficients=False, absolute=True)  # the best!

    print 'rain_coefficients=False, UNK=True, absolute=True'
    FinnSeg(train_coefficients=False, UNK=True, absolute=True)

    # FinnSeg(a=0.70, b=0.18, c=0.01, d=0.01, e=0.08, f=0.02)

    # no false positives!
    # FinnSeg(a=0.8, c=0.05, d=0.05, f=0.1, absolute=True)
