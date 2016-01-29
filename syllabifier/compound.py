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

from collections import Counter, defaultdict, namedtuple
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

# Data ------------------------------------------------------------------------

TRAINING = finn.training_set()
VALIDATION = finn.dev_set()
TEST = finn.test_set()


# Linguistic constraints ------------------------------------------------------

Constraint = namedtuple('Constraint', 'name test unviolable')
Constraint.__str__ = lambda self: self.name


# unviolable
def nuclei(segments):
    return 1 if all(_nuclei(seg) for seg in segments) else 0


# unviolable
def sonseq(segments):
    return 1 if all(_sonseq(seg) for seg in segments) else 0


# violable
def harmonic(segments):
    return 1 if all(_harmonic(seg) for seg in segments) else 0


# violable
def word_final(segments):
    return 1 if all(_word_final(seg) for seg in segments) else 0


# violable
def boundaries(segments):
    return 1.0 / len(segments)

CONSTRAINTS = [
    Constraint('nuclei', nuclei, True),
    Constraint('sonseq', sonseq, True),
    Constraint('harmonic', harmonic, False),
    Constraint('word-final', word_final, False),
    Constraint('boundaries', boundaries, False),
    ]


# FinnSeg ---------------------------------------------------------------------

class FinnSeg(object):

    def __init__(self, training=TRAINING, validation=VALIDATION, Eval=True,
                 filename='data/morfessor', train_weights=True,
                 smoothing='stupid', unviolable=False, UNK=False,
                 constraints=CONSTRAINTS, weights=None):

        # preprend "ngram" constraint to the list of constraints
        ngram = Constraint('ngram', lambda: None, False)
        constraints = [ngram, ] + constraints

        if weights:
            # if weights do not sum to 1 or if the constraint-to-weight ratio
            # is off, throw an error
            if len(weights) != len(constraints) or round(sum(weights), 4) != 1:
                raise ValueError('Uh oh.')

        else:
            # create weights, with ngrams having a weight of 1.0, if no weights
            # were given
            weights = [1.0, ] + [0.0 for c in xrange(len(constraints))]

        # train weights if ngrams do not receive a weight of 1.0, or if
        # train_weights is specified
        self.train_weights = False if weights[0] == 1.0 else train_weights

        # if unviolable is specified, treat nuclei and sonority sequencing
        # tests as unviolable constraints
        self.unviolable = unviolable

        # if unviolable is specified and the weights are untrained,
        # re-shape the weights of violables constraints to sum to 1
        if unviolable and not self.train_weights:
            cw = zip(constraints, weights)
            divisor = float(sum(w for c, w in cw if not c.unviolable))

            for i, (c, w) in enumerate(cw):
                weights[i] = w / divisor if not c.unviolable else w

        self.weighted_constraints = zip(constraints, weights)

        # set training and validation data
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

        # train segmenter
        self.train()

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

        if self.train_weights:
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
        # equal to the score returned by self._score_ngrams(), unless
        # self.unviolable is True
        if self.weighted_constraints[0][1] < 1 or self.unviolable:

            # convert score from negative to positive
            score = 100.0 - (score * -1.0)
            score /= 100.0

            # get segmentation as list: e.g., 'book=worm' > ['book', 'worm']
            segments = replace_umlauts(candidate).split('=')

            # multiply the ngram score by the ngram weight
            score *= self.weighted_constraints[0][1]

            # score phonotactic features
            for constraint, weight in self.weighted_constraints[1:]:
                score_ = constraint.test(segments)

                # treat nuclei and sonority sequencing tests as unviolable
                # constraints
                if self.unviolable and constraint.unviolable and not score_:
                    return 0, candidate

                # compute composite score
                score += score_ * weight

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
        for component in re.split(r'(-| )', word):

            if len(component) > 1:

                # use the language model to obtain the component's morphemes
                comp = map(
                    lambda m: m.replace(' ', '').replace('-', ''),
                    self.model.viterbi_segment(component.lower())[0],
                    )
                morphemes.extend(comp)

            else:
                morphemes.append(comp)

        # convert morphemes into string form
        morphemes = '_'.join(morphemes)

        return morphemes

    def get_candidates(self, word, gold_base_form=True):
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

        if gold_base_form:

            # convert candidates into gold_base form
            for i, c in enumerate(candidates):
                candidates = c.replace('#', '=').replace('X', '')
                candidates = replace_umlauts(candidates)
                candidates[i] = candidates

        return candidates

    # Evalutation -------------------------------------------------------------

    def evaluate(self):
        # results include true positives, false positives, true negatives,
        # false negatives, and accurately identified compounds with 'bad'
        # segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}
        precision, recall, tp, fp, fn = [], [], 0, 0, 0

        for t in self.validation_tokens:
            word = self.segment(t.orth)
            gold = replace_umlauts(t.gold_base, put_back=True)

            label = self._word_level_evaluate(word, gold, t.is_complex)
            results[label].append((word, gold))

            if label != 'TN':
                p, r, tp_, fp_, fn_ = self._boundary_level_evaluate(word, gold)
                precision.append(p)
                recall.append(r)
                tp += tp_
                fp += fp_
                fn += fn_

        TP = len(results['TP'])
        FP = len(results['FP'])
        TN = len(results['TN'])
        FN = len(results['FN'])
        bad = len(results['bad'])

        # calculate precision, recall, and F-measures on a word-by-word basis
        P = (TP * 1.0) / (TP + FP + bad)
        R = (TP * 1.0) / (TP + FN + bad)
        F1 = (2.0 * P * R) / (P + R)
        F05 = ((0.5**2 + 1.0) * P * R) / ((0.5**2 * P) + R)

        # calculate precision, recall, and F-measures on a
        # boundary-by-boundary basis
        p = float(sum(precision)) / len(precision)
        r = float(sum(recall)) / len(recall)
        f1 = (2.0 * p * r) / (p + r)
        f05 = ((0.5**2 + 1.0) * p * r) / ((0.5**2 * p) + r)

        # TODO: calculate perplexity

        self.report = (
            '\n'
            '---- Evaluation: FinnSeg ----------------------------------------'
            # '\n\nTrue negatives:\n\t%s'
            # '\n\nTrue positives:\n\t%s'
            # '\n\nFalse negatives:\n\t%s'
            # '\n\nFalse positives:\n\t%s'
            # '\n\nBad segmentations:\n\t%s'
            '%s%s'
            '\n\nWeights:\n\t%s'
            '\n\nWord-Level:'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tTN:\t%s\n\tFN:\t%s\n\tBad:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s'
            '\n\nBoundary-Level:'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tFN:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                # '\n\t'.join(['%s (%s)' % t for t in results['TN']]),
                # '\n\t'.join(['%s (%s)' % t for t in results['TP']]),
                # '\n\t'.join(['%s (%s)' % t for t in results['FN']]),
                # '\n\t'.join(['%s (%s)' % t for t in results['FP']]),
                # '\n\t'.join(['%s (%s)' % t for t in results['bad']]),
                '\n\n** Unviolable constraints' if self.unviolable else '',
                '\n\n** UNK modeling' if self.UNK else '',
                '\n\t'.join(['%s=%s' % t for t in self.weighted_constraints]),
                TP, FP, TN, FN, bad, P, R, F1, F05,
                tp, fp, fn, p, r, f1, f05,
                )

        print self.report

    def _word_level_evaluate(self, word, gold, is_complex):
        # true positive or true negative
        if word == gold:
            label = 'TP' if is_complex else 'TN'

        # bad segmentation or false positive
        elif '=' in word:
            label = 'bad' if is_complex else 'FP'

        # false negative
        else:
            label = 'FN'

        return label

    def _boundary_level_evaluate(self, word, gold):
        tp = 0
        fp = 0
        fn = 0

        if word == gold:

            tp += word.count('=') + word.count(' ') + word.count('-')
            precision = 1
            recall = 1

        else:
            gold_index = 0
            word_index = 0

            for i in range(max(len(word), len(gold))):
                w = word[i - word_index]
                g = gold[i - gold_index]

                if g == w and g != '=':
                    continue

                elif g == w == '=':
                    tp += 1

                elif g == '=':
                    fn += 1
                    word_index += 1

                else:
                    fp += 1
                    gold_index += 1

            try:
                precision = float(tp) / (tp + fp)
                recall = float(tp) / (tp + fn)

            except ZeroDivisionError:
                precision = 0
                recall = 0

        return precision, recall, tp, fp, fn


# MaxEnt ----------------------------------------------------------------------

class MaxEntInput(object):

    def __init__(self):
        self.FinnSeg = FinnSeg(train_coefficients=False, Eval=False)
        self.create_maxent_input()
        self.tableaux = None

    def create_maxent_input(self):
        try:
            open('data/MaxEntInputTest.csv', 'rb')

        except IOError:
            print 'Generating tableaux...'

            # underlying forms, candidates, and frequency columns
            col3 = ['', '', '']

            # constraint columns
            tableaux = [
                col3 + ['Ngram', 'Nuclei', 'Word-final', 'Harmonic', 'SonSeq'],
                col3 + ['C1',    'C2',     'C3',         'C4',       'C5'],
                ]

            for t in self.FinnSeg.training_tokens:
                Input = t.orth.lower()
                candidates = self._get_candidates(Input)

                # delete winning candidate to preprend it to outputs
                try:
                    winner_violations = candidates.get(t.gold_base, 0)
                    del candidates[t.gold_base]

                except (ValueError, KeyError):
                    pass

                outputs = [(t.gold_base, winner_violations), ]
                outputs += candidates.items()

                # if there are no losing candidates, exclude the token from the
                # tableaux
                if len(outputs) == 1:
                    continue

                all_violations = self._get_constraint_violations(outputs)

                # append the winner to the tableaux
                Input = Input.encode('utf-8')
                tableaux.append([Input, outputs[0][0], 1] + all_violations[0])

                # append the losers to the tableaux
                for output, violations in zip(outputs, all_violations)[1:]:
                    tableaux.append(['', output[0], 0] + violations)

            # write tableaux to a csv file
            with open('data/MaxEntInput.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(tableaux)

            self.tableaux = tableaux

    def _get_candidates(self, word):
        scored_candidates = {}
        scores = {}
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

                    # collect ngram violations before the candidate component
                    # is converted into string form
                    score = self.FinnSeg._stupid_backoff_score(candidate)
                    score *= -1.0

                    # convert candidate into string form
                    candidate = ''.join(candidate[1:-1])
                    candidate = candidate.replace('#', '=').replace('X', '')
                    candidate = replace_umlauts(candidate)

                    comp_candidates.append(candidate)
                    scores[candidate] = round(score)

                candidates.append(comp_candidates)

            else:
                candidates.append(comp)

        # generate full candidates
        candidates = [c for c in product(*candidates)]

        # collect ngram violations for full candidates
        for candidate in candidates:
            violations = 0

            for morphemes in candidate:
                violations += scores.get(morphemes, 0)

            # convert fulls candidates into string form
            candidate = ''.join(candidate)
            scored_candidates[candidate] = violations

        return scored_candidates

    def _get_constraint_violations(self, outputs):
        # constraints: a  b  c  d  e
        violations = [[0, 0, 0, 0, 0] for i in xrange(len(outputs))]

        for i, output in enumerate(outputs):
            violations[i][0] = int(output[1])

            for seg in re.split(r'=|-| ', output[0]):
                violations[i][1] += 0 if _nuclei(seg) else 1
                violations[i][2] += 0 if _word_final(seg) else 1
                violations[i][3] += 0 if _harmonic(seg) else 1
                violations[i][4] += 0 if _sonseq(seg) else 1

            violations[i] = map(lambda n: '' if n == 0 else n, violations[i])

        return violations

    # TODO: extract weights for FinnSeg

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # MaxEntInput()

    FinnSeg(train_weights=False)
    # FinnSeg(train_weights=False, UNK=True)
    # FinnSeg(train_weights=False, unviolable=True)  # No. 2
    # FinnSeg(train_weights=False, UNK=True, unviolable=True)
    # FinnSeg(weights=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    # # no false positives!
    # FinnSeg(weights=[0.80, 0.0, 0.0, 0.05, 0.05, 0.1], unviolable=True)

    # # MaxEnt: 5 constraints
    # maxent_weights = [
    #     1.5631777946044625,         # ngram
    #     1.3520000193826711,         # nuclei
    #     8.93479259279159,           # sonseq
    #     1.3877787807814457E-17,     # harmonic
    #     4.904941422169471,          # word-final
    #     0.0,                        # boundaries
    #     ]
    # Sum = sum(maxent_weights)
    # weights = [w / Sum for w in maxent_weights]
    # FinnSeg(weights=weights)  # No. 1
