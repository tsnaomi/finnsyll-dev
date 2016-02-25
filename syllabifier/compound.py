# coding=utf-8

# A LOT OF COUNTING AND DIVIDING.

try:
    import cpickle as pickle

except ImportError:
    import pickle

import math
import morfessor
import phonology as phon
import re

from itertools import izip_longest as izip, product
from os import sys, path
from tabulate import tabulate

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # UH OH

import finnsyll as finn


# Data ------------------------------------------------------------------------

TRAINING = finn.training_set()
VALIDATION = finn.dev_set()
TEST = finn.test_set()


# Linguistic constraints ------------------------------------------------------

class Constraint:

    def __init__(self, name, test, weight=0.0):
        self.name = name
        self.test = test
        self.weight = weight

    def __str__(self):
        if self.weight:
            return '%s=%s' % (self.name, self.weight)

        return self.name

C1 = Constraint('MnWrd', phon.min_word)
C2 = Constraint('SonSeq', phon.sonseq)
C3 = Constraint('*VVC', phon.not_VVC)
C4 = Constraint('Harmonic', phon.harmonic)
C5 = Constraint('Word#', phon.word_final)

CONSTRAINTS = [C1, C2, C3, C4, C5]


# FinnSeg ---------------------------------------------------------------------

class FinnSeg(object):

    def __init__(self, training=TRAINING, validation=VALIDATION, Eval=True, filename='data/morfessor', smoothing='stupid', UNK=False, constraints=CONSTRAINTS, unviolable=False, weighted=False):  # noqa
        # the segmenter can take either a violable or unviolable approach
        # (or neither)
        if unviolable and weighted:
            msg = 'May not select both "weighted" and "unviolable".'
            raise ValueError(msg)

        # select the ngram scoring function
        if smoothing.lower() == 'stupid':
            self.ngram_score = self._stupid_backoff_score

        else:
            raise ValueError('Invalid smoothing algorithm specified.')

        # if unviolable is specified, use unviolable linguistic constraints
        if unviolable:
            # set the score-candidates function
            self._score_candidates = self._score_candidates_unviolable

            # set the description for the evaluation report
            self.description = '%s** Unviolable constraints\n\n\t%s' % (
                '** UNK modeling\n' if UNK else '',
                '\n\t'.join([c.__str__() for c in constraints]),
                )

        # if weighted is specified, use weighted linguistic constraints
        elif weighted:
            # set the "ngram constraint" weight to whatever weight mass is
            # missing from the provided constraints
            self.ngram_weight = 1.0 - sum(c.weight for c in constraints)

            # including the "ngram constraint", the constraint weights must
            # collectively sum to 1
            if self.ngram_weight < 0:
                msg = 'The sum of the constraint weights may not exceed 1.'
                raise ValueError(msg)

            self._score_candidates = self._score_candidates_weighted
            self.description = '%s** Weighted constraints\n\n\t%s\n\t%s' % (
                '** UNK modeling\n' if UNK else '',
                '\n\t'.join([c.__str__() for c in constraints]),
                'Ngram=%s' % self.ngram_weight,
                )

        # if neither weighted nor unviolable is specified, use only the
        # language model
        else:
            self._score_candidates = self._score_candidates_ngram
            self.description = '** Language model alone%s' % (
                '\n** UNK modeling\n' if UNK else '',
                )

        # constraint details
        self.weighted = weighted
        self.unviolable = unviolable
        self.constraints = constraints
        self.constraint_count = len(constraints)
        self.constraint_names = [c.name for c in constraints]

        # set training and validation data
        self.training_tokens = training
        self.validation_tokens = validation

        # Morfessor model
        self.model = None

        # filename of training text and morfessor model binary file
        self.filename = filename

        # ngram and open vocabulary containers, etc.
        self.UNK = UNK
        self.ngrams = {}
        self.vocab = set()
        self.total = 0
        self.smoothing = smoothing

        # train segmenter
        self.train()

        # evaluation report
        self.report = None

        # evaluate segmenter on validation set
        if Eval:
            self.evaluate()

    # Training ----------------------------------------------------------------

    def train(self, train_coefficients=True):
        self._train_morfessor()
        self._train_ngrams()

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
                tokens = phon.replace_umlauts(tokens, put_back=True)
                tokens = tokens.lower().encode('utf-8')

                with open(filename + '.txt', 'w') as f:
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

    # Smoothing/Scoring -------------------------------------------------------

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

        score = 100.0 - (score * -1.0)
        score /= 100.0

        return score

    # Segmentation ------------------------------------------------------------

    def segment(self, word):
        token = []

        # split the word along any overt delimiters and iterate across the
        # components
        for comp in re.split(r'(-| )', word.lower()):

            if len(comp) > 1:

                # use the language model to obtain the component's morphemes
                morphemes = self.model.viterbi_segment(comp)[0]

                candidates = []
                delimiter_sets = product(['#', 'X'], repeat=len(morphemes) - 1)

                # produce and score each candidate segmentation
                for d in delimiter_sets:
                    candidate = [x for y in izip(morphemes, d) for x in y]
                    candidate = filter(None, candidate)
                    candidates.append(candidate)

                # # select the best-scoring segmentation
                # comp = self._select_best(comp, candidates)
                candidates = self._score_candidates(comp, candidates)

                # # if multiple candidates have the same score, select the
                # # least segmented candidate
                # best = max(candidates)[0]
                # candidates = filter(lambda c: c[0] == best, candidates)

                # if len(candidates) > 1:
                #     candidates.sort(key=lambda c: c[1].count('='))
                #     candidate = candidates[0][1]

                # else:
                #     candidate = max(candidates)[1]

                comp = max(candidates)[1]
                comp = phon.replace_umlauts(comp, put_back=True)

            token.append(comp)

        # return the segmentation in string form
        return ''.join(token)

    def _score_candidates_ngram(self, comp, candidates):
        # (['#', 'm', 'X', 'm', '#', 'm', '#'], 'mm=m')
        for i, cand in enumerate(candidates):
            cand = ''.join(cand)
            cand = cand.replace('#', '=').replace('X', '')
            cand = phon.replace_umlauts(cand)
            candidates[i] = (['#', ] + candidates[i] + ['#', ], cand)

        return [(self.ngram_score(c1), c2) for c1, c2 in candidates]

    def _score_candidates_unviolable(self, comp, candidates):
        count = len(candidates)

        # a boolean indicating if the component is likely foreign
        f = phon.is_foreign(comp)

        # (['#', 'm', 'X', 'm', '#', 'm', '#'], 'mm=m')
        for i, cand in enumerate(candidates):
            cand = ''.join(cand)
            cand = cand.replace('#', '=').replace('X', '')
            cand = phon.replace_umlauts(cand)
            candidates[i] = (['#', ] + candidates[i] + ['#', ], cand)

        if count > 1:
            #         Cand1   Cand2
            # C1      [0,     0]
            # C2      [0,     0]
            # C3      [0,     0]
            tableau = [[0] * count for i in range(self.constraint_count)]

            for i, const in enumerate(self.constraints):
                for j, cand in enumerate(candidates):
                    for seg in cand[1].split('='):
                        tableau[i][j] += 0 if const.test(seg, f) else 1

                # ignore violations when they are incurred by every candidate
                min_violations = min(tableau[i])
                tableau[i] = map(lambda v: v - min_violations, tableau[i])

            violations = {
                c[1]: sum(tableau[r][i] for r in range(self.constraint_count))
                for i, c in enumerate(candidates)
                }

            # filter out candidates that violate any constraints
            candidates = filter(lambda c: not violations[c[1]], candidates)

            # if every candidate violates some constraint, back off to the
            # simplex candidate
            if len(candidates) == 0:
                return [(1.0, comp)]

        return [(self.ngram_score(c1), c2) for c1, c2 in candidates]

    def _score_candidates_weighted(self, comp, candidates):
        # adjust the constraint test to apply to a set of segments instead
        # of to a single segment
        def extend(const, cand, foreign):
            return 1.0 if all(const.test(c, foreign) for c in cand) else 0.0

        # a boolean indicating if the component is likely foreign
        f = phon.is_foreign(comp)

        # (['#', 'm', 'X', 'm', '#', 'm', '#'], 'mm=m', ['mm', 'm'])
        for i, cand in enumerate(candidates):
            cand = ''.join(cand)
            cand = cand.replace('#', '=').replace('X', '')
            cand = phon.replace_umlauts(cand)
            candidates[i] = (
                ['#', ] + candidates[i] + ['#', ],
                cand,
                cand.split('='),
                )

        scored_candidates = []

        for c1, c2, c3 in candidates:
            score = sum(extend(c, c3, f) * c.weight for c in self.constraints)
            score += self.ngram_score(c1) * self.ngram_weight
            scored_candidates.append((score, c2))

        return scored_candidates

    # Info --------------------------------------------------------------------

    def get_info(self, orth, gold='N/A'):
        foreign = ' (foreign)' if phon.is_foreign(orth) else ''
        candidates = self.get_candidates(orth)
        inputs = [c[1] for c in candidates]

        info = '\n%s%s\nGold: %s\nWinner: %s\nMorphemes: %s%s' % (
            orth, foreign, gold, self.segment(orth), self.get_morphemes(orth),
            ' (Morfessor error)' if gold not in inputs else '',
            )

        headers = [''] + self.constraint_names + ['Ngram']
        violations = [[i] + [0] * self.constraint_count + [''] for i in inputs]

        for i, row in enumerate(violations):

            # tally linguistic constaint violations
            for seg in re.split(r'=|-| ', row[0]):
                for j, const in enumerate(self.constraints, start=1):
                    violations[i][j] += 0 if const.test(seg, foreign) else 1

            # tally "ngram" violations
            for seg in re.split(r'-| ', candidates[i][0]):
                candidate = re.split(r'(X|#)', seg)
                candidate = ['#'] + candidate + ['#']
                violations[i][-1] += ' %s' % self.ngram_score(candidate)

            # replace zeros with empty strings
            violations[i] = map(lambda n: '' if n == 0 else n, violations[i])

        tableau = tabulate(violations, headers=headers).replace('\n', '\n\t')
        info += '\n\n\t%s' % tableau

        return info

    def get_morphemes(self, word, string_form=True):
        morphemes = []

        # split the word along any overt delimiters and iterate across the
        # components
        for comp in re.split(r'(-| )', word.lower()):

            if len(comp) > 1:

                # use the language model to obtain the component's morphemes
                comp = self.model.viterbi_segment(comp)[0]
                morphemes.extend(comp)

            else:
                morphemes.append(comp)

        if string_form:

            # convert morphemes into string form
            morphemes = '{' + ', '.join(morphemes) + '}'

        return morphemes

    def get_candidates(self, word, gold_base_form=False):
        candidates = []

        # split the word along any overt delimiters and iterate across the
        # components
        for comp in re.split(r'(-| )', word.lower()):

            if len(comp) > 1:

                # use the language model to obtain the component's morphemes
                morphemes = self.model.viterbi_segment(comp)[0]

                comp_candidates = []
                delimiter_sets = product(['#', 'X'], repeat=len(morphemes) - 1)

                # produce and score each candidate segmentation
                for d in delimiter_sets:
                    candidate = [x for y in izip(morphemes, d) for x in y]
                    candidate = filter(None, candidate)
                    comp_candidates.append(''.join(candidate))

                candidates.append(comp_candidates)

            else:
                candidates.append(comp)

        # convert candidates into string form
        candidates = [''.join(c) for c in product(*candidates)]

        # convert candidates into gold_base form
        # # (['#', 'm', 'X', 'm', '#', 'm', '#'], 'mm=m')
        for i, c in enumerate(candidates):
            c = c.replace('#', '=').replace('X', '')
            c = phon.replace_umlauts(c)
            candidates[i] = (candidates[i], c)

        if gold_base_form:
            candidates = [cand[1] for cand in candidates]

        return candidates

    # Evalutation -------------------------------------------------------------

    def evaluate(self):
        # results include true positives, false positives, true negatives,
        # false negatives, and accurately identified compounds with 'bad'
        # segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}
        tp, fp, fn = 0.0, 0.0, 0.0

        for t in self.validation_tokens:
            word = self.segment(t.orth)
            gold = phon.replace_umlauts(t.gold_base, put_back=True)

            label = self._word_level_evaluate(word, gold, t.is_complex)
            results[label].append((
                (word, gold, self.get_morphemes(t.orth)),
                t,
                ))

            # if the word is a closed compound
            if label != 'TN':
                tp_, fp_, fn_ = self._boundary_level_evaluate(word, gold)
                tp += tp_
                fp += fp_
                fn += fn_

        TP = len(results['TP'])
        FP = len(results['FP'])
        TN = len(results['TN'])
        FN = len(results['FN'])
        bad = len(results['bad'])

        # calculate precision, recall, and F-measures on a word-by-word basis
        P = float(TP) / (TP + FP + bad)
        R = float(TP) / (TP + FN + bad)
        F1 = (2.0 * P * R) / (P + R)
        F05 = (float(0.5**2) * P * R) / ((0.5**2 * P) + R)
        ACCURACY = float(TP + TN) / (TP + TN + FP + FN + bad)

        # calculate precision, recall, and F-measures on a
        # boundary-by-boundary basis
        tn = ''.join(
            self.get_morphemes(t.orth) for t in self.validation_tokens
            ).count(',') - tp - fp - fn
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2.0 * p * r) / (p + r)
        f05 = (float(0.5**2) * p * r) / ((0.5**2 * p) + r)
        accuracy = float(tp + tn) / (tp + tn + fp + fn)

        # TODO: calculate perplexity

        self.report = (
            '\n'
            '---- Evaluation: FinnSeg ----------------------------------------'
            # '\n\nTrue negatives:\n\t%s'
            # '\n\nTrue positives:\n\t%s'
            # '\n\nFalse negatives:\n\t%s'
            # '\n\nFalse positives:\n\t%s'
            # '\n\nBad segmentations:\n\t%s'
            '\n\n%s'
            '\n\nWord-Level:'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tTN:\t%s\n\tFN:\t%s\n\tBad:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s\n\tAcc.:\t%s'
            '\n\nBoundary-Level:'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tFN:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s\n\tAcc.:\t%s\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                # '\n\t'.join(['%s (%s) %s' % t[0] for t in results['TN']]),
                # '\n\t'.join(['%s (%s) %s' % t[0] for t in results['TP']]),
                # '\n\t'.join(['%s (%s) %s' % t[0] for t in results['FN']]),
                # '\n\t'.join(['%s (%s) %s' % t[0] for t in results['FP']]),
                # '\n\t'.join(['%s (%s) %s' % t[0] for t in results['bad']]),
                self.description,
                TP, FP, TN, FN, bad, P, R, F1, F05, ACCURACY,
                int(tp), int(fp), int(fn), p, r, f1, f05, accuracy,
                )

        print self.report

    def _word_level_evaluate(self, word, gold, is_complex):
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

    def _boundary_level_evaluate(self, word, gold):
        tp, fp, fn = 0, 0, 0

        if word == gold:
            tp += word.count('=')

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

        return tp, fp, fn


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    FinnSeg()
    FinnSeg(unviolable=True)

    maxent_weights = [
        5.305329775582556,          # MnWord
        6.153882174305419,          # SonSeq
        4.163580509823143,          # *VVC
        1.0976921761078926,         # Harmonic
        2.2953719576790137,         # Word#
        1.7446722100824115,         # Ngram
        ]

    Sum = sum(maxent_weights)
    weights = [w / Sum for w in maxent_weights]

    for i in xrange(len(CONSTRAINTS)):
        CONSTRAINTS[i].weight = weights[i]

    FinnSeg(weighted=True, constraints=CONSTRAINTS)

    # F = FinnSeg(Eval=False)
    # for t in VALIDATION:
    #     print F.get_info(t.orth.lower(), t.gold_base)
