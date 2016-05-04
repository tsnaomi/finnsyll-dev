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

    def __repr__(self):
        return self.__str__()

C1 = Constraint('MnWrd', phon.min_word)
C2 = Constraint('SonSeq', phon.sonseq)
C3 = Constraint('Word#', phon.word_final)
C4 = Constraint('Harmonic', phon.harmonic)

# boosts precision at the expense accuracy
C5 = Constraint('#Word', phon.word_initial)

# unranked constraints
CONSTRAINTS = [C1, C2, C3, C4]


# FinnSeg ---------------------------------------------------------------------

class FinnSeg(object):

    def __init__(self, training=TRAINING, validation=VALIDATION, constraints=CONSTRAINTS, filename='data/morfessor', Eval=True, Print=True, annotation=True, approach=None, excl_train_loans=False, excl_val_loans=False):  # noqa
        # note the particulars
        self.approach = approach
        self.annotation = annotation
        self.baseline = approach == 'Baseline'
        self.Print = Print

        # filename of the training text and morfessor model binary file
        self.filename = filename

        # configure the maxent weights
        if approach == 'Maxent':
            # set the NGRAM constraint weight to whatever weight mass is
            # missing from the provided constraint weights
            self.ngram_weight = 1.0 - sum(c.weight for c in constraints)

            # including NGRAM, the constraint weights must sum to 1
            if self.ngram_weight < 0:
                raise ValueError('The constraints weights must sum to 1.')

        # set constraint details
        self.constraints = constraints
        self.constraint_count = len(constraints)
        self.constraint_names = [str(c) for c in constraints]

        # initialize a description of the segmenter
        self.description = {
            'Baseline': '** Baseline (zero segmentation)',
            'OT': '** Optimality Theoretic constraints',
            'Maxent': '** Weighted (maxent) constraints',
            'Unviolable': '** Unviolable constraints',
            None: '** Language model alone',
        }[approach]

        if not self.annotation and not self.baseline:
            # note in the description that annotations have been excluded from
            # training morfessor and the language model
            self.description += '\n** Excl. annotation in training'

            # revise self.filename to indicate that annotations have been
            # excluded from training morfessor and the language model
            self.filename += '-exclAnnotation'

        if excl_train_loans and not self.baseline:
            # filter periphery words from the training data
            training = training.filter_by(is_loanword=False)

            # revise self.filename to indicate that periphery words have been
            # excluded from training the language model
            self.filename += '-exclTrainLoans'

            # note in the description that periphery words have been excluded
            # from training the language model
            self.description += '\n** Excl. loans in training'

        if excl_val_loans and not self.baseline:
            # filter periphery words from the validation data
            validation = validation.filter_by(is_loanword=False)

            # note in the description that periphery words have been excluded
            # from validation
            self.description += '\n** Excl. loans in validation'

        try:

            if approach and not self.baseline:
                # note in the description the constraints and, if applicable,
                # their weights
                self.description += '\n\nConstraints:\n\t%s' % \
                    '\n\t'.join(self.constraint_names)

                self.description += '\n\tNgram=%s' % self.ngram_weight

        except AttributeError:
            pass

        # set the training and validation data
        self.training_tokens = training
        self.validation_tokens = validation

        # set the score_candidates function based on the approach
        self.score_candidates = {
            'OT': self._score_candidates_OT,
            'Maxent': self._score_candidates_maxent,
            'Unviolable': self._score_candidates_unviolable,
            None: self._score_candidates_ngram,
        }.get(approach)

        # set the ngram scoring function
        self.ngram_score = self._stupid_backoff_score

        # train the segmenter
        self.train()

        # evaluate the segmenter on the validaiton set
        if Eval:
            self.evaluate()

    # Train -------------------------------------------------------------------

    def train(self):
        self._train_morfessor()
        self._train_ngrams()

    def _train_morfessor(self):
        io = morfessor.MorfessorIO()
        filename = self.filename + '-training'

        # load the morfessor model, or train and save one if it is nonexistent
        try:
            self.model = io.read_binary_model_file(filename + '.bin')

        except IOError:
            # load training data, or create training data if it is nonexistent
            try:
                train_data = list(io.read_corpus_file(filename + '.txt'))

            except IOError:
                print 'Creating training data...'

                delimiter = ' ' if self.annotation else ''

                tokens = ' '.join([t.gold_base for t in self.training_tokens])
                tokens = tokens.replace('-', ' ').replace('=', delimiter)
                tokens = phon.replace_umlauts(tokens, put_back=True)
                tokens = tokens.lower().encode('utf-8')

                with open(filename + '.txt', 'w') as f:
                    f.write(tokens)

                train_data = list(io.read_corpus_file(filename + '.txt'))

            print 'Training Morfessor model...'

            self.model = morfessor.BaselineModel()
            self.model.load_data(train_data)
            self.model.train_batch(finish_threshold=0.001)
            io.write_binary_model_file(filename + '.bin', self.model)

    def _train_ngrams(self):  # noqa
        filename = self.filename + '-ngrams'

        # load ngrams, or train and save ngrams if they are nonexistent
        try:
            self.ngrams, self.vocab, self.total = \
                pickle.load(open(filename + '.pickle'))

        except IOError:
            print 'Training ngrams...'

            self.ngrams = {}
            self.vocab = set()
            self.total = 0

            if self.annotation:
                base_form = lambda t: t.gold_base

            else:
                base_form = lambda t: t.base

            for t in self.training_tokens:
                stems = re.split(r'=|-| ', base_form(t))

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
                self.vocab.update(word)

                # get unigram, bigram, and trigram counts
                for i, morpheme in enumerate(word):
                    self.ngrams.setdefault(morpheme, 0)
                    self.ngrams[morpheme] += 1
                    self.total += 1

                    if i > 0:
                        bigram = word[i-1] + ' ' + morpheme
                        self.ngrams.setdefault(bigram, 0)
                        self.ngrams[bigram] += 1

                    if i > 1:
                        trigram = word[i-2] + ' ' + bigram
                        self.ngrams.setdefault(trigram, 0)
                        self.ngrams[trigram] += 1

                    # if i > 2:
                    #     fourgram = word[i-3] + ' ' + trigram
                    #     self.ngrams.setdefault(fourgram, 0)
                    #     self.ngrams[fourgram] += 1

            self.vocab = filter(lambda w: w in self.ngrams.keys(), self.vocab)

            # pickle ngrams to file
            pickle.dump(
                [self.ngrams, self.vocab, self.total],
                open(filename + '.pickle', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL,
                )

    # Smooth/score ------------------------------------------------------------

    def _stupid_backoff_score(self, candidate):
        score = 0

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
            score += math.log(C_count * 0.4 * 0.4)
            score -= math.log(self.total + len(self.vocab) + 1)

        return score

    # Segment -----------------------------------------------------------------

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

                candidates = self.score_candidates(comp, candidates)  # noqa

                best = max(candidates)[0]
                candidates = filter(lambda c: c[0] == best, candidates)

                # if multiple candidates have the same score, select the
                # least segmented candidate
                if len(candidates) > 1:
                    candidates.sort(key=lambda c: c[1].count('='))
                    comp = candidates[0][1]

                else:
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

    def _score_candidates_OT(self, comp, candidates):  # noqa

        def ranked(violation_vector):
            # [1, 2, 3, 4] > 1234
            return int(''.join(str(v) for v in violation_vector))

        def partial(violation_vector):
            # assuming MinWrd > Everything:
            # [1, 2, 3, 4] > 19
            return int(''.join(
                [str(violation_vector[0]), str(sum(violation_vector[1:]))]
                ))

            # unranked = sum(violation_vector[1:])

            # if unranked > 9:
            #     ranked = violation_vector[0]
            #     print 'UH-OH'
            # else:
            #     ranked = violation_vector[0] * 10

            # return int(''.join([str(ranked), str(unranked)]))

        def unranked(violation_vector):
            # [1, 2, 3, 4] > 10
            return sum(violation_vector)

        count = len(candidates)

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
                        tableau[i][j] += not const.test(seg)

                # ignore violations when they are incurred by every candidate
                min_violations = min(tableau[i])
                tableau[i] = map(lambda v: v - min_violations, tableau[i])

            # create a violations vector for each candidate:
            # {c1: [0, 1, 2, 4], }
            violations = {
                c[1]:
                    [tableau[r][i] for r in range(self.constraint_count)]
                    for i, c in enumerate(candidates)
                }

            # # violable ranked constraints
            # violations = {k: ranked(v) for k, v in violations.iteritems()}

            # violable partially-ranked constraints
            violations = {k: partial(v) for k, v in violations.iteritems()}

            # # violable unranked constraints
            # violations = {k: unranked(v) for k, v in violations.iteritems()}

            # reduce the candidate set to those that tie in violating the
            # fewest highest-ranked constraints
            Min = min(violations.itervalues())
            candidates = filter(lambda c: violations[c[1]] == Min, candidates)

        return [(self.ngram_score(c1), c2) for c1, c2 in candidates]

    def _score_candidates_maxent(self, comp, candidates):
        # apply the constraint test to a set of segments
        def extend(constraint, candidate):
            return sum(not constraint.test(c) for c in candidate)

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

            # Hayes & Wilson 2008
            score = sum(extend(c, c3) * c.weight for c in self.constraints)
            score += abs(self.ngram_score(c1)) * self.ngram_weight
            score = math.exp(-score)

            scored_candidates.append((score, c2))

        return scored_candidates

    def _score_candidates_unviolable(self, comp, candidates):
        count = len(candidates)

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
                        tableau[i][j] += 0 if const.test(seg) else 1

                # ignore violations when they are incurred by every candidate
                min_violations = min(tableau[i])
                tableau[i] = map(lambda v: v - min_violations, tableau[i])

            # tally the number of violations for each candidate
            violations = {
                c[1]: sum(
                    tableau[r][i] for r in range(self.constraint_count)
                    ) for i, c in enumerate(candidates)
                }

            # filter out candidates that violate any constraints
            candidates = filter(lambda c: not violations[c[1]], candidates)

            # if every candidate violates some constraint, back off to the
            # simplex candidate
            if len(candidates) == 0:
                return [(1.0, comp)]

        return [(self.ngram_score(c1), c2) for c1, c2 in candidates]

    # Inform ------------------------------------------------------------------

    def get_info(self, orth, gold=None):
        loan = ' (loan)' if phon.is_loanword(orth) else ''
        candidates = self.get_candidates(orth)
        inputs = [c[1] for c in candidates]
        morphemes = '{' + ', '.join(
            '%s: %s' % (m, self.ngrams.get(m, 0))
            for m in self.get_morphemes(orth, string_form=False)
            ) + '}'

        info = '\n%s%s\nGold: %s\nWinner: %s\nMorphemes: %s%s' % (
            orth, loan, gold, self.segment(orth), morphemes,
            ' (Morfessor error)' if gold and gold not in inputs else '',
            )

        headers = [''] + self.constraint_names + ['Ngram']
        violations = [[i] + [0] * self.constraint_count + [''] for i in inputs]

        for i, row in enumerate(violations):

            # tally linguistic constaint violations
            for seg in re.split(r'=|-| ', row[0]):
                for j, const in enumerate(self.constraints, start=1):
                    violations[i][j] += 0 if const.test(seg) else 1

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

        # convert morphemes into string form
        if string_form:
            morphemes = '{' + ', '.join(morphemes) + '}'

        return morphemes

    def get_candidates(self, word):
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

        return candidates

    # Evaluate ----------------------------------------------------------------

    def evaluate(self):
        results = self._curate_results()
        self._evaluate(results)

    def _curate_results(self):
        # results include true positives, false positives, true negatives,
        # false negatives, and accurately identified compounds with 'bad'
        # segmentations
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}

        if self.baseline:
            # always return simplex
            get_word = lambda t: t.orth.lower()
        else:
            get_word = lambda t: self.segment(t.orth)

        for t in self.validation_tokens:
            word = get_word(t)
            gold = phon.replace_umlauts(t.gold_base, put_back=True)
            label = self._label(word, gold, t.is_complex)
            results[label].append((word, gold, self.get_morphemes(t.orth)))

            # # if phon.violates_constraint(t.gold_base):
            # if label == 'FP' or label == 'FN':
            #     print label
            #     print self.get_info(t.orth, gold=t.gold_base)
            #     print

        return results

    def _evaluate(self, results):
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

        self.report = (
            '\n'
            '---- Evaluation: FinnSeg ----------------------------------------'
            # '\n\nTrue negatives:\n\t%s'
            # '\n\nTrue positives:\n\t%s'
            '\n\nFalse negatives:\n\t%s'
            '\n\nFalse positives:\n\t%s'
            '\n\nBad segmentations:\n\t%s'
            '\n\n%s'
            '\n\nWord-Level:'
            '\n\tTP:\t%s\n\tFP:\t%s\n\tTN:\t%s\n\tFN:\t%s\n\tBad:\t%s'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tF0.5:\t%s\n\tAcc.:\t%s'
            '\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                # '\n\t'.join(['%s (%s) %s' % t for t in results['TN']]),
                # '\n\t'.join(['%s (%s) %s' % t for t in results['TP']]),
                '\n\t'.join(['%s (%s) %s' % t for t in results['FN']]),
                '\n\t'.join(['%s (%s) %s' % t for t in results['FP']]),
                '\n\t'.join(['%s (%s) %s' % t for t in results['bad']]),
                self.description,
                TP, FP, TN, FN, bad, P, R, F1, F05, ACCURACY,
                )

        if self.Print:
            print self.report

        # save word-level performance
        self.precision = P
        self.recall = R
        self.f1 = F1
        self.accuracy = ACCURACY

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

# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    # FinnSeg(approach='Baseline')

    # FinnSeg()
    # FinnSeg(excl_val_loans=True)
    # FinnSeg(annotation=False)

    FinnSeg(approach='Unviolable')
    # FinnSeg(approach='Unviolable', excl_val_loans=True)

    # FinnSeg(approach='OT')
    # FinnSeg(approach='OT', excl_val_loans=True)

    # # maxent-4-exclLoans
    # maxent_weights = [
    #     9.836141130279886,          # MnWord
    #     10.228120047000191,         # SonSeq
    #     3.071492845484277,          # Word#
    #     8.430433259658745,          # Harmonic
    #     1.7105398264864282,         # Ngram
    #     ]

    # Sum = sum(maxent_weights)
    # weights = [w / Sum for w in maxent_weights]

    # for i in xrange(len(CONSTRAINTS)):
    #     CONSTRAINTS[i].weight = weights[i]

    # FinnSeg(approach='Maxent', constraints=CONSTRAINTS)
    # FinnSeg(approach='Maxent', constraints=CONSTRAINTS, excl_val_loans=True)

    # TRAINING_SUBSET = TRAINING[:14290]
    # FinnSeg(
    #     training=TRAINING_SUBSET,
    #     approach='Unviolable',
    #     filename='data/morfessor-14290Subset',
    #     )
