# coding=utf-8

import csv
import re

from compound import FinnSeg, TRAINING
from sys import argv


# Maxent Grammar Tool input generator -----------------------------------------

class MaxEntInput(object):

    def __init__(self, excl_loans=False, filename=None, training=TRAINING):
        # if excl_loans is specified, exclude periphery words from training the
        # maxent weights
        if excl_loans:
            training = training.filter_by(is_loanword=False)

        self.tokens = training

        # initialize FinnSeg model
        self.F = FinnSeg(Eval=False)
        # self.F = FinnSeg(Eval=False, excl_train_loans=True)

        # compose an informative filename
        self.filename = 'data/maxent-%s%s%s-input.txt' % (
            filename + '-' if filename else '',
            len(self.F.constraints),
            '-exclLoans' if excl_loans else ''
            )

        # simplify interactions with the MaxEntGrammarTool
        print 'Output filename: ', self.filename.replace('input', 'output')

        self.create_maxent_input()

    def create_maxent_input(self):
        print 'Generating tableaux...'

        # C1, C2, C3, etc.
        abbrev = ['C%s' % n for n in range(1, self.F.constraint_count + 2)]

        # underlying form, candidate, frequency, and constraint columns
        tableaux = [[''] * 3 + self.F.constraint_names + ['Ngram']]
        tableaux += [[''] * 3 + abbrev]

        # for each token, generate a violations tableau and append it to
        # the master tableaux
        for t in self.tokens:

            try:
                tableaux += self.create_tableau(t)

            except TypeError:
                continue

        # write tableaux to a tab-delimited text file
        with open(self.filename, 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(tableaux)

        self.tableaux = tableaux

    def create_tableau(self, token):
        Input = token.orth.lower()
        Gold = token.gold_base
        candidates = self.F.get_candidates(Input)

        # isolate the winning candidate...
        try:
            winner = filter(lambda c: c[1] == Gold, candidates)[0]
            candidates.remove(winner)

        except IndexError:
            winner = (0, Gold)

        # ...and prepend it to the list of candidates
        candidates = [winner] + candidates

        # if there are no losing candidates, exclude the token from the
        # tableaux
        if len(candidates) == 1:
            return None

        return self.accrue_violations(Input, candidates)

    def accrue_violations(self, Input, candidates):
        violations = [
            ['', c[1],  0] + [0] * (self.F.constraint_count + 1)
            for c in candidates
            ]

        # add the input and the output form's frequency next to the winning
        # candidate
        violations[0][0] = Input.encode('utf-8')
        violations[0][2] = 1

        for i, row in enumerate(violations):

            # collect linguistic constaint violations
            for seg in re.split(r'=|-| ', row[1]):
                for j, constraint in enumerate(self.F.constraints, start=3):
                    violations[i][j] += 0 if constraint.test(seg) else 1

            try:
                # collect "ngram" violations
                for comp in re.split(r'-| ', candidates[i][0]):
                    candidate = re.split(r'(X|#)', comp)
                    candidate = ['#'] + candidate + ['#']
                    score = self.F.ngram_score(candidate)
                    violations[i][-1] += round(abs(score))

            # accommodate Morfessor errors
            except TypeError:
                pass

            # replace zero violations with empty strings
            violations[i][3:] = map(
                lambda n: '' if n == 0 else n,
                violations[i][3:],
                )

        return violations


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    excl_loans = '-f' in argv

    try:
        MaxEntInput(
            excl_loans=excl_loans,
            filename=argv[1] if argv[1] != '-f' else None,
            )

    except IndexError:
        MaxEntInput(excl_loans=excl_loans)
