# coding=utf-8

import csv
import re

from new_compound import FinnSeg
from datetime import datetime
from phonology import is_foreign
from sys import argv


F = FinnSeg(Eval=False)

date = str(datetime.utcnow())


# MaxEnt Harmonic Grammar -----------------------------------------------------

class MaxEntInput(object):

    def __init__(self, ignore_foreign=False, training=True, filename=None):
        self.ignore_foreign = ignore_foreign
        self.tokens = F.training_tokens if training else F.validation_tokens

        filename = date if not filename else filename.capitalize()
        filename = ('Training' if training else 'Validation') + filename
        self.filename = 'data/MaxEnt-' + filename + '-Input.txt'

        self.create_maxent_input()
        self.tableaux = None

    def create_maxent_input(self):
        try:
            open(self.filename, 'rb')

        except IOError:
            print 'Generating tableaux...'

            # C1, C2, C3, etc.
            abbrev = ['C%s' % n for n in range(1, F.constraint_count + 2)]

            # underlying form, candidate, frequency, and constraint columns
            tableaux = [[''] * 3 + F.constraint_names + ['Ngram']]
            tableaux += [[''] * 3 + abbrev]

            # for each token, generate a violations tableau and append it to
            # the master tableaux
            for t in self.tokens:

                try:
                    tableaux += self.prepare_tableau(t)

                except TypeError:
                    continue

            # write tableaux to a tab-delimited text file
            with open(self.filename, 'wb') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(tableaux)

            self.tableaux = tableaux

    def prepare_tableau(self, token):
        Input = token.orth.lower()
        Gold = token.gold_base

        # ignore foreign words in the constraint weighting
        if self.ignore_foreign and is_foreign(Input):
            return None

        candidates = F.get_candidates(Input)

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
            ['', c[1],  0] + [0] * (F.constraint_count + 1)
            for c in candidates
            ]

        # add the input and the output form's frequency next to the winning
        # candidate
        violations[0][0] = Input.encode('utf-8')
        violations[0][2] = 1

        for i, row in enumerate(violations):

            # collect linguistic constaint violations
            for seg in re.split(r'=|-| ', row[1]):
                for j, const in enumerate(F.constraints, start=3):
                    violations[i][j] += 0 if const.test(seg, False) else 1

            try:
                # collect "ngram" violations
                for comp in re.split(r'-| ', candidates[i][0]):
                    candidate = re.split(r'(X|#)', comp)
                    candidate = ['#'] + candidate + ['#']
                    score = F.ngram_score(candidate)
                    violations[i][-1] += round(100 - score * 100)

            # accomodate Morfessor errors
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
    ignore_foreign = '-f' in argv
    training = '-v' not in argv

    try:
        MaxEntInput(
            ignore_foreign=ignore_foreign,
            training=training,
            filename=argv[1] if argv[1] not in '-f -v' else None,
            )

    except IndexError:
        MaxEntInput(ignore_foreign=ignore_foreign, training=training)
