# coding=utf-8

import app as finn
import sys

from datetime import datetime
from tabulate import tabulate


# supply Transition and Query objects with a _prune() method
class Base(object):

    @staticmethod
    def _prune(table):
        '''Remove empty columns from an error report table.'''

        def transpose(table):
            return zip(*table)

        def remove(table):
            return [list(row) for row in table if any(row)]

        return remove(transpose(remove(transpose(table)))) if table else table


# test changes to the syllabifier or compounder
class Transition(Base):

    def __init__(self, pdf=False, compounding_eval=False, gold_base=False):
        self.gold_base = gold_base
        self.report = self.transition()

        if pdf:
            # save report as a pdf file
            self.write_to_file(compounding_eval=compounding_eval)

        # prevent changes from saving to the database
        finn.db.session.rollback()

    def transition(self):
        '''Temporarily re-syllabify gold tokens and generate error report.'''
        if self.gold_base:
            tokens = finn.Token.query.filter(finn.Token.is_complex.isnot(None))

        else:
            tokens = finn.get_gold_tokens()

        # calculate the overall accuracy prior to the transition
        verified = tokens.count()
        gold = tokens.filter_by(is_gold=True).count()
        pre_accuracy = (float(gold) / verified) * 100

        # calculate compound identification accuracy prior to the transition
        try:
            gold_compounds = finn.get_gold_compounds()
            gold_compounds_count = gold_compounds.count()
            pre_test_compounds_count = gold_compounds.filter_by(
                is_test_compound=True).count()
            pre_compound_accuracy = (float(pre_test_compounds_count) /
                                     gold_compounds_count) * 100 or 'N/A'
            pre_false_positives = tokens.filter_by(
                is_test_compound=True).filter_by(is_complex=False).count()
            pre_false_negatives = tokens.filter_by(
                is_test_compound=False).filter_by(is_complex=True).count()

        except ZeroDivisionError:
            pre_compound_accuracy = 'N/A'
            pre_false_positives = 'N/A'
            pre_false_negatives = 'N/A'

        # transitioning...
        for t in tokens:
            t._test_syll1, t._rules1 = t.test_syll1, t.rules1
            t._test_syll2, t._rules2 = t.test_syll2, t.rules2
            t._test_syll3, t._rules3 = t.test_syll3, t.rules3
            t._test_syll4, t._rules4 = t.test_syll4, t.rules4
            t._is_gold = t.is_gold
            t._p_r = t.p_r
            t.inform_base()
            t.detect_is_compound()
            t.syllabify(gold_base=self.gold_base)

        # calculate the overall accuracy after the transition
        gold = tokens.filter_by(is_gold=True).count()
        post_accuracy = (float(gold) / verified) * 100

        # calculate compound identification accuracy after the transition
        try:
            post_test_compounds_count = gold_compounds.filter_by(
                is_test_compound=True).count()
            post_compound_accuracy = (float(pre_test_compounds_count) /
                                      gold_compounds_count) * 100
            post_false_positives = tokens.filter_by(
                is_test_compound=True).filter_by(is_complex=False).count()
            post_false_negatives = tokens.filter_by(
                is_test_compound=False).filter_by(is_complex=True).count()

        except ZeroDivisionError:
            post_compound_accuracy = 'N/A'
            post_false_positives = 'N/A'
            post_false_negatives = 'N/A'

        # curate a list of all of the tokens whose gold statuses have changed
        changed = [t for t in tokens if t._is_gold != t.is_gold]

        # separate tokens that have changed from bad to good
        bad_to_good = self._prune(map(
            lambda t: self._parse(t),
            filter(lambda t: t.is_gold, changed),
            ))

        # separate tokens that have changed from good to bad -- eeek!
        good_to_bad = self._prune(map(
            lambda t: self._parse(t),
            filter(lambda t: not t.is_gold, changed),
            ))

        # get pruned headers for table
        good_headers = self._get_headers(bad_to_good)
        bad_headers = self._get_headers(good_to_bad)

        # compose the report
        report = (
            '\n'
            '%s'
            '---- EVALUATION -------------------------------------------------'
            '\nBEFORE\nOverall accuracy: %s'
            '\nCompound identification accuracy: %s (%s/%s)'
            '\nFalse negatives: %s\nFalse positives: %s'
            '\n\nAFTER\nOverall accuracy: %s'
            '\nCompound identification accuracy: %s (%s/%s)'
            '\nFalse negatives: %s\nFalse positives: %s\n'
            '-----------------------------------------------------------------'
            '\n\nFROM BAD TO GOOD (%s)\n%s\n\nFROM GOOD TO BAD (%s)\n%s'
            '\n\n%s BAD TOKENS'
            '\n'
            ) % (
                'GOLD BASE SYLLABIFICATIONS\n\n' if self.gold_base else '',
                round(pre_accuracy, 4),
                pre_compound_accuracy,
                pre_test_compounds_count,
                gold_compounds_count,
                pre_false_negatives,
                pre_false_positives,
                round(post_accuracy, 4),
                post_compound_accuracy,
                post_test_compounds_count,
                gold_compounds_count,
                post_false_negatives,
                post_false_positives,
                len(bad_to_good),
                tabulate(bad_to_good, headers=good_headers),  # create table
                len(good_to_bad),
                tabulate(good_to_bad, headers=bad_headers),  # create table
                tokens.filter_by(is_gold=False).count()
            )

        # if evaluating the gold_base, show which words are still bad
        if self.gold_base:
            still_bad = self._prune(map(
                lambda t: self._bad_parse(t),
                tokens.filter_by(is_gold=False),
                ))
            report += tabulate(still_bad)

        print report

        return report

    def write_to_file(self, compounding_eval=False):
        '''Write report to file.'''
        date = str(datetime.utcnow())

        if compounding_eval:
            filename = 'syllabifier/compound/evaluations/%s.txt' % date

        else:
            filename = 'printouts/reports/%s.txt' % date

        with open(filename, 'w') as f:
            f.write(self.report.encode('utf-8'))

    @staticmethod
    def _get_headers(table):
        '''Create headers for an error report table.'''
        if table:
            HEADERS = [
                'test 1', 'rules 1',
                'test 2', 'rules 2',
                'test 3', 'rules 3',
                'test 4', 'rules 4',
                ]

            li = table[0]
            length = len(li)
            caret = li.index('>')
            headers = HEADERS[:caret - 1] + ['p / r', '']
            headers += HEADERS[:length - caret - 1]

            if length % 2 == 0:
                headers[-2:] = 'p / r', 'compound'

            else:
                headers[-1] = 'p / r'

            return headers

        return ''

    @staticmethod
    def _parse(token):
        '''Extract data from a token and return it as a list.'''
        return [
            token._test_syll1, token._rules1,
            token._test_syll2, token._rules2,
            token._test_syll3, token._rules3,
            token._test_syll4, token._rules4,
            token._p_r,
            '>',
            token.test_syll1, token.rules1,
            token.test_syll2, token.rules2,
            token.test_syll3, token.rules3,
            token.test_syll4, token.rules4,
            token.p_r,
            'C' if token.is_compound else '',
            ]

    @staticmethod
    def _bad_parse(token):
        '''Extract data from a token and return it as a list.'''
        return [
            token._test_syll1, token._rules1,
            token._test_syll2, token._rules2,
            token._test_syll3, token._rules3,
            token._test_syll4, token._rules4,
            '>',
            token.syll1,
            token.syll2,
            token.syll3,
            token.syll4,
            token.p_r,
            'C' if token.is_compound else '',
            ]


# create tabulated queries
class Query(Base):

    def __init__(self, tokens, filename=None, pdf=None):
        self.tokens = tokens
        self.table = self.create_table()

        if filename and pdf:
            # save query as a pdf file
            self.write_to_file(tokens, filename)

    def create_table(self):
        '''Create a query table.'''
        table = self._prune([self._parse(t) for t in self.tokens])
        headers = self._get_headers(table)
        table = tabulate(table, headers=headers)

        print table

        return table

    def write_to_file(self):
        '''Write tabulated query to file.'''
        filename = 'prinouts/queries/%s.txt' % self.filename

        with open(filename, 'w') as f:
            f.write(self.table.encode('utf-8'))

    @staticmethod
    def _get_headers(table):
        '''Create headers for a query table.'''
        HEADERS = [
            'freq',
            'orth',
            'test 1',
            'rules 1',
            'test 2',
            'rules 2',
            'test 3',
            'rules 3',
            'test 4',
            'rules 4',
            '',
            'gold 1',
            'gold 2',
            'gold 3',
            'gold 4',
            'good',
            'compound',
            ]

        li = table[0]
        length = len(li)
        caret = li.index('>')
        headers = HEADERS[:caret] + HEADERS[10: 10 + (length - caret)]

        if length - caret % 2 == 0:
            headers[-2:] = 'good', 'compound'

        else:
            headers[-1] = 'good'

        return headers

    @staticmethod
    def _parse(token):
        '''Extract data from a Token and return it as a list.'''
        return [
            token.freq,
            token.orth,
            token.test_syll1,
            token.rules1,
            token.test_syll2,
            token.rules2,
            token.test_syll3,
            token.rules3,
            token.test_syll4,
            token.rules4,
            '>',
            token.syll1,
            token.syll2,
            token.syll3,
            token.syll4,
            'good' if token.is_gold else 'bad' if token.is_gold is False else '',  # noqa
            'C' if token.is_compound else '',
            ]


if __name__ == '__main__':
    Transition(
        pdf='--pdf' in sys.argv,
        compounding_eval='-c' in sys.argv,
        gold_base='-g' in sys.argv,
        )

    # # SUPPLY TOKENS AND FILENAME HERE
    # tokens = finn.get_gold_tokens()
    # filename = None

    # Query(tokens, filename, pdf='--pdf' in sys.argv)
