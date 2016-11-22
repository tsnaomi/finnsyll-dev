# coding=utf-8

import sys

from datetime import datetime
from tabulate import tabulate

from app import db, get_gold_tokens, Token  # noqa


# supply Test and Query objects with several methods
class Table(object):

    def write_to_file(self):
        '''Write report to file.'''
        with open(self.filename, 'w') as f:
            f.write(self.report.encode('utf-8'))

    def _get_table(self, tokens):
        '''Create a table for tokens.'''
        table = [self.get_table_headers(), ]
        table.extend([self.get_table_row(t) for t in tokens])
        table = self.prune(table)

        return table

    @staticmethod
    def prune(table):
        '''Remove empty columns from table.'''

        def transpose(table):
            return zip(*table)

        def remove(table):
            return [list(row) for row in table if any(row[1:])]

        return remove(transpose(remove(transpose(table)))) if table else table

    @staticmethod
    def tabulate(table):
        '''Tabulate table for pretty printing.'''
        return tabulate(table, headers='firstrow' if table else [])


# test changes to the syllabifier
class Test(Table):

    def __init__(self, pdf=False):
        self.report = None
        self.tokens = get_gold_tokens()
        self.test_transition()

        if pdf:
            # create filename
            date = str(datetime.utcnow())
            self.filename = 'records/tests/%s.txt' % date

            # save report as a pdf file
            self.write_to_file()

        # prevent changes from saving to the database
        db.session.rollback()

    def test_transition(self):
        '''Temporarily re-syllabify gold tokens and generate error report.'''
        # calculate the overall accuracy prior to the transition
        verified = self.tokens.count()
        correct = self.tokens.filter_by(is_gold=True).count()
        pre_acc = (float(correct) / verified) * 100

        # transitioning...
        for t in self.tokens:

            # save previous results
            for attr in ['test_syll', 'rules']:
                for n in range(1, 17):
                    setattr(t, '_' + attr + str(n), getattr(t, attr + str(n)))

            for attr in ['is_gold', 'p_r']:
                setattr(t, '_' + attr, getattr(t, attr))

            # split and syllabify token
            t.split()
            t.syllabify()

        # calculate the overall accuracy after the transition
        correct = self.tokens.filter_by(is_gold=True).count()
        post_acc = (float(correct) / verified) * 100

        # curate a list of all of the tokens whose gold statuses have changed
        changed = [t for t in self.tokens if t._is_gold != t.is_gold]

        # create table for tokens that have changed from bad to good
        bad_to_good = self.get_table(changed, lambda t: t.is_gold)

        # create table for  tokens that have changed from good to bad -- eeek!
        good_to_bad = self.get_table(changed, lambda t: not t.is_gold)

        # compose the report
        self.get_report(pre_acc, post_acc, bad_to_good, good_to_bad)

    def get_report(self, pre_acc, post_acc, bad_to_good, good_to_bad):
        '''Generate an error report.'''
        # tabulate tables
        bad_to_good_table = self.tabulate(bad_to_good)
        good_to_bad_table = self.tabulate(good_to_bad)

        self.report = (
            '\n'
            '---- SYLLABIFIER EVALUATION -------------------------------------'
            '\nPre-accuracy:\t%s\nPost-accuracy:\t%s\n'
            '-----------------------------------------------------------------'
            '\n\nFROM BAD TO GOOD (%s)\n%s'
            '\n\nFROM GOOD TO BAD (%s)\n%s'
            '\n\n%s BAD TOKENS\n'
            ) % (
            round(pre_acc, 4),
            round(post_acc, 4),
            len(bad_to_good) - 1,
            bad_to_good_table,
            len(good_to_bad) - 1,
            good_to_bad_table,
            self.tokens.filter_by(is_gold=False).count(),
            )

        print self.report

    def get_table(self, tokens, filter_func=None):
        '''Create a table for tokens filtered by filter_func.'''
        tokens = filter(filter_func, tokens)
        table = self._get_table(tokens)

        return table

    @staticmethod
    def get_table_headers():
        '''Create table headers.'''
        headers = []

        for i in range(2):
            # test 1, rules 1, test 2, rules 2, test 3, rules 3, etc...
            columns = [['test %s' % n, 'rules %s' % n] for n in range(1, 17)]
            headers.extend(reduce(lambda x, y: x + y, columns) + ['p / r', ])

        headers.insert(9, '>')
        headers.extend(['compound', 'split'])

        return headers

    @staticmethod
    def get_table_row(token):
        '''Extract data from token and return it as a list/row.'''
        row = []

        for onset in ['_', '']:
            for n in range(1, 17):
                attrs = ['test_syll', 'rules']
                row.extend([getattr(token, onset + a + str(n)) for a in attrs])

            row.append(getattr(token, onset + 'p_r'))

        row.insert(9, '>')
        row.append(token.gold_base if token.is_complex else '')
        row.append(token.test_base.decode('utf-8') if token.is_split else '')

        return row


# create tabulated queries
class Query(Table):

    def __init__(self, tokens, filename=None):
        self.report = None
        self.tokens = tokens
        self.get_report()

        if filename:
            # create filename
            date = str(datetime.utcnow())
            self.filename = 'records/queries/%s %s.txt' % (filename, date)

            # save query as a pdf file
            self.write_to_file()

    def get_report(self):
        '''Create a query table.'''
        table = self._get_table(self.tokens)
        self.report = self.tabulate(table)

        print self.report

    @staticmethod
    def get_table_headers():
        '''Create headers for a query table.'''
        columns = [['test %s' % n, 'rules %s' % n] for n in range(1, 17)]
        headers = ['orth', 'freq']
        headers += reduce(lambda x, y: x + y, columns) + ['>', ]
        headers += ['gold %s' % n for n in range(1, 17)]
        headers += ['status', 'compound', 'split']

        return headers

    @staticmethod
    def get_table_row(token):
        '''Extract data from a token and return it as a list/row.'''

        def goldclass(token):
            if token.is_gold:
                return 'good'

            elif token.is_gold is False:
                return 'bad'

        row = [token.orth, token.freq]

        for n in range(1, 17):
            for attr in ['test_syll', 'rules']:
                row.append(getattr(token, attr + str(n)))

        for n in range(1, 17):
            row.append(getattr(token, 'syll' + str(n)))

        row.insert(10, '>')
        row.append(goldclass(token) or '')
        row.append(token.gold_base if token.is_complex else '')
        row.append(token.test_base if token.is_split else '')

        return row


if __name__ == '__main__':
    Test(pdf='--pdf' in sys.argv)

    # tokens = get_gold_tokens()

    # try:
    #     Query(tokens, filename=sys.argv[1])

    # except IndexError:
    #     Query(tokens)
