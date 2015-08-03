# coding=utf-8

import finnsyll as finn
import sys

from datetime import datetime
from tabulate import tabulate


def get_headers(grid):
    ''' '''
    if grid:

        HEADERS = [
            'test 1', 'rules 1',
            'test 2', 'rules 2',
            'test 3', 'rules 3',
            'test 4', 'rules 4',
            ]

        li = grid[0]
        length = len(li)
        caret = li.index('>')
        headers = HEADERS[:caret - 1] + ['p / r', '']
        headers += HEADERS[:length - caret - 1]

        if length % 2 == 0:
            headers[-2:] = ['p / r', 'compound']

        else:
            headers[-1] = 'p / r'

        return headers

    return ''


def has_changed(token):
    ''' '''
    return token._is_gold != token.is_gold or token._p_r != token.p_r


def parse(token):
    ''' '''
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


def prune(grid):
    ''' '''

    def transpose(grid):
        return zip(*grid)

    def remove(grid):
        return [list(row) for row in grid if any(row)]

    return remove(transpose(remove(transpose(grid)))) if grid else grid


def transition(pdf=False):
    '''Temporarily re-syllabify verified tokens and create a report.'''
    tokens = finn.Token.query.filter(finn.Token.is_gold.isnot(None))

    for t in tokens:
        t._test_syll1, t._rules1 = t.test_syll1, t.rules1
        t._test_syll2, t._rules2 = t.test_syll2, t.rules2
        t._test_syll3, t._rules3 = t.test_syll3, t.rules3
        t._test_syll4, t._rules4 = t.test_syll4, t.rules4
        t._is_gold = t.is_gold
        t._p_r = t.p_r
        t.inform_base()
        t.detect_if_compound()
        t.syllabify()

    # curate a list of all of the tokens whose gold statuses have changed
    changed = [t for t in tokens if has_changed(t)]

    bad_to_good = prune(map(lambda t: parse(t), filter(lambda t: t.is_gold, changed)))   # noqa
    good_to_bad = prune(map(lambda t: parse(t), filter(lambda t: not t.is_gold, changed)))   # noqa

    good_headers = get_headers(bad_to_good)
    bad_headers = get_headers(good_to_bad)

    report = 'FROM BAD TO GOOD (%s)\n' % len(bad_to_good)
    report += tabulate(bad_to_good, headers=good_headers)
    report += '\n\nFROM GOOD TO BAD (%s)\n' % len(good_to_bad)
    report += tabulate(good_to_bad, headers=bad_headers)
    report += '\n\n%s BAD TOKENS' % tokens.filter_by(is_gold=False).count()

    if pdf:
        filename = 'syllabifier/reports/%s.txt' % str(datetime.utcnow())

        with open(filename, 'w') as f:
            f.write(report.encode('utf-8'))

    print report

    finn.db.session.rollback()


if __name__ == '__main__':
    transition(pdf='--pdf' in sys.argv)
