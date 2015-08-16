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
            'test 5', 'rules 5',
            'test 6', 'rules 6',
            'test 7', 'rules 7',
            'test 8', 'rules 8',
            'test 9', 'rules 9',
            'test 10', 'rules 10',
            'test 11', 'rules 11',
            'test 12', 'rules 12',
            'test 13', 'rules 13',
            'test 14', 'rules 14',
            'test 15', 'rules 15',
            'test 16', 'rules 16',
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
    return token._is_gold != token.is_gold


def parse(token):
    ''' '''
    return [
        token._test_syll1, token._rules1,
        token._test_syll2, token._rules2,
        token._test_syll3, token._rules3,
        token._test_syll4, token._rules4,
        token._test_syll5, token._rules5,
        token._test_syll6, token._rules6,
        token._test_syll7, token._rules7,
        token._test_syll8, token._rules8,
        token._test_syll9, token._rules9,
        token._test_syll10, token._rules10,
        token._test_syll11, token._rules11,
        token._test_syll12, token._rules12,
        token._test_syll13, token._rules13,
        token._test_syll14, token._rules14,
        token._test_syll15, token._rules15,
        token._test_syll16, token._rules16,
        token._p_r,
        '>',
        token.test_syll1, token.rules1,
        token.test_syll2, token.rules2,
        token.test_syll3, token.rules3,
        token.test_syll4, token.rules4,
        token.test_syll5, token.rules5,
        token.test_syll6, token.rules6,
        token.test_syll7, token.rules7,
        token.test_syll8, token.rules8,
        token.test_syll9, token.rules9,
        token.test_syll10, token.rules10,
        token.test_syll11, token.rules11,
        token.test_syll12, token.rules12,
        token.test_syll13, token.rules13,
        token.test_syll14, token.rules14,
        token.test_syll15, token.rules15,
        token.test_syll16, token.rules16,
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
        t._test_syll5, t._rules5 = t.test_syll5, t.rules5
        t._test_syll6, t._rules6 = t.test_syll6, t.rules6
        t._test_syll7, t._rules7 = t.test_syll7, t.rules7
        t._test_syll8, t._rules8 = t.test_syll8, t.rules8
        t._test_syll9, t._rules9 = t.test_syll9, t.rules9
        t._test_syll10, t._rules10 = t.test_syll10, t.rules10
        t._test_syll11, t._rules11 = t.test_syll11, t.rules11
        t._test_syll12, t._rules12 = t.test_syll12, t.rules12
        t._test_syll13, t._rules13 = t.test_syll13, t.rules13
        t._test_syll14, t._rules14 = t.test_syll14, t.rules14
        t._test_syll15, t._rules15 = t.test_syll15, t.rules15
        t._test_syll16, t._rules16 = t.test_syll16, t.rules16
        t._is_gold = t.is_gold
        t._p_r = t.p_r
        t.inform_base()
        t.detect_is_compound()
        t.syllabify()

    # curate a list of all of the tokens whose gold statuses have changed
    changed = [t for t in tokens if has_changed(t)]

    bad_to_good = prune(map(
        lambda t: parse(t),
        filter(lambda t: t.is_gold, changed),
        ))
    good_to_bad = prune(map(
        lambda t: parse(t),
        filter(lambda t: not t.is_gold, changed),
        ))

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

    # from timeit import timeit
    # print timeit(
    #     'transition()',
    #     setup='from __main__ import transition',
    #     number=10,
    #     )
