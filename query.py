# coding=utf-8

import finnsyll as finn
import sys

from tabulate import tabulate
from transition import prune as p


def get_headers(table, prune=False):
    ''' '''
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

    if prune:
        # import pdb; pdb.set_trace()
        li = table[0]
        length = len(li)
        caret = li.index('>')
        headers = HEADERS[:caret] + HEADERS[10: 10 + (length - caret)]
        # headers += HEADERS[10:length]

        if length - caret % 2 == 0:
            headers[-2:] = ['good', 'compound']

        else:
            headers[-1] = 'good'

    else:
        headers = HEADERS

    return headers


def parse(token):
    ''' '''
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
        'good' if token.is_gold else 'bad' if token.is_gold is False else '',
        'C' if token.is_compound else '',  # C for compound
        ]


def tabulate_to_file(query, filename, prune=False):
    ''' '''
    table = [parse(t) for t in query]

    if prune:
        table = p(table)

    headers = get_headers(table, prune=prune)

    table = tabulate(table, headers=headers)

    filename = 'syllabifier/queries/%s.txt' % filename

    with open(filename, 'w') as f:
        f.write(table.encode('utf-8'))


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    prune = '--p' in sys.argv

    tokens = finn.Token.query.filter(finn.Token.is_gold.isnot(None))
    # APPLY FILTERS HERE

    # -------------------------------------------------------------------------

    # GIVE A FILENAME
    filename = None

    tabulate_to_file(tokens, filename, prune=prune)
