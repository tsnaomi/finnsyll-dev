# coding=utf-8

import csv

from datetime import datetime

from app import VV


# poetry csv generation -------------------------------------------------------

def generate_poetry_csv(filename='_static/data/poetry.csv'):
    ''' '''

    # write the poetry data to file
    with open(filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')

        # add the header row with column titles
        writer.writerow([])  # TODO

        # add rows
        for vv in VV.query.filter_by(verified=True):
            writer.writerow(get_row(vv))


def get_row(vv):
    ''' '''
    return [
        # poet
        vv._poet.surname,

        # sequence
        vv.sequence,

        # word
        vv._variant._token.orth,

        # ALTERNATIVELY: (ASK ARTO)
        # vv.split,
        # 2 if vv.split == 'split' else 1 if vv.split == 'join' else 0
        # 's' if vv.split == 'split' else 'j' if vv.split == 'join' else 'u'

        # is_joined
        1 if vv.split == 'split' else 0,

        # is_split
        1 if vv.split == 'join' else 0,

        # is_unsure
        1 if vv.split == 'unknown' else 0,

        # position/scansion -- RENAME? (ASK ARTO)
        vv.scansion,

        # sb.follows
        1 if vv.is_heavy else 0,

        # is_word_initial -- NOT CONSIDERING COMPOUNDS, TELL ARTO!
        # IF SEGMENTING COMPOUNDS, DOCUMENT COMPOUND SPLIT
        1 if vv.is_stressed else 0,
        ]


# misc. -----------------------------------------------------------------------

def timestamp():
    '''Return current UTC time in HH:MM format.'''
    print datetime.utcnow().strftime('%I:%M')


if __name__ == '__main__':
    timestamp()

    generate_poetry_csv()

    timestamp()
