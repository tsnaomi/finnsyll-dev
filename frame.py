# coding=utf-8

import csv

from datetime import datetime

from app import Token
from syllabifier import _FinnSyll


def timestamp():
    '''Return current UTC time in HH:MM format.'''
    print datetime.utcnow().strftime('%I:%M')


def encode(u):
    '''Replace umlauts and convert "u" to a byte string.'''
    return u.replace(u'ä', u'{').replace(u'ö', u'|').replace(u'Ä', u'{') \
        .replace(u'Ö', u'|').encode('utf-8')


def get_syll_count(word, vowelless_syll=False):
    '''Return the number of syllables in "word" as a string.'''
    count = sum(word.count(i) for i in ('.', '-', ' ', '_'))

    if not vowelless_syll:
        return str(count + 1)

    return str(count) + vowelless_syll


def get_info(word):
    '''Get the syllabification, vowels, and weights for "word".'''
    info = []

    annotations = _FinnSyll.annotate(word)
    vowelless_syll = '*' if '*' in annotations[0][1] else ''

    for syll, stress, weights, vowels in annotations:
        count = get_syll_count(syll, vowelless_syll)

        info.extend([
            encode(syll),       # syllabification
            count,              # syllable count
            stress,             # stresses
            encode(vowels),     # vowels qualities
            weights,            # weights
            ])

    info += ('', ) * (20 - len(info))  # fill out empty columns

    return info


def generate_data_frame(filename='./_static/data/aamulehti-1999.csv'):
    '''Generate the data frame!'''
    data = [[
        # Aamulehti details
        'orth', 'freq', 'pos', 'msd', 'lemma',

        # lemma syllabifications, syllable counts, stresses, vowel qualities,
        # and weights
        'P:1-lemma', 'C:1-lemma', 'S:1-lemma', 'V:1-lemma', 'W:1-lemma',
        'P:2-lemma', 'C:2-lemma', 'S:2-lemma', 'V:2-lemma', 'W:2-lemma',
        'P:3-lemma', 'C:3-lemma', 'S:3-lemma', 'V:3-lemma', 'W:3-lemma',
        'P:4-lemma', 'C:4-lemma', 'S:4-lemma', 'V:4-lemma', 'W:4-lemma',

        # orth syllabifications, syllable counts, stresses, vowel qualities,
        # and weights
        'P:1', 'C:1', 'S:1', 'V:1', 'W:1',
        'P:2', 'C:2', 'S:2', 'V:2', 'W:2',
        'P:3', 'C:3', 'S:3', 'V:3', 'W:3',
        'P:4', 'C:4', 'S:4', 'V:4', 'W:4',

        # a boolean indicating whether the syllabifications are accurate
        # (for gold standard rows only)
        'is-gold',

        # a boolean indicating whether the word contain's a deleted /k/
        # (for gold standard rows only)
        'k-stem',

        # gold standard syllabification (for gold standard rows only)
        'gold1', 'gold2', 'gold3',
        ], ]

    # add gold rows
    for t in Token.query.filter(Token.is_gold.isnot(None)) \
            .filter_by(is_aamulehti=True) \
            .order_by(Token.is_gold.desc(), Token.orth) \
            .yield_per(1000):

        # create row
        data.append([
            # Aamulehti details
            encode(t.orth.lower()),
            t.freq,
            t.pos.lower(),
            t.msd.lower(),
            encode(t.lemma.lower()),

            # lemma syllabifications, syllable counts, weights, etc.
            ] + get_info(t.lemma.lower())

            # orth syllabifications, syllable counts, weights, etc.
            + get_info(t.orth.lower()) + [

            # is_gold
            int(t.is_gold),

            # k-stem
            '' if t.is_gold is None else 1 if '[k-deletion' in t.note else 0,

            # gold standard syllabifications
            encode(t.syll1),
            encode(t.syll2),
            encode(t.syll3),
            ])

    # add non-gold rows
    for t in Token.query.filter_by(is_gold=None, is_aamulehti=True) \
            .order_by(Token.orth) \
            .yield_per(1000):

        # create row
        data.append([
            # Aamulehti details
            encode(t.orth.lower()),
            t.freq,
            t.pos.lower(),
            t.msd.lower(),
            encode(t.lemma.lower()),

            # lemma syllabifications, syllable counts, weights, etc.
            ] + get_info(t.lemma.lower())

            # orth syllabifications, syllable counts, weights, etc.
            + get_info(t.orth.lower()))

    # write data frame to file
    with open(filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


if __name__ == '__main__':
    timestamp()

    generate_data_frame()

    timestamp()
