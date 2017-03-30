# coding=utf-8

import codecs
import csv

from datetime import datetime

from app import Token


def token_yield():
    '''Yield tokens from the database.'''
    count = Token.query.count()
    start = 0
    end = x = 1000

    while start + x < count:
        for token in Token.query.order_by(
                Token.is_gold,
                Token.orth,
                ).slice(start, end):
            yield token

        start = end
        end += x

    for token in Token.query.order_by(Token.orth).slice(start, count):
        yield token


def generate_data_frame(filename='./_static/data/data-frame.csv'):
    '''Generate the data frame!'''

    def encode(u):
        return u.encode('utf-8')

    data = [[
        # Aamulehti details
        'orth', 'lemma', 'freq', 'pos', 'msd',

        # # number of syllables in lemma
        # 'sylls_in_lemma',

        # # weights and vowel quality for lemma
        # 'lemma_vowels',

        # syllabifications
        'syll1', 'syll2', 'syll3', 'syll4',

        # a boolean indicating whether the syllabifications are accurate
        # (for gold standard rows only)
        'is-gold',

        # a boolean indicating whether the word contain's a deleted /k/
        # (for gold standard rows only)
        'k-stem',

        # gold standard syllabification (for gold standard rows only)
        'gold1', 'gold2', 'gold3',
        ], ]

    for t in token_yield():

        # ask for forgiveness
        try:
            is_gold = int(t.is_gold)

        except TypeError:
            is_gold = ''

        data.append([
            # Aamulehti details
            encode(t.orth.lower()),
            encode(t.readable_lemma()),
            t.freq,
            t.pos.lower(),
            t.msd.lower(),

            # # number of syllables in lemma
            # t.get_sylls_in_lemma(),

            # # weights and vowel quality for lemmas
            # t.get_lemma_vowels(),

            # syllabifications
            encode(t.test_syll1),
            encode(t.test_syll2),
            encode(t.test_syll3),
            encode(t.test_syll4),

            # is_gold
            is_gold,

            # k-stem
            '' if t.is_gold is None else 1 if '[k-deletion' in t.note else 0,

            # gold standard syllabifications
            encode(t.syll1),
            encode(t.syll2),
            encode(t.syll3),
            ])

    with open(filename, 'wb') as f:
        f.write(codecs.BOM_UTF8)
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


if __name__ == '__main__':
    print datetime.utcnow().strftime('%I:%M')

    generate_data_frame()

    print datetime.utcnow().strftime('%I:%M')
