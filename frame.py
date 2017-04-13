# coding=utf-8

import csv
import re

from datetime import datetime

from app import Token
from syllabifier import _FinnSyll


def encode(u):
    '''Replace umlauts and convert "u" to a byte string.'''
    return u.replace(u'ä', u'{').replace(u'ö', u'|').encode('utf-8')


def get_lemma_info(lemma):
    '''Get the syllabification, vowels, and weights for "lemma".'''
    info = []

    for lemma_syll in _FinnSyll.syllabify(lemma):
        # split the lemma into syllables and word breaks
        # e.g., 'ram.saun mm' > ['ram', 'saun', ' ', 'mm']
        syllables = re.split(r'( |-|_)|\.', lemma_syll, flags=re.I | re.U)
        syllables = [s for s in syllables if s is not None]

        count = 0
        vowels = ''
        weights = ''
        trail = ''

        for syll in syllables:

            try:
                # get the first vowel in the syllable
                vowels += re.search(
                    ur'([ieaouäöy]{1})',
                    syll,
                    flags=re.I | re.U,
                    ).group(1)

                # get the weight of the syllable (light or heavy)
                weights += 'L' if re.match(
                    ur'(^|[^ieaouäöy]+)[ieaouäöy]{1}$',
                    syll,
                    flags=re.I | re.U,
                    ) else 'H'

                # update the syllable count
                count += 1

            except (AttributeError, IndexError):

                # if syll is a word break (e.g., a space or hyphen)
                if syll in ' _-':
                    vowels += ' '
                    weights += ' '

                # if syll is a vowel-less syllable (e.g., the acronym 'MM')
                else:
                    trail = ' *'

        info.extend((
            encode(lemma_syll),                 # lemma syllabification
            str(count) + trail,                 # syllable count
            encode(vowels).upper() + trail,     # vowels
            weights + trail,                    # weights
            ))

    info += ('', ) * (16 - len(info))  # fill out empty columns

    return info


def generate_data_frame(filename='./_static/data/aamulehti-1999.csv'):
    '''Generate the data frame!'''
    data = [[
        # Aamulehti details
        'orth', 'freq', 'pos', 'msd', 'lemma',

        # lemma syllabifications, syllable counts, vowel qualities, and weights
        'lemma1', 'count1', 'vowels1', 'weights1',
        'lemma2', 'count2', 'vowels2', 'weights2',
        'lemma3', 'count3', 'vowels3', 'weights3',
        'lemma4', 'count4', 'vowels4', 'weights4',

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

            # lemma syllabifications, syllable counts, vowel qualities, etc.
            ] + get_lemma_info(t.lemma.lower()) + [

            # syllabifications
            encode(t.test_syll1),
            encode(t.test_syll2),
            encode(t.test_syll3),
            encode(t.test_syll4),

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

            # lemma syllabifications, syllable counts, vowel qualities, etc.
            ] + get_lemma_info(t.lemma.lower()) + [

            # syllabifications
            encode(t.test_syll1),
            encode(t.test_syll2),
            encode(t.test_syll3),
            encode(t.test_syll4),
            ])

    # write data frame to file
    with open(filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


if __name__ == '__main__':
    print datetime.utcnow().strftime('%I:%M')

    generate_data_frame()

    print datetime.utcnow().strftime('%I:%M')
