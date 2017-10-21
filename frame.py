# coding=utf-8

import csv

from datetime import datetime

from app import Token
from syllabifier import _FinnSyll
from utilities import encode


# annotation functions --------------------------------------------------------

def get_annotations(word):
    '''Get the syllabification, vowels, and weights for "word".'''
    row = []
    annotations = _FinnSyll.annotate(word)

    for syll, stress, weights, vowels in annotations:
        row.extend([
            encode(syll),               # syllabification
            get_syll_count(stress),     # syllable count
            stress,                     # stresses
            encode(vowels),             # vowels qualities
            weights,                    # weights
            ])

    row += ('', ) * (20 - len(row))  # fill out empty columns
    row = get_compound_info(word, stress) + row  # prepend compound info

    return row


def get_compound_info(word, stress):
    '''Return "word"'s compound split and its number of constituent words.'''
    split = _FinnSyll.split(word)
    split = '' if word == split else encode(split)

    word_count = str(stress.count('P'))

    if '*' in stress:
        word_count += '*'

    return [split, word_count]


def get_syll_count(stress):
    '''Return the number of syllables in an expression's final word.'''
    try:
        stress = stress[stress.rindex('P'):]

        # if the final word is a vowel-less syllable...
        if '*' in stress:
            return '0*'

        # if the final word is kosher...
        return len(stress)

    # if the entire expression in a vowel-less syllable...
    except ValueError:
        return '0*'


# data frame generation -------------------------------------------------------

def generate_data_frame(filename='./_static/data/aamulehti-1999.csv'):
    '''Generate the data frame!'''

    # write data frame to file
    with open(filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')

        # add the header row with column titles
        writer.writerow(get_headers())

        # add gold rows
        for t in Token.query.filter(Token.is_gold.isnot(None)) \
                .filter_by(is_aamulehti=True) \
                .order_by(Token.is_gold.desc(), Token.orth) \
                .yield_per(1000):
            writer.writerow(get_gold_row(t))

        # add non-gold rows
        for t in Token.query.filter_by(is_gold=None, is_aamulehti=True) \
                .order_by(Token.orth) \
                .yield_per(1000):
            writer.writerow(get_row(t))


def get_headers():
    '''Return column headers.'''
    return [
        # Aamulehti details (word, freq, pos, msd, lemma)
        'word', 'freq', 'pos', 'msd', 'lemma',

        # the lemma's compound split (if any)
        'lem-split',

        # the lemma's word count
        'lem-WC',

        # lemma syllabifications, syllable counts, stresses, vowel qualities,
        # and weights
        'lem-P1', 'lem-C1', 'lem-S1', 'lem-V1', 'lem-W1',
        'lem-P2', 'lem-C2', 'lem-S2', 'lem-V2', 'lem-W2',
        'lem-P3', 'lem-C3', 'lem-S3', 'lem-V3', 'lem-W3',
        'lem-P4', 'lem-C4', 'lem-S4', 'lem-V4', 'lem-W4',

        # the word's compound split (if any)
        'split',

        # the word's word count
        'WC',

        # orth syllabifications, syllable counts, stresses, vowel qualities,
        # and weights
        'P1', 'C1', 'S1', 'V1', 'W1',
        'P2', 'C2', 'S2', 'V2', 'W2',
        'P3', 'C3', 'S3', 'V3', 'W3',
        'P4', 'C4', 'S4', 'V4', 'W4',

        # a boolean indicating whether the syllabifications are accurate
        # (for gold standard rows only)
        'is-gold',

        # a boolean indicating whether the word contain's a deleted /k/
        # (for gold standard rows only)
        'k-stem',

        # gold standard syllabification (for gold standard rows only)
        'gold1', 'gold2', 'gold3'
        ]


def get_gold_row(tok):
    '''Return the token's annotations, plus its gold standard details.'''
    return get_row(tok) + [
        # is-gold
        int(tok.is_gold),

        # k-stem
        '' if tok.is_gold is None else 1 if '[k-deletion' in tok.note else 0,

        # gold standard syllabifications
        encode(tok.syll1),
        encode(tok.syll2),
        encode(tok.syll3),
        ]


def get_row(tok):
    '''Return the token's annotations.'''
    return ([
        # Aamulehti details (word, freq, pos, msd, lemma)
        encode(tok.orth.lower()),
        tok.freq,
        tok.pos.lower(),
        tok.msd.lower(),
        encode(tok.lemma.lower()),

        # the lemma's compound split, syllabifications, counts, weights, etc.
        ] + get_annotations(tok.lemma.lower())

        # the word's compound split, syllabifications, counts, weights, etc.
        + get_annotations(tok.orth.lower())
        )


# misc. -----------------------------------------------------------------------

def timestamp():
    '''Return current UTC time in HH:MM format.'''
    print datetime.utcnow().strftime('%I:%M')


if __name__ == '__main__':
    timestamp()

    generate_data_frame()

    timestamp()
