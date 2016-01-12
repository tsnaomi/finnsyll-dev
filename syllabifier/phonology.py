# coding=utf-8

import re

# Finnish phones --------------------------------------------------------------

# Finnish vowels
VOWELS = [u'i', u'e', u'A', u'y', u'O', u'a', u'u', u'o']
# ä is replaced by A
# ö is replaced by O


# Finnish diphthongs
DIPHTHONGS = [
    u'ai', u'ei', u'oi', u'Ai', u'Oi', u'au', u'eu', u'ou', u'ey', u'Ay',
    u'Oy', u'ui', u'yi', u'iu', u'iy', u'ie', u'uo', u'yO', u'oy']


# Finnish consonants
CONSONANTS = [
    u'b', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', u'p',
    u'q', u'r', u's', u't', u'v', u'x', u'z', u"'"]


# Finnish consonant clusters (see Karlsson 1985, #4)
CLUSTERS = [
    u'bl', u'br', u'dr', u'fl', u'fr', u'gl', u'gr', u'kl', u'kr', u'kv',
    u'pl', u'pr', u'cl', u'qv', u'schm']


# Phonemic functions ----------------------------------------------------------

def is_vowel(ch):
    return ch in VOWELS


def is_consonant(ch):
    # return ch in CONSONANTS
    return not is_vowel(ch)  # includes 'w' and other foreign characters


def is_coronal(ch):
    return ch in [u's', u'z', u'd', u't', u'r', u'n', u'l']


def is_sonorant(ch):
    return ch in [u'm', u'n', u'l', u'r']


def is_cluster(ch):
    return ch in CLUSTERS


def is_diphthong(chars):
    return chars in DIPHTHONGS


def is_long(chars):
    return chars == chars[0] * len(chars)


# Vowel harmony ---------------------------------------------------------------

FRONT_VOWELS = [u'A', u'y', u'O']

BACK_VOWELS = [u'a', u'u', u'o']

NEUTRAL_VOWELS = [u'e', u'i']


def is_front(ch):
    return ch in FRONT_VOWELS


def is_back(ch):
    return ch in BACK_VOWELS


def is_neutral(ch):
    return ch in NEUTRAL_VOWELS


DEPTH = {
    'A': 'front',
    'y': 'front',
    'O': 'front',
    'a': 'back',
    'u': 'back',
    'o': 'back',
    }


def is_harmonic(chars):
    # check if the vowels agree in front/back harmony
    vowels = filter(is_vowel, [ch for ch in chars])
    vowels = filter(lambda x: not is_neutral(x), vowels)
    depths = map(lambda x: DEPTH[x], vowels)

    return len(set(depths)) < 2


# Phonotactic functions -------------------------------------------------------

sonorities = {
    # sibilant /s/
    u's': 0,

    # obstruents
    u'p': 1,
    u'b': 1,
    u't': 1,
    u'd': 1,
    u'c': 1,  # TODO
    u'q': 1,  # TODO
    u'x': 1,  # TODO
    u'k': 1,
    u'g': 1,
    u"'": 1,
    u'f': 1,
    u'v': 1,
    u'z': 1,
    u'h': 1,

    # approximants
    u'l': 2,
    u'r': 2,
    u'j': 2,
    u'w': 2,  # TODO

    # nasals
    u'm': 3,
    u'n': 3,
    }


def check_nuclei(word):
    # check if the nucleus is composed of more than one vowel
    return len(filter(is_vowel, word)) > 1


def check_word_final(word):
    # check if the word ends in a vowel or coronal consonant
    return is_vowel(word[-1]) or is_coronal(word[-1])


def check_sonseq(word):
    # check if the word has good sonority peaks

    def is_sloping(seq, rising=True):
        slope = [sonorities.get(s, 0) for s in seq]

        return slope == sorted(list(set(slope)), reverse=not rising)

    parts = re.split(r'([ieAyOauo]+)', word)
    onset, coda = parts[0], parts[-1]

    if not onset or len(onset) == 1 or is_cluster(onset) or is_sloping(onset):
        return not coda or len(coda) == 1 or is_sloping(coda, rising=False)

    return False


# Normalization functions -----------------------------------------------------

def replace_umlauts(word, put_back=False):  # use translate()
    '''If put_back is True, put in umlauts; else, take them out!'''
    if put_back:
        word = word.replace(u'A', u'ä').replace(u'A', u'\xc3\xa4')
        word = word.replace(u'O', u'ö').replace(u'O', u'\xc3\xb6')

    else:
        word = word.replace(u'ä', u'A').replace(u'\xc3\xa4', u'A')
        word = word.replace(u'ö', u'O').replace(u'\xc3\xb6', u'O')

    return word


# Syllable functions ----------------------------------------------------------

def split_syllable(syllable):
    '''Split syllable into a tuple of its parts: onset, nucleus, and coda.'''
    syll = replace_umlauts(syllable).lower()  # put_back?
    nucleus = ''.join([v for v in syll if v in VOWELS])
    onset, nucleus, coda = syll.partition(nucleus)

    return (onset, nucleus, coda)


# Sonority functions ----------------------------------------------------------

def get_sonorities(syllabified_word):
    '''Return the specified word's sonority structure.'''
    syllables = syllabified_word.split('.')
    sonorities = []

    for syllable in syllables:

        try:
            onset, nucleus, coda = split_syllable(syllable)
            sonority = nucleus.title()  # make first character uppercase
            sonorous_syllable = onset + sonority + coda
            sonorities.append(sonorous_syllable)

        except ValueError:
            sonorities.append(u'?')

    sonorities = u'.'.join(sonorities)

    return sonorities


# Weight functions ------------------------------------------------------------

def get_weights(syllabified_word):
    '''Return the specified word's weight structure.'''
    syllables = syllabified_word.split('.')
    weights = [_get_syllable_weight(syll) for syll in syllables]
    weights = u'.'.join(weights)

    return weights


def _get_syllable_weight(syllable):
    '''Return the syllable weight of the given single syllable.'''
    CV = u'L'  # (C)V
    CVC = u'H'  # (C)VC+
    CVV = u'H'  # (C)VV+C*

    try:
        onset, nucleus, coda = split_syllable(syllable)

        # if the nucleus is long
        if len(nucleus) > 1:
            return CVV

        # if a coda is present
        elif coda:
            return CVC

        # if the syllable is light
        return CV

    except ValueError:
        return u'?'


# -----------------------------------------------------------------------------
