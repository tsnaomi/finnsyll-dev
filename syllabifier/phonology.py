# coding=utf-8


# Finnish phones --------------------------------------------------------------

# Finnish vowels
VOWELS = [u'i', u'e', u'A', u'y', u'O', u'a', u'u', u'o']
# ä is replaced by A
# ö is replaced by O


# Finnish diphthongs
DIPHTHONGS = [
    u'ai', u'ei', u'oi', u'Ai', u'Oi', u'au', u'eu', u'ou', u'ey', u'Ay',
    u'Oy', u'ui', u'yi', u'iu', u'iy', u'ie', u'uo', u'yO']


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
    return ch in CONSONANTS


def is_cluster(ch):
    return ch in CLUSTERS


def is_diphthong(chars):
    return chars in DIPHTHONGS


def is_long(chars):  # assumes len(chars) == 2
    return chars[0] == chars[1]


def contains_diphthong(chars):
    return any(i for i in DIPHTHONGS if i in chars)


def contains_VV(chars):
    VV_SEQUENCE = [  # no diphthongs and no long vowels
        'iA', 'iO', 'ia', 'io', 'eA', 'eO', 'ea', 'eo', 'Ae', 'AO', 'Aa', 'Au',
        'Ao', 'ye', 'yA', 'ya', 'yu', 'yo', 'Oe', 'OA', 'Oa', 'Ou', 'Oo', 'ae',
        'aA', 'ay', 'aO', 'ao', 'ue', 'uA', 'uy', 'uO', 'ua', 'oe', 'oA', 'oy',
        'oO', 'oa']

    if not contains_VVV(chars):
        VV = [i for i in VV_SEQUENCE if i in chars]

        return VV[0] if VV else False

    return False


def contains_Vu_diphthong(chars):
    if not contains_VVV(chars):
        # includes genuine diphthongs
        Vu_DIPHTHONGS = ['au', 'eu', 'ou', 'iu', 'Au', 'yu', 'Ou']

        return any(i for i in Vu_DIPHTHONGS if i in chars)

    return False


def contains_Vy_diphthong(chars):
    if not contains_VVV(chars):
        # includes genuine diphthongs
        Vy_DIPHTHONGS = ['ey', 'Ay', 'Oy', 'iy', 'ay', 'uy', 'oy']

        return any(i for i in Vy_DIPHTHONGS if i in chars)

    return False


def contains_VVV(chars):
    for i, c in enumerate(chars[:-2]):

        if is_vowel(c):
            return is_vowel(chars[i + 2])

    return False


def replace_umlauts(word, put_back=False):
    '''If put_back is True, put in umlauts; else, take them out!'''
    if put_back:
        word = word.replace(u'A', u'ä').replace(u'A', u'\xc3\xa4')
        word = word.replace(u'O', u'ö').replace(u'O', u'\xc3\xb6')

    else:
        word = word.replace(u'ä', u'A').replace(u'\xc3\xa4', u'A')
        word = word.replace(u'ö', u'O').replace(u'\xc3\xb6', u'O')

    return word


# Phonotactic functions -------------------------------------------------------

def split_syllable(syllable):
    '''Split syllable into a tuple of its parts: onset, nucleus, and coda.'''
    syll = replace_umlauts(syllable).lower()  # put_back?
    nucleus = ''.join([v for v in syll if v in VOWELS])
    onset, nucleus, coda = syll.partition(nucleus)

    return (onset, nucleus, coda)


def is_inseparable_vowels(chars):
    return is_diphthong(chars) or is_long(chars)


def is_consonantal_onset(chars):
    return is_cluster(chars) or is_consonant(chars)


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
