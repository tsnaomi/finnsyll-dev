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


# Finnish stress (see Anttila 2008)
SON_HIGH = [u'i', u'e', u'u', u'y']
SON_LOW = [u'a', u'A', u'o', u'O']


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

# Return the sonority of a syllable
annotate_sonority = lambda vowel: vowel[0].upper() if vowel else '?'


def get_sonorities(syllables):  # PLUG
    sonorities = []

    for syllable in syllables:
        nucleus = split_syllable(syllable)[1]
        sonority = annotate_sonority(nucleus)
        sonorities.append(sonority)

    return sonorities


# Weight functions ------------------------------------------------------------

CV = 0  # (C)V
CVC = 1  # (C)VC+
CVV = 2  # (C)VV+C*

annotate_weight = lambda w: 'L' if w else 'H'


def get_weights(syllables):  # PLUG
    '''Given a list of syllables, return a list of corresponding weights.'''
    weights = [_get_syllable_weight(syll) for syll in syllables]

    return weights


def _get_syllable_weight(syllable):
    '''Return the syllable weight of a single syllable.'''
    # Syllable weights in increasing order of weight, for deciding which to
    # stress in a sequence of two syllables
    onset, nucleus, coda = split_syllable(syllable)

    # if the nucleus is long
    if len(nucleus) > 1:
        return CVV

    # if a coda is present
    elif coda:
        return CVC

    # if the syllable is light
    return CV


def is_heavy(weight):
    '''Return True if weight is greater than the weight of a light syllable.'''
    return weight > 0


def is_heavier(weight1, weight2):
    '''Return True if weight1 is heavier than weight2.'''
    return weight1 > weight2


# Stress functions ------------------------------------------------------------

UNSTRESSED = 0
PRIMARY = 1
SECONDARY = 2

# Modelled after the CMU Pronouncing Dictionary
annotate_stress = lambda s: 'U' if s == 0 else 'P' if s == 1 else 'S'


def get_stress_pattern(weights):  # PLUG
    '''Given a list of weights, return a list of corresponding stresses.'''
    stress = []
    alternative_stress = []

    wl = len(weights)

    if wl == 1 and not is_heavy(weights[0]):
        stress.append(UNSTRESSED)

    else:
        stress.append(PRIMARY)

    wl -= 1

    for i in xrange(wl):
        stress.append(UNSTRESSED)

    # FINNSYLL: the first syllable is always stressed, and the following
    # syllable is never stressed, so we start with the third syllable
    i = 0

    # Initially stressing odd syllables, i.e., even indices
    stress_parity = 0

    while i < wl:

        # if it is an odd syllable and has the potential to receive stress
        if i % 2 == stress_parity:

            # # indice of following syllable
            n = i + 1

            # FINNSYLL: shift stress foward by one if the following syllable is
            # already stressed (to avoid clash)... or if the following syllable
            # is heavier and nonfinal
            if is_heavier(weights[n], weights[i]) and n < wl:
                stress[n] = SECONDARY

                i = n
                stress_parity = (stress_parity + 1) % 2

            else:
                stress[i] = SECONDARY

        # FINNSYLL: to avoid clash, ignore the following syllable
        i += 2

    if len(weights) > 1 and is_heavy(weights[-1]):
        alternative_stress = _get_alternative_stress_pattern(weights, stress)

    return stress, alternative_stress


def _get_alternative_stress_pattern(weights, stress):  # PLUG
    # FINNSYLL: optionally stress a final heavy syllable... if the preceding
    # syllable is light and stressed, make its stress optional
    if stress[-2] == UNSTRESSED:
        stress = SECONDARY

    elif stress[-2] == SECONDARY and not is_heavy(weights[-2]):
        stress[-1] = SECONDARY
        stress[-2] = UNSTRESSED

    return stress

# -----------------------------------------------------------------------------
