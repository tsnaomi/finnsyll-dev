# coding=utf-8


# Finnish phones --------------------------------------------------------------

# Finnish vowels
VOWELS = ['i', 'e', 'A', 'y', 'O', 'a', 'u', 'o']
# VOWELS = ['i', 'e', 'ä', 'y', 'ö', 'a', 'u', 'o']
# \"{a} = a with Umlaut (ä) (\xc3\xa4)
# \"{o} = o with Umlaut (ö) (\xc3\xb6)


# Finnish diphthongs
DIPHTHONGS = [
    'ai', 'ei', 'oi', 'Ai', 'Oi', 'au', 'eu', 'ou', 'ey', 'Ay', 'Oy', 'ui',
    'yi', 'iu', 'iy', 'ie', 'uo', 'yO']
# DIPHTHONGS = [
#     'ai', 'ei', 'oi', 'äi', 'öi', 'au', 'eu', 'ou', 'ey', 'äy', 'öy', 'ui',
#     'yi', 'iu', 'iy', 'ie', 'uo', 'yö']


# Finnish consonants
CONSONANTS = [
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's',
    't', 'v', 'x', 'z', "'"]  # "'" is incl. for words like vaa'an


# Finnish consonant clusters (see Karlsson 1985, #4)
CLUSTERS = [
    'bl', 'br', 'dr', 'fl', 'fr', 'gl', 'gr', 'kl', 'kr', 'kv', 'pl', 'pr',
    'cl', 'qv', 'schm']

# Finnish stress (see Anttila 2008)
SON_HIGH = ['i', 'e', 'u', 'y']
SON_LOW = ['a', 'ä', 'o', 'ö']


# UNICODE PLAYGROUND ----------------------------------------------------------

# VOWELS = [unicode(v, 'utf-8') for v in VOWELS]
# DIPHTHONGS = [unicode(d, 'utf-8') for d in DIPHTHONGS]
# CONSONANTS = [unicode(c, 'utf-8') for c in CONSONANTS]
# CLUSTERS = [unicode(c, 'utf-8') for c in CLUSTERS]
# CLUSTER_LENGTHS = set(len(c) for c in CLUSTERS)
# SON_HIGH = [unicode(v, 'utf-8') for v in SON_HIGH]
# SON_LOW = [unicode(v, 'utf-8') for v in SON_LOW]


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


# Phonotactic functions -------------------------------------------------------

def split_syllable(syllable):
    '''Split syllable into a tuple of its parts: onset, nucleus, and coda.'''
    syll = syllable.lower()
    nucleus = ''.join([v for v in syll if v in VOWELS])
    onset, nucleus, coda = syll.partition(nucleus)

    return (onset, nucleus, coda)


def is_inseparable_vowels(chars):
    return is_diphthong(chars) or is_long(chars)


def is_consonantal_onset(chars):
    return is_cluster(chars) or is_consonant(chars)


# Sonority functions ----------------------------------------------------------

# Return the sonority of a syllable
get_syllable_sonority = lambda vowel: vowel[0].upper if vowel else '?'  # HUH?


def get_sonorities(syllables):  # PLUG
    sonorities = []

    for syllable in syllables:
        nucleus = split_syllable(syllable)[1]
        sonority = get_syllable_sonority(nucleus)
        sonorities.append(sonority)


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
    # syllable is light and stressed, makes its stress optional
    if stress[0][-2] == UNSTRESSED:
        stresses = SECONDARY

    elif stresses[0][-2] == SECONDARY and not is_heavy(weights[-2]):
        stresses[1][-1] = SECONDARY
        stresses[1][-2] = UNSTRESSED

    return stress

# -----------------------------------------------------------------------------
