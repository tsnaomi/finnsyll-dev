# coding=utf-8

from Levenshtein import distance
from os import sys, path
from phonology import replace_umlauts

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


# Clouet2014 ------------------------------------------------------------------

class ClouetDaille2014(object):

    def __init__(self,):
        import finnsyll as finn
        self.tokens = finn.training_set()

    def split(self, word):
        try:
            # candidates = [[word, ], ]
            candidates = []

            # split a candidate compound by generating all its possible
            # two-part segmentations beginning with the components of minimum
            # permitted length 3 characters, which is a frequent choice for
            # compound splitting systems

            iterator = ([word[:i], word[i:]] for i in range(3, len(word) - 2))

            for segment in iterator:

                # TODO: apply rules to restore independent lexemes from
                # non-independent components (e.g., "ed" > "e", as in
                # "health-based" > "health base"); if rules are not available
                # or sufficient, propose potential lemmas using normalized
                # Levenshtein distance

                candidates.append(list(segment))

                # split the RIGHT SIDE component further in a recursive manner,
                # up until a parameterized level corresponding to the maximum
                # expected number of components
                # c = segment.pop()

                # if len(c) > 2:
                #     for e in ([c[:i], c[i:]] for i in range(3, len(c) - 2)):
                #         candidates.append(list(segment) + e)

            try:
                scored = [(cand, self.score(cand)) for cand in candidates]
                best_candidate, SCORE = max(scored, key=lambda c: c[1])

                print best_candidate, SCORE

                if SCORE > 0.80:
                    return '='.join(best_candidate)

            except ValueError:
                pass

            return word

        except IndexError:
            # the word is too short to be a compound
            return word

    def score(self, candidate):
        SCORE = 1

        if len(candidate) == 1:
            return self.interpolate(candidate[0])

        elif len(candidate) == 3:
            SCORE += self.interpolate(candidate[0])
            SCORE += self.score(candidate[1:])

            if all([True for c in candidate if c in self.database]):
                SCORE /= 2.0

            else:
                SCORE /= 3.0

        else:
            SCORE += self.interpolate(candidate[0])
            SCORE += self.interpolate(candidate[1])
            SCORE /= 2.0

        return SCORE

    def interpolate(self, comp):
        sim = max(distance(comp, str(lemma)) for lemma in self.lemmas)
        inCorpus = 1 if comp in self.corpus.keys() else 0  # same as inDico
        DSpec = self.corpus.get(comp, 0)

        # Score(comp) = α sim(comp, lemma) + β inDico + γ inCorpus + δ DSpec
        return (0.25 * sim) + (0.5 * inCorpus) + (0.25 * DSpec)


# Morfessor -------------------------------------------------------------------

# In morphological segmentation, compounds are word forms, constructions are
# morphs, and atoms are characters. In chunking, compounds are sentences,
# constructions are phrases, and atoms are words.


# -----------------------------------------------------------------------------

def delimit(word):
    '''Insert syllable breaks at non-delimited compound boundaries.'''
    return word


if __name__ == '__main__':

    # WRITE TESTS

    words = [
        'pian',             # pian
        'talous',           # talous
        'jääkiekkoilu',     # jää=kiekkoilu
        'erinomaisesti',    # erin=omaisesti
        'asianajaja',       # asian=ajaja
        'rahoituserien',    # rahoitus=erien
        'vastikAAn',        # vast=ikan
        'kansanäänestys',   # kansan=äänestys
        ]

    for word in words:
        print replace_umlauts(delimit(replace_umlauts(word)), put_back=True)
