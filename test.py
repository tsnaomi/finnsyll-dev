# coding=utf-8

import syllabifier.phonology as phon
import unittest

from gutenberg import u_y_final_diphthongs


class TestPhonotactics(unittest.TestCase):

    def test_min_word(self):
        pairs = [
            # violations
            ('a', False),               # V
            ('n', False),               # C
            ('kA', False),              # CV
            ('en', False),              # VC
            ('jat', False),             # CVC
            ('kk', False),              # CC

            # not violations
            ('AA', True),               # VV
            ('vAi', True),              # CVV
            ('ien', True),              # VVC
            ('ita', True),              # VCV
        ]

        for t, e in pairs:
            self.assertEqual(phon.min_word(t, None), e)

    def test_not_VVC(self):
        pairs = [
            # violations
            ('ien', False),             # VVC
            ('aak', False),

            # not violations
            ('jAA', True),              # CVV
            ('kaa', True),
            ('kak', True),              # CVC
            ('pAin', True),             # CVVC
            ('aaak', True),             # VVVC
            ('akk', True),              # VVC
        ]

        for t, e in pairs:
            self.assertEqual(phon.not_VVC(t, None), e)

    def test_word_final(self):
        pairs = [
            # violations
            ('sulok', False),           # C[-coronal]#
            ('hyp', False),
            ('pitem', False),
            ('heng', False),
            ('k', False),
            ('hoid', False),            # /d/-final
            ('d', False),
            ('af', False),              # foreign-final
            ('berg', False),

            # not violations
            ('sairaan', True),          # C[+coronal]#
            ('oikeus', True),
            ('jat', True),              #
            ('n', True),                #
            ('jAA', True),              # V-final
            ('bott', True),             # CC[+coronal]#
        ]

        for t, e in pairs:
            self.assertEqual(phon.word_final(t, None), e)

    def test_sonseq(self):
        pairs = [
            # violations
            ('dipl', False),            # bad sonority slopes
            ('ntupien', False),
            ('mpaa', False),
            ('nnin', False),
            ('nn', False),
            ('tsheenien', False),
            ('tlAA', False),

            # not violations
            ('psykologi', True),        # borrowed word-initial CC
            ('tsaari', True),
            ('snobi', True),
            ('draken', True),
            ('draama', True),
            ('primakovin', True),       # Finnish word-initial CC
            ('prosentti', True),
            ('stolaisilla', True),
            ('kritiikki', True),
            ('plAA', True),
            ('klAA', True),
            ('trAA', True),
            ('spAA', True),
            ('skAA', True),
            ('stressaavalle', True),    # Finnish word-initial CCC
            ('strategiansa', True),
            ('spriille', True),
            ('a', True),                # V
            ('n', True),                # C
            ('kas', True),              # CVC

            # uncertain
            ('niks', False),            # Finnish word-final CC
            ('naks', False),
            ('kops', False),
            ('raps', False),
            ('ritts', False),
            ('britannin', False),       # foreign CC
            ('friikeille', False),
            ('berg', False),
        ]

        for t, e in pairs:
            self.assertEqual(phon.sonseq(t, False), e)

    def test_harmonic(self):
        pairs = [
            # violations
            ('kesAillan', False),       # closed compounds
            ('taaksepAin', False),
            ('muutostOitA', False),

            # not violations
            ('kesA', True),             # Finnish stems
            ('illan', True),
            ('taakse', True),
            ('pAin', True),
            ('muutos', True),
            ('tOitA', True),

            # uncertain
            ('kallstrOm', False),           # loanwords
            ('donaueschingenissA', False),
            ('finlaysonin', False),
            ('affArer]', False),
            ]

        for t, e in pairs:
            self.assertEqual(phon.harmonic(t, False), e)


class TestDiphthongSearch(unittest.TestCase):

    def setUp(self):

        def find_diphthongs(word):
            matches = u_y_final_diphthongs(word)

            try:
                if len(matches) == 1:
                    return matches[0].groups()

                return reduce(lambda x, y: x.groups() + y.groups(), matches)

            except TypeError:
                return ()

        self.find_diphthongs = find_diphthongs

    def test_extract_standalone_diphthongs(self):
        diphthongs = ['au', 'eu', 'ou', 'iu', 'iy', 'ey', 'äy', 'öy']

        for vv in diphthongs:
            self.assertEqual(self.find_diphthongs(vv), (vv, ))
            self.assertEqual(self.find_diphthongs('h%skutellen' % vv), (vv, ))
            self.assertEqual(self.find_diphthongs('hämärt%sk' % vv), (vv, ))

            # word-initial extraction
            self.assertEqual(self.find_diphthongs('%srinkoiset' % vv), (vv, ))

            # word-final extraction
            self.assertEqual(self.find_diphthongs('k%s' % vv), (vv, ))

    def test_extract_multiple_diphthongs(self):
        pairs = [
            ('houkoutellen', ('ou', 'ou')),
            ('houkkautellen', ('ou', 'au')),
            ('oukoutellen', ('ou', 'ou')),
            ('houkkau', ('ou', 'au')),
            ]

        for t, e in pairs:
            self.assertEqual(self.find_diphthongs(t), e)

    def test_ignore_longer_vowel_sequences(self):
        test = ['hämärtäyä', 'hämärtääy', 'hämärtäyäk', 'hämärtääyk']

        for t in test:
            self.assertEqual(self.find_diphthongs(t), ())


if __name__ == '__main__':
    unittest.main()
