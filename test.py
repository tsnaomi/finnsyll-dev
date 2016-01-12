# coding=utf-8

import syllabifier.phonology as phon
import unittest

from gutenberg import u_y_final_diphthongs


class TestPhonotactics(unittest.TestCase):

    def test_good_nuclei(self):
        pairs = [
            ('b', False),
            ('ba', False),
            ('baa', True),
            ('baaa', True),
            ]

        for t, e in pairs:
            self.assertEqual(phon.check_nuclei(t), e)

    def test_good_word_final(self):
        pairs = [
            ('ba', True),
            ('bi', True),
            ('bat', True),
            ('bak', False),
            ('bam', False),
            ]

        for t, e in pairs:
            self.assertEqual(phon.check_word_final(t), e)

    def test_is_harmonic(self):
        pairs = [
            ('AyO', True),
            ('auo', True),
            ('A', True),
            ('a', True),
            ('i', True),
            ('e', True),
            ('AyOie', True),
            ('auoie', True),
            ('AyOauo', False),
            ('AyOauoie', False),
            ]

        for t, e in pairs:
            self.assertEqual(phon.is_harmonic(t), e)

    def test_good_sonseq(self):
        pairs = [
            ('b', True),
            ('a', True),
            ('at', True),
            ('ta', True),
            ('stra', True),
            ('blak', True),
            ('belg', True),
            ('bb', False),
            ('ntan', False),
            ('kabl', False),
            ]

        for t, e in pairs:
            self.assertEqual(phon.check_sonseq(t), e)


class TestStrictDiphthongSearch(unittest.TestCase):

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
