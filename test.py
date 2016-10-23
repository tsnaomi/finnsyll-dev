# coding=utf-8

import unittest

from gutenberg import u_y_final_diphthongs


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
