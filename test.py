# coding=utf-8

import unittest

from gutenberg import _get_u_y_final_diphthongs, _get_html, _get_phonotactics


class TestGuntenbergExtraction(unittest.TestCase):

    def setUp(self):

        def find_diphthongs(word):
            return tuple(m.group(2) for m in _get_u_y_final_diphthongs(word))

        self.find_diphthongs = find_diphthongs

    def test_standalone_diphthongs(self):
        '''Extract diphthongs that do not appear within VVV+ sequences.'''
        diphthongs = ['au', 'eu', 'ou', 'iu', 'iy', 'ey', 'äy', 'öy']

        for vv in diphthongs:
            # extract entire string
            self.assertEqual(self.find_diphthongs(vv), (vv, ))

            # word-initial syllable with onset extraction
            self.assertEqual(self.find_diphthongs('C%sCACAC' % vv), (vv, ))

            # word-final syllable with coda extraction
            self.assertEqual(self.find_diphthongs('CACAC%sC' % vv), (vv, ))

            # word-initial extraction
            self.assertEqual(self.find_diphthongs('%sCACACA' % vv), (vv, ))

            # word-final extraction
            self.assertEqual(self.find_diphthongs('ACACAC%s' % vv), (vv, ))

    def test_diphthongs_in_triphthongs(self):
        '''Avoid diphthongs that appear in VVV+ sequences.'''
        test = ['CACACäyä', 'CACACääy', 'CACACäyäC', 'CACACääyC']

        for t in test:
            self.assertEqual(self.find_diphthongs(t), ())

    def test_multiple_diphthongs(self):
        '''Extract multiple diphthongs from a single word.'''
        pairs = [
            ('CouCouCA', ('ou', 'ou')),
            ('CouCCauCA', ('ou', 'au')),
            ('ouCouCACA', ('ou', 'ou')),
            ('CouCCau', ('ou', 'au')),
            ]

        for t, e in pairs:
            self.assertEqual(self.find_diphthongs(t), e)

    def test_html(self):
        '''Test html representations of sequences.'''
        # without umlauts
        word = 'CauCauCA'
        sequences = _get_u_y_final_diphthongs(word)
        expected = ['C<strong>au</strong>CauCA', 'CauC<strong>au</strong>CA']

        for seq, html in zip(sequences, expected):
            self.assertEqual(_get_html(seq, word), html)

        # with umlauts
        word = 'CäyCäyCA'.decode('utf-8')
        sequences = _get_u_y_final_diphthongs(word)
        expected = ['C<strong>äy</strong>CäyCA', 'CäyC<strong>äy</strong>CA']

        for seq, html in zip(sequences, expected):
            self.assertEqual(_get_html(seq, word), html)

    def test_phonotactics(self):
        '''Test the detection of weight and primary stress.'''
        # without umlauts
        word = 'CauCauCCau'
        sequences = _get_u_y_final_diphthongs(word)
        expected = [
            # is_heavy, is_stressed, split
            (False, True, 'join'),
            (True, False, None),
            (False, False, None),
            ]

        for seq, correct in zip(sequences, expected):
            i = seq.start(2)
            self.assertEqual(_get_phonotactics(seq, i, word), correct)

        # with umlauts
        word = 'CäyCäyCCäuC'.decode('utf-8')
        sequences = _get_u_y_final_diphthongs(word)
        expected = [
            # is_heavy, is_stressed, split
            (False, True, 'join'),
            (True, False, None),
            (True, False, None),
            ]

        for seq, correct in zip(sequences, expected):
            i = seq.start(2)
            self.assertEqual(_get_phonotactics(seq, i, word), correct)

if __name__ == '__main__':
    unittest.main()
