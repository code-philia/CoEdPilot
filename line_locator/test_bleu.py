import unittest
from bleu import normalize, count_ngrams, cook_refs, cook_test, bleuFromMaps, bleu


class TestBLEUFunctions(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize('This, is a test!'), [
                         'this', ',', 'is', 'a', 'test', '!'])
        self.assertEqual(normalize('Hello\nWorld'), ['hello', 'world'])
        self.assertEqual(normalize('No-Extra    Spaces'),
                         ['no-extra', 'spaces'])

    def test_count_ngrams(self):
        words = ['this', 'is', 'a', 'a', 'test']
        counts = count_ngrams(words, n=2)
        self.assertEqual(
            counts,
            {
                ('this',): 1,
                ('is',): 1,
                ('a',): 2,
                ('test',): 1,
                ('this', 'is'): 1,
                ('is', 'a'): 1,
                ('a', 'a'): 1,
                ('a', 'test'): 1,
            },
        )

    def test_cook_refs(self):
        refs = ['this is a test', 'this is another test']
        cooked_refs = cook_refs(refs)
        self.assertIn(('this', 'is'), cooked_refs[1])
        # Shortest reference length is 4.
        self.assertEqual(min(cooked_refs[0]), 4)

    def test_cook_test(self):
        refs = cook_refs(['this is a test'])
        test = cook_test('this is a great test', refs)
        self.assertEqual(test['testlen'], 5)  # Candidate length
        self.assertEqual(test['correct'], [4, 2, 1, 0])  # n-gram matches

    def test_bleu(self):
        refs = ['this is a test']
        candidate = 'this is a test'
        score = bleu(refs, candidate)
        self.assertAlmostEqual(score[0], 1.0)  # Perfect match

        candidate = 'completely unrelated sentence'
        score = bleu(refs, candidate)
        self.assertAlmostEqual(score[0], 0.0)  # No match

    def test_bleuFromMaps(self):
        ref_map = {'1': ['this is a test']}
        pred_map = {'1': ['this is a test']}
        score = bleuFromMaps(ref_map, pred_map)
        self.assertAlmostEqual(score[0], 100.0)  # Perfect match in percentage


if __name__ == '__main__':
    unittest.main()
