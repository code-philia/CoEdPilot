import unittest
from locator_metric import calc_em_acc, calc_precision_recall_f1


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Simulate predictions and ground truth
        self.predictions = ['1\tkeep', '2\tadd',
                            '3\treplace', '4\tkeep', '5\tadd']
        self.ground_truth = ['1\tkeep', '2\tadd',
                             '3\treplace', '4\tkeep', '5\tkeep']

    def test_calc_em_acc(self):
        em, acc = calc_em_acc(self.predictions, self.ground_truth)
        # Expected EM and Accuracy values
        expected_em = 0.8  # 4/5
        # 8/10 (since "add" vs "keep" in last example counts as mismatch)
        expected_acc = 0.8

        self.assertAlmostEqual(em, expected_em, places=2)
        self.assertAlmostEqual(acc, expected_acc, places=2)

    def test_calc_precision_recall_f1(self):
        precision, recall, f1 = calc_precision_recall_f1(
            self.predictions, self.ground_truth)
        # Expected Precision, Recall, F1 values
        expected_precision = 5 / 6
        expected_recall = 8 / 9
        expected_f1 = 37 / 45

        self.assertAlmostEqual(precision, expected_precision, places=2)
        self.assertAlmostEqual(recall, expected_recall, places=2)
        self.assertAlmostEqual(f1, expected_f1, places=2)


if __name__ == '__main__':
    unittest.main()
