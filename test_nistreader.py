#!/usr/bin/env python
"""
Unit tests for the reader
"""

import unittest
from nistreader import NistReader as NR


class TestNistReader(unittest.TestCase):

    def test_train_header(self):
        nr = NR('train')
        self.assertEqual(nr.items, 60000)
        self.assertEqual(nr.rows, 28)
        self.assertEqual(nr.cols, 28)
        nr.close()

    def test_test_header(self):
        nr = NR('t10k')
        self.assertEqual(nr.items, 10000)
        self.assertEqual(nr.rows, 28)
        self.assertEqual(nr.cols, 28)
        nr.close()

    def test_read_and_skip(self):
        nr = NR('train')
        nr.skip_items(10)
        l1, i1 = nr.read_item()
        nr.close()
        nr2 = NR('train')
        for i in range(0, 10):
            nr2.read_item()
        l2, i2 = nr2.read_item()
        nr2.close()
        self.assertEqual(l1, l2)
        self.assertEqual(i1, i2)

    def test_read_acceptable(self):
        nr = NR('train')
        for i in range(0, 10):
            for c in range(0, 10):
                label, image = nr.read_acceptable([c])
                self.assertEqual(c, label)
        nr.close()

    def test_read_balanced_binary(self):
        nr = NR('train')
        data = nr.read_balanced(100, [0, 1])
        self.assertEqual(len(data[0]), 100)
        self.assertEqual(len(data[1]), 100)
        nr.close()
        zeros, ones = [], []
        nr2 = NR('train')
        while len(zeros) < 100:
            label, image = nr2.read_acceptable([0])
            zeros.append(image)
        nr2.close()
        nr3 = NR('train')
        while len(ones) < 100:
            label, image = nr3.read_acceptable([1])
            ones.append(image)
        nr3.close()
        self.assertEqual(data[0], zeros)
        self.assertEqual(data[1], ones)


if __name__ == '__main__':
    unittest.main()
