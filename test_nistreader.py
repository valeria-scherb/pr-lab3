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


if __name__ == '__main__':
    unittest.main()
