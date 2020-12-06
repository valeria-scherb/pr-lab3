#!/usr/bin/env python
"""
Unit tests for the EM algorithm
Depends on correctness of NIST reader
"""

import unittest, random
from nistreader import NistReader as NR
from expmax import ExpMax as EM


class TestExpMax(unittest.TestCase):

    def test_clustering(self):
        nr, nrt = NR('train'), NR('t10k')
        cl_data = nr.read_balanced(100, [0, 1])
        ts_data = nrt.read_balanced(100, [0, 1])
        nr.close()
        nrt.close()
        random.seed(12345)
        em = EM(nr.im_size)
        data = cl_data[0] + cl_data[1]
        random.shuffle(data)
        em.load_train(data)
        em.run(10)
        # Test classification on training data
        cl_res_0 = [em.classify(x) for x in cl_data[0]]
        cl_res_1 = [em.classify(x) for x in cl_data[1]]
        av_cr_0 = sum(cl_res_0) / len(cl_res_0)
        av_cr_1 = sum(cl_res_1) / len(cl_res_1)
        # Must not be too close
        self.assertGreater(abs(av_cr_0 - av_cr_1), 0.5)
        # Must be closer to 0 or 1
        self.assertGreater(abs(av_cr_0 - 0.5), 0.2)
        self.assertGreater(abs(av_cr_1 - 0.5), 0.2)
        # Test classification on testing data
        cl_res_0 = [em.classify(x) for x in ts_data[0]]
        cl_res_1 = [em.classify(x) for x in ts_data[1]]
        av_cr_0 = sum(cl_res_0) / len(cl_res_0)
        av_cr_1 = sum(cl_res_1) / len(cl_res_1)
        # Must not be too close
        self.assertGreater(abs(av_cr_0 - av_cr_1), 0.5)
        # Must be closer to 0 or 1 (data is worse)
        self.assertGreater(abs(av_cr_0 - 0.5), 0.1)
        self.assertGreater(abs(av_cr_1 - 0.5), 0.1)


if __name__ == '__main__':
    unittest.main()
