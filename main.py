#!/usr/bin/env python
"""
Main program
"""

from nistreader import NistReader as NR
from expmax import ExpMax as EM
import random

random_seed = 12345
train_samples = 100
test_samples = 500
test_samples_t10k = 500
iterations = 10

# -----------------------------------------------------------------------------

random.seed(12345)
nr = NR('train')
nrt = NR('t10k')

train = nr.read_balanced(train_samples/2, [0, 1])
test1 = nr.read_balanced(test_samples/2, [0, 1])
test2 = nr.read_balanced(test_samples_t10k/2, [0, 1])

data = train[0] + train[1]
random.shuffle(data)

em = EM(nr.im_size, 2)
em.load_train(data)
em.run(iterations)

cl_res_0 = [em.classify(x) for x in train[0]]
cl_res_1 = [em.classify(x) for x in train[1]]
av_cr_0 = sum(cl_res_0) / len(cl_res_0)
av_cr_1 = sum(cl_res_1) / len(cl_res_1)

print('[Training] Average class for zeros:', av_cr_0)
print('[Training] Average class for ones: ', av_cr_1)

