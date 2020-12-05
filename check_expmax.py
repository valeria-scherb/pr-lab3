#!/usr/bin/env python
"""
Check out and look at ExpMax class
"""

from nistreader import NistReader as NR
from expmax import ExpMax as EM
import random

random.seed(12345)
nr = NR('train')
cldata = nr.read_balanced(10, [0, 1])
data = cldata[0] + cldata[1]
random.shuffle(data)

random.seed(54321)
em = EM(nr.im_size)
print('K', em.K)
print('D', em.D)
print('mu', em.mu)
print('pi', em.pi)

em.load_train(data)
em.run()

print('mu1', em.mu)
print('pi1', em.pi)
print('z1', em.z)

em.run()

print('mu2', em.mu)
print('pi2', em.pi)
print('z2', em.z)
