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
print('Zeros:', [em.class_info(x) for x in cldata[0]])
print('Ones: ', [em.class_info(x) for x in cldata[1]])

em.run()

print('mu2', em.mu)
print('pi2', em.pi)
print('z2', em.z)
print('Zeros:', [em.class_info(x) for x in cldata[0]])
print('Ones: ', [em.class_info(x) for x in cldata[1]])

em.run()

print('mu3', em.mu)
print('pi3', em.pi)
print('z3', em.z)
print('Zeros:', [em.class_info(x) for x in cldata[0]])
print('Ones: ', [em.class_info(x) for x in cldata[1]])

print('')
print('Try classifying')
print('Zeros:', [em.classify(x) for x in cldata[0]])
print('Ones: ', [em.classify(x) for x in cldata[1]])
