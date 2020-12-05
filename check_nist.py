#!/usr/bin/env python
"""
Check out NIST reader to see some images
"""

from nistreader import NistReader as NR

nr = NR('train')
items = nr.read_balanced(5)
print('Zeros (0)')
for i in range(0, 5):
    for x in nr.make_matrix(items[0][i]):
        print(' '.join(['#' if y else ' ' for y in x]))
print('')
print('Ones (1)')
for i in range(0, 5):
    for x in nr.make_matrix(items[1][i]):
        print(' '.join(['#' if y else ' ' for y in x]))
