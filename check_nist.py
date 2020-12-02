#!/usr/bin/env python
"""
Check out NIST reader to see some images
"""

from nistreader import NistReader as NR

nr = NR('train')
for i in range(0, 9):
    label, image = nr.read_item()
    print('Label:', label)
    for x in image:
        print(' '.join(['#' if y else ' ' for y in x]))