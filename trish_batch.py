#!/usr/bin/env python3

import sys
import os
from trish import *

def disp(res, nb=10):
    res = sorted(res, reverse=True)
    for i in range(min(nb, len(res))):
        print("Rank:", i + 1, "Score:", res[i][0], "Files:", res[i][1])


def check(directory):
    res = []
    filenames = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as f:
            f_lines = f.readlines()
        filenames.append((filename, get_meta(f_lines)))
    for i in range(len(filenames) - 1):
        for j in range(i + 1, len(filenames)):
            res.append((compare_meta(filenames[i][1], filenames[j][1]), \
                        filenames[i][0] + " -> " + filenames[j][0]))
    return res


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} folder [nbdisp]")
        exit(1)
    nb = 10 if len(sys.argv) == 2 else int(sys.argv[2])
    res = check(sys.argv[1])
    disp(res, nb)
