import re
import sys

from collections import defaultdict, namedtuple
from itertools import count


FileMeta = namedtuple('FileMeta', ['canon_map', 'canon_lines'])


split_re = re.compile(' |(\w+)')


def get_groups(line):
    it = iter(re.split(split_re, line))
    next(it)
    while True:
        try:
            gwrd = next(it)
            ggrp = next(it)
        except StopIteration:
            break
        if gwrd:
            yield (gwrd, True)
        elif ggrp:
            yield (ggrp, False)


def get_canon(line):
    wmap = {}
    wlist = []
    i = 0
    for group, is_word in get_groups(line):
        if not is_word:
            wlist.append(group)
            continue

        gstatus = wmap.get(group)
        if gstatus is not None:
            wlist.append(gstatus)
        else:
            wmap[group] = i
            wlist.append(i)
            i += 1
    return tuple(wlist)


def get_meta(lines):
    canon_map = defaultdict(set)
    canon_lines = []
    canon_i = 0
    for i, line in enumerate(lines):
        canon_line = get_canon(line)
        if canon_line:
            canon_map[canon_line].add(canon_i)
            canon_lines.append(canon_line)
            canon_i += 1
    return FileMeta(canon_map, canon_lines)


def get_count(linea, lineb, a, b):
    for i in count():
        if linea + i >= len(a.canon_lines) \
           or lineb + i >= len(b.canon_lines) \
           or a.canon_lines[linea + i] != b.canon_lines[lineb + i]:
            return i


def compare_meta(a, b):
    score = 0
    for canon, occ_set_a in a.canon_map.items():
        occ_set_b = b.canon_map.get(canon)
        if occ_set_b is None:
            continue
        for occa in occ_set_a:
            for occb in occ_set_b:
                lscore = get_count(occa, occb, a, b)
                if lscore > 2:
                    score += lscore
    return score


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} fileA fileB")
        exit(1)

    with open(sys.argv[1], 'r') as fileA:
        lineA_lines = fileA.readlines()

    with open(sys.argv[2], 'r') as fileB:
        lineB_lines = fileB.readlines()

    metas = list(map(get_meta, (lineA_lines, lineB_lines)))
    ca = compare_meta(*metas)
    metas.reverse()
    cb = compare_meta(*metas)
    return ca + cb
