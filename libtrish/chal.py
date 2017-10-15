from itertools import count


class Challenge():
    def __init__(self, a, adesc, b, bdesc):
        self.a = a
        self.b = b
        self.adesc = adesc
        self.bdesc = bdesc

    def __str__(self):
        return f'{self.adesc}\t{self.bdesc}'


def evaluate_match(conf, a, line_a, b, line_b):
    for i in count():
        if line_a + i >= len(a.lines) \
           or line_b + i >= len(b.lines) \
           or a.lines[line_a + i] != b.lines[line_b + i]:
            return i


def get_pairs(a, b):
    for canon, occ_set_a in a.canon_occ.items():
        occ_set_b = b.canon_occ.get(canon)
        if occ_set_b is None:
            continue
        for occa in occ_set_a:
            for occb in occ_set_b:
                yield (occa, occb)


def evaluate_chal(conf, chal):
    stats = conf.ChalStats(conf)
    for occa, occb in get_pairs(chal.a, chal.b):
        eval_res = conf.evaluate_match(chal.a, occa + 1, chal.b, occb + 1)
        if eval_res:
            stats.register(eval_res)
    stats.finalize(conf)
    return stats
