from .chal import evaluate_chal, Challenge
from itertools import chain

def _getlines(filename):
    with open(filename, 'r') as fp:
        return fp.readlines()

def compare(conf, file_a, file_b):
    chal_fnames = (file_a, file_b)
    chal_lines = map(_getlines, chal_fnames)
    chal_metas = map(conf.compute_meta, chal_lines)
    chal = Challenge(*chain(*zip(chal_metas, chal_fnames)))
    chal_res = conf.evaluate_chal(chal)
    conf.exporter(chal, chal_res)
