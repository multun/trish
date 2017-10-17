import os
from itertools import chain
from libtrish.conf import import_conf, DefaultTrishConf


def _getlines(filename):
    with open(filename, 'r') as fp:
        return fp.readlines()


def compare(conf, file_a, file_b):
    chal_fnames = (file_a, file_b)
    chal_lines = map(_getlines, chal_fnames)
    chal_metas = map(conf.compute_meta, chal_lines)
    chal = conf.Challenge(*chain(*zip(chal_metas, chal_fnames)))
    chal_res = conf.evaluate_chal(chal)
    conf.exporter(chal, chal_res)


def load_conf(spec=None):
    conf_spec = spec or os.environ.get('TRISH_CONF')
    return import_conf(conf_spec) if conf_spec else DefaultTrishConf
