from collections import namedtuple, defaultdict
from .canon import canonicalize

LinesMeta = namedtuple('LinesMeta', ['canon_occ', 'lines'])

def compute_meta(conf, lines):
    canon_map = defaultdict(set)
    canon_lines = []
    canon_i = 0
    canon_stream = conf.canonicalize(conf.group_tokens(conf.get_tokens(lines)))
    for i, canon_line in enumerate(canon_stream):
        if canon_line:
            canon_map[canon_line].add(canon_i)
            canon_lines.append(canon_line)
            canon_i += 1
    return LinesMeta(canon_map, canon_lines)
