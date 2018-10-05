import re
import sys

from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pathlib import PosixPath
from pprint import pprint

try:
    from tqdm import tqdm
    progressbar = tqdm
except:
    progressbar = lambda x: x

TOKEN_SPLITTER = re.compile(' |(\w+)')

DEFAULT_WINDOW_SIZE = 3

import logging
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format='%(asctime)s:%(levelname)s - %(message)s')
logger = logging.getLogger()

def tokenize(line):
    is_group = False
    for token in re.split(TOKEN_SPLITTER, line):
        if token:
            yield (token.strip(), is_group)

        is_group = not is_group


def canonicalize(lexed_line):
    wmap = {}
    wlist = []
    i = 0
    for group, is_word in lexed_line:
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


def sliding_window(array, width):
    if len(array) < width:
        return

    for i in range(len(array) - width + 1):
        yield tuple(array[i:i+width])


class Line():
    def __init__(self, source_file, line_number):
        self.source_file = source_file
        self.line_number = line_number

    @property
    def codebase(self):
        return self.source_file.codebase

    def __str__(self):
        return f'{self.source_file}:{self.line_number}'

    def __repr__(self):
        return f'<Line {self}>'


class SourceFile():
    def __init__(self, codebase, path):
        self.codebase = codebase
        self.path = path
        self.size = 0

    def newline(self):
        nline = Line(self, self.size)
        self.size += 1
        return nline

    def lines(self):
        try:
            with self.path.open() as f:
                for line_content in f.readlines():
                    yield (self.newline(), line_content)
        except (FileNotFoundError, UnicodeDecodeError):
            pass

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f'<SourceFile {self} ({self.codebase})>'


class Codebase():
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def __repr__(self):
        return f'<Codebase {self.name}>'

    def pairmaker(self, o):
        if hash(self) < hash(o):
            return lambda a, b: (a, b)
        return lambda a, b: (b, a)

    def pair(self, o):
        return self.pairmaker(o)(self, o)

    @staticmethod
    def from_path(path):
        return Codebase(path, path.name)

    def find_sources(self, glob):
        for source_path in self.path.glob(glob):
            yield SourceFile(self, source_path)


def preprocess_source(source_file, window_size):
    keylist = []
    for source_line, line_content in source_file.lines():
        # preprocess line content
        raw_key = canonicalize(tokenize(line_content))
        # create and store a metadata container
        keylist.append((raw_key, source_line))

    # keylist now contains a list of preprocessed line, along with
    # where they appeared in the first place

    # yield groups of preprocessed lines, so they can be later looked up
    for line_group in sliding_window(keylist, window_size):
        # the line number of the group is the line number of its first line
        source_line = line_group[0][1]
        yield (tuple(map(lambda x: x[0], line_group)), source_line)


def preprocess_sources(source_files, window_size):
    '''creates line groups'''
    for source_file in progressbar(source_files):
        yield from preprocess_source(source_file, window_size)


def correlate_line_groups(line_groups):
    hitmap = defaultdict(list)

    for line_group, source_line in line_groups:
        hitmap[line_group].append(source_line)

    return list(hitmap.items())


def correlate_sources(source_files, window_size):
    line_groups = preprocess_sources(source_files, window_size)
    return correlate_line_groups(line_groups)


def _trish_parser():
    parser = ArgumentParser(description='Run trish')
    paa = parser.add_argument

    paa('--pattern', action='store', default=None,
        help=("Recursively look for files with names matching"
              "the specified pattern. If no pattern is given, "
              "targets are assumed to be files."))

    paa('-w', '--window_size', action='store', type=int, default=DEFAULT_WINDOW_SIZE,
        help='minimum size for a group of similar lines to be considered')

    # paa('--versus', action='store_true',
    #     help="challenge the first candidate against the others")

    paa('targets', nargs='+', default=[],
        help='Target files or folders to test')

    return parser


def find_sources(targets, pattern):
    targets = map(PosixPath, targets)

    if pattern is None:
        return [SourceFile(None, path) for path in targets]

    return list(chain.from_iterable(Codebase.from_path(path)
                                    .find_sources(pattern)
                                    for path in targets))


def pairs(l):
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            yield (l[i], l[j])

class Match():
    def __init__(self, *items):
        assert len(items) == 2
        self.items = tuple(items)

    @property
    def left(self):
        return self.items[0]

    @property
    def right(self):
        return self.items[1]

    def __getitem__(self, i):
        return self.items[i]

    def __str__(self):
        return f'{self.left} / self.right'

    def __repr__(self):
        return f'<Match {self.left} {self.right}>'


def group_codebase_files(matches):
    # a map from a pair of codebases to a map of pair of files
    # to list of pairs of matches
    codebases_map = defaultdict(lambda: defaultdict(list))
    for keygroup, occurences in progressbar(matches):
        for line_a, line_b in pairs(occurences):
            base_a, base_b = line_a.codebase, line_b.codebase
            if base_a is base_b:
                continue

            pairmaker = base_a.pairmaker(base_b)
            codebase_pair = pairmaker(base_a, base_b)
            line_pair = pairmaker(line_a, line_b)
            file_pair = pairmaker(line_a.source_file, line_b.source_file)
            codebases_map[codebase_pair][file_pair].append(Match(*line_pair))

    return {k: dict(v) for k, v in codebases_map.items()}

class LineRun():
    def __init__(self):
        self.lines = []
        self.neighbors = []
        self.visited = False

class LineCluster():
    def __init__(self):
        self.lines = []
        self.min_line = None
        self.max_line = None

    def __len__(self):
        return len(self.lines)

    def update(self, line):
        self.lines.append(line)
        line_no = line.line_number
        if self.min_line is None or line_no < self.min_line.line_number:
            self.min_line = line
        if self.max_line is None or line_no > self.max_line.line_number:
            self.max_line = line

    def __repr__(self):
        return (f'<LineCluster {len(self)}'
                f' {self.min_line.line_number}'
                f':{self.max_line.line_number}>')


def group_lines(options, codebases_map):
    '''
    {
       (base_a, base_b): {
           (file_a, file_b): [
               (line_a, line_b)
           ]
       }
    }
    '''
    res = {}
    for codebase_pair, file_map in progressbar(codebases_map.items()):
        file_res = {}
        for file_pair, matches in file_map.items():
            left_file, right_file = file_pair
            run_map = {}
            def compute_runs(pair_i):
                # the same item can be involved in more than a single pair
                # the set can be avoided by changing algorithms instead
                match_list = list({match[pair_i] for match in matches})
                # insane stuff can be done here
                # we may sort this O(file_size), which may not be a good idea,
                # given that most of the time, len(matches) <<< file_size.
                # we could store a list of file lines somewhere, and test if the
                # line has matched using a hash set.
                match_list.sort(key=lambda m: m.line_number)

                runs = []

                run = LineRun()
                expected_i = None

                def end_run():
                    nonlocal run
                    if run.lines:
                        runs.append(run)
                        run = LineRun()

                for line in match_list:
                    if expected_i is None or line.line_number != expected_i:
                        end_run()

                    run_map[line] = run
                    run.lines.append(line)

                end_run()
                return match_list, runs

            left_matches, left_runs = compute_runs(0)
            right_matches, right_runs = compute_runs(1)

            for left_line, right_line in matches:
                left_run = run_map[left_line]
                right_run = run_map[right_line]
                left_run.neighbors.append(right_run)
                right_run.neighbors.append(left_run)

            # groups connected clusters, marking these as visited
            def get_cluster(run):
                if run.visited:
                    return

                run.visited = True
                yield from run.lines
                for neighbor in run.neighbors:
                    yield from get_cluster(neighbor)

            clusters = []
            # runs should be connected now, so it doesn't matter which side is
            # iterated over
            for run in left_runs:
                new_cluster = list(get_cluster(run))
                if not new_cluster:
                    continue

                left_cluster = LineCluster()
                right_cluster = LineCluster()
                for line in new_cluster:
                    if line.source_file is left_file:
                        left_cluster.update(line)
                    else:
                        right_cluster.update(line)

                assert len(left_cluster) and len(right_cluster)
                clusters.append((left_cluster, right_cluster))

            file_res[file_pair] = clusters
        res[codebase_pair] = file_res
    return res


def rate_grouped_lines(codebases_map):
    res = {}
    for codebase_pair, file_map in codebases_map.items():
        matches = chain.from_iterable(file_map.values())
        res[codebase_pair] = sum(match[2] ** 2 for match in matches)

    return res

def process_matches(options, matches):
    res = {}

    def store(f, *args, **kwargs):
        f_res = f(*args, **kwargs)
        res[f.__name__] = f_res
        return f_res

    '''
    {
       Canon: [Lines]
    }
    '''
    res['matches'] = matches
    '''
    {
       (base_a, base_b): {
           (file_a, file_b): [
               (line_a, line_b)
           ]
       }
    }
    '''
    logger.info('grouping codebases / files')
    codebase_file_groups = store(group_codebase_files, matches)
    '''
    {
       (base_a, base_b): {
           (file_a, file_b): [
               <LineCluster size begin:end>
           ]
       }
    }
    '''
    logger.info('graph madness')
    lengthful_matches = store(group_lines, options, codebase_file_groups)
    return lengthful_matches, res
    # return store(rate_grouped_lines, lengthful_matches), res


def main(args=sys.argv[1:]):
    options = _trish_parser().parse_args(args=args)
    source_files = find_sources(options.targets, options.pattern)
    window_size = options.window_size
    logger.info('correlating sources')
    matches = correlate_sources(source_files, window_size)
    scores, metadata = process_matches(options, matches)
    # pprint(metadata)
    # for codebase_pair, score in scores.items():
    #     codebase_a, codebase_b = codebase_pair
    #     print(f'{score}\t{codebase_a.name}\t{codebase_b.name}')

if __name__ == '__main__':
    main(sys.argv[1:])
