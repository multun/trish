import re
import sys

from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pathlib import PosixPath
from pprint import pprint

TOKEN_SPLITTER = re.compile(' |(\w+)')

DEFAULT_WINDOW_SIZE = 3

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
    for source_file in source_files:
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


def group_codebase_files(matches):
    # a map from a pair of codebases to a map of pair of files
    # to list of pairs of matches
    codebases_map = defaultdict(lambda: defaultdict(list))
    for keygroup, occurences in matches:
        for line_a, line_b in pairs(occurences):
            base_a, base_b = line_a.codebase, line_b.codebase
            if base_a is base_b:
                continue

            pairmaker = base_a.pairmaker(base_b)
            codebase_pair = pairmaker(base_a, base_b)
            line_pair = pairmaker(line_a.line_number, line_b.line_number)
            file_pair = pairmaker(line_a.source_file, line_b.source_file)
            codebases_map[codebase_pair][file_pair].append(line_pair)

    return {k: dict(v) for k, v in codebases_map.items()}


def group_lines(options, codebases_map):
    res = {}
    for codebase_pair, file_map in codebases_map.items():
        file_res = {}
        for file_pair, matches in file_map.items():
            # group matches between files by geometric angle
            angle_map = defaultdict(list)
            def line_pair_angle(line_pair):
                return line_pair[1] - line_pair[0]

            for line_pair in matches:
                angle_map[line_pair_angle(line_pair)].append(line_pair)

            ranges = []
            for angle, matches in angle_map.items():
                matches.sort()

                range_start = None
                expected_i = None

                def end_range():
                    if range_start is not None:
                        range_end = expected_i - 1
                        range_len = range_end - range_start
                        ranges.append((range_start,
                                       range_start + angle,
                                       range_len + options.window_size))

                for line_pair in matches:
                    line_a, line_b = line_pair
                    if line_a != expected_i:
                        end_range()
                        range_start = line_a
                    expected_i = line_a + 1

                end_range()

            file_res[file_pair] = ranges
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

    res['matches'] = matches
    codebase_file_groups = store(group_codebase_files, matches)
    lengthful_matches = store(group_lines, options, codebase_file_groups)
    return store(rate_grouped_lines, lengthful_matches), res


def main(args=sys.argv[1:]):
    options = _trish_parser().parse_args(args=args)
    source_files = find_sources(options.targets, options.pattern)
    window_size = options.window_size
    matches = correlate_sources(source_files, window_size)
    scores, metadata = process_matches(options, matches)
    pprint(metadata)
    # for codebase_pair, score in scores.items():
    #     codebase_a, codebase_b = codebase_pair
    #     print(f'{score}\t{codebase_a.name}\t{codebase_b.name}')

if __name__ == '__main__':
    main(sys.argv[1:])
