#!/usr/bin/env python3

"""
Copyright (c) 2019 Victor Collod <victor.collod@epita.fr>

Trish is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

Trish is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with trish.  If not, see <http://www.gnu.org/licenses/>.
"""

import json
import logging
import re
import sys

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from fnmatch import fnmatch
from functools import partial
from itertools import chain
from pathlib import Path, PosixPath
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

T = TypeVar("T")

try:
    from tqdm import tqdm  # type: ignore

    progressbar = tqdm  # pylint: disable=invalid-name
except ImportError:

    def progressbar(iterator: T) -> T:
        """a progressbar placeholder, doing nothing with the iterator it's given"""
        return iterator


DEFAULT_WINDOW_SIZE = 3


def _trish_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run trish")

    parser.add_argument(
        "--pattern",
        action="store",
        default=None,
        help=(
            "Recursively look for files with names matching"
            "the specified pattern. If no pattern is given, "
            "targets are assumed to be files."
        ),
    )

    parser.add_argument(
        "--ignore", action="append", default=[], help="Pattern of files to ignore"
    )

    parser.add_argument(
        "-c",
        "--clusters-log-dir",
        action="store",
        default=None,
        help="Log clusters of similar lines to the given directory.",
    )

    parser.add_argument(
        "-w",
        "--window_size",
        action="store",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="minimum size for a group of similar lines to be considered",
    )

    parser.add_argument(
        "--unordered-line-group",
        action="store_true",
        default=False,
        help="Inore the order of lines inside windows",
    )

    parser.add_argument(
        "targets", nargs="+", default=[], help="Target files or folders to test"
    )

    return parser


TOKEN_SPLITTER = re.compile(r" |(\w+)")

logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s - %(message)s",
)

logger = logging.getLogger()  # pylint: disable=invalid-name


def tokenize(line: str) -> Generator[Tuple[str, bool], None, None]:
    """Splits a string on word and spaces boundaries, (token, is_word) pairs

    >>> list(tokenize("a test () sample!"))
    [('a', True), ('test', True), ('()', False), ('sample', True), ('!', False)]
    """
    is_group = False
    for token in re.split(TOKEN_SPLITTER, line):
        if token is not None:
            token = token.strip()

        if token:
            yield (token, is_group)

        is_group = not is_group


CanonicalLineItem = Union[str, int]
CanonicalLine = Tuple[CanonicalLineItem, ...]


def canonicalize(lexed_line: Iterable[Tuple[str, bool]]) -> CanonicalLine:
    """Takes a lexed line and returns a canonical tuple, where each word is
    replaced by the index of its first occurence.

    >>> canonicalize(tokenize("a test () sample!"))
    (0, 1, '()', 2, '!')
    """
    wmap: Dict[str, int] = {}
    wlist: List[CanonicalLineItem] = []
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


def sliding_window(
    array: Sequence[T], width: int
) -> Generator[Tuple[T, ...], None, None]:
    """
    >>> list(sliding_window([1, 2, 3, 4], 2))
    [(1, 2), (2, 3), (3, 4)]
    """
    if len(array) < width:
        return

    for i in range(len(array) - width + 1):
        yield tuple(array[i : i + width])


class Line:
    """A line of code. It has a source file, and a line number"""

    __slots__ = ("source_file", "line_number")

    def __init__(self, source_file: "SourceFile", line_number: int):
        self.source_file = source_file
        self.line_number = line_number

    @property
    def number_format(self) -> str:
        """Formats the line number, highlighting the line
        type (it comes from an actual file).
        """
        return f"r{self.line_number}"

    @property
    def codebase(self) -> "Codebase":
        return self.source_file.codebase

    def __str__(self) -> str:
        return f"{self.source_file}:{self.line_number}"

    def __repr__(self) -> str:
        return f"<Line {self}>"


class SourceFile:
    """A source file. Lines refer to it, it belongs to a codebase,
    and has a size, in lines.
    """

    __slots__ = ("codebase", "path", "size")

    def __init__(self, codebase: Optional["Codebase"], path: Path):
        self.path = path
        self.size = 0

        if codebase is None:
            codebase = Codebase(path, path.name)

        self.codebase = codebase

    def newline(self) -> Line:
        nline = Line(self, self.size)
        self.size += 1
        self.codebase.line_count += 1
        return nline

    def lines(self) -> Generator[Tuple[Line, str], None, None]:
        try:
            with self.path.open() as source_file:
                for line_content in source_file.readlines():
                    yield (self.newline(), line_content)
        except (FileNotFoundError, UnicodeDecodeError):
            pass

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"<SourceFile {self}>"


class Codebase:
    """A codebase is a set of source files.
    It has a size in lines, a path and a name.
    """

    __slots__ = ("path", "name", "line_count")

    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        self.line_count = 0

    def get_size(self) -> int:
        return self.line_count

    def __repr__(self) -> str:
        return f"<Codebase {self.name}>"

    def pairmaker(self, other: "Codebase") -> Callable[[T, T], Tuple[T, T]]:
        if id(self) < id(other):
            return lambda a, b: (a, b)
        return lambda a, b: (b, a)

    @staticmethod
    def from_path(path: Path) -> "Codebase":
        return Codebase(path, path.name)

    def find_sources(
        self, ignore_list: List[str], glob: str
    ) -> Generator[SourceFile, None, None]:
        for source_path in self.path.glob(glob):
            for pattern in ignore_list:
                if fnmatch(source_path.name, pattern):
                    break
            else:
                yield SourceFile(self, source_path)


class VLine:
    """A Vline is a line after preprocessing"""

    __slots__ = ("canonical_line", "number", "origin_line")

    def __init__(
        self, canonical_line: CanonicalLine, number: int, origin_line: Line
    ) -> None:
        # this isn't the line number from the source file,
        # but another line number computed after preprocessing
        self.number = number
        self.origin_line = origin_line
        self.canonical_line = canonical_line

    @property
    def source_file(self) -> SourceFile:
        return self.origin_line.source_file

    @property
    def codebase(self) -> Codebase:
        return self.origin_line.codebase

    @property
    def origin_line_number(self) -> int:
        return self.origin_line.line_number

    @property
    def number_format(self) -> str:
        """Formats the line number, highlighting the line type.
        It's a virtual line, numbered after preprocessing.
        """
        return f"v{self.number}"

    def __str__(self) -> str:
        return str(self.origin_line)

    def __repr__(self) -> str:
        return f"<VLine {self.origin_line}>"


# remove consecutive empty lines
def normalize_empty_lines(
    lines: Iterable[Tuple[CanonicalLine, Line]]
) -> Generator[VLine, None, None]:
    empty_line = None
    line_number = 1
    for canonical_line, source_line in lines:
        if canonical_line:
            if empty_line is not None:
                yield empty_line
                empty_line = None
            yield VLine(canonical_line, line_number, source_line)
            line_number += 1
        elif empty_line is None:
            empty_line = VLine(canonical_line, line_number, source_line)
            line_number += 1


TupleLineGroup = Tuple[CanonicalLine, ...]
SetLineGroup = FrozenSet[CanonicalLine]
LineGroup = TypeVar("LineGroup", TupleLineGroup, SetLineGroup)
SourceId = Tuple[LineGroup, VLine]


def preprocess_source(
    source_file: SourceFile, window_size: int, line_group_type: Type[LineGroup]
) -> Generator[SourceId[LineGroup], None, None]:
    keylist = list(
        normalize_empty_lines(
            (canonicalize(tokenize(line_content)), source_line)
            for source_line, line_content in source_file.lines()
        )
    )
    # print('\n'.join(str(k.canonical_line) for k in keylist))

    # keylist now contains a list of preprocessed line, along with
    # where they appeared in the first place

    # yield groups of preprocessed lines, so they can be later looked up
    for line_group in sliding_window(keylist, window_size):
        # the line number of the group is the line number of its first line
        source_line = line_group[0]
        yield (
            line_group_type(map(lambda l: l.canonical_line, line_group)),
            source_line,
        )


def preprocess_sources(
    source_files: Iterable[SourceFile],
    window_size: int,
    line_group_type: Type[LineGroup],
) -> Generator[SourceId[LineGroup], None, None]:
    """creates line groups"""
    for source_file in progressbar(source_files):
        yield from preprocess_source(source_file, window_size, line_group_type)


LineGroupMatches = List[Tuple[LineGroup, List[VLine]]]
LineGroupOccurences = Mapping[LineGroup, List[VLine]]


def correlate_line_groups(
    line_groups: Iterable[SourceId[LineGroup]]
) -> LineGroupMatches[LineGroup]:
    hitmap: LineGroupOccurences[LineGroup] = defaultdict(list)

    for line_group, source_line in line_groups:
        hitmap[line_group].append(source_line)

    return list(hitmap.items())


def correlate_sources(
    source_files: Iterable[SourceFile],
    window_size: int,
    line_group_type: Type[LineGroup],
) -> LineGroupMatches[LineGroup]:
    line_groups = preprocess_sources(source_files, window_size, line_group_type)
    return correlate_line_groups(line_groups)


def find_sources(
    targets: Iterable[str], ignore_list: List[str], pattern: Optional[str]
) -> List[SourceFile]:
    path_targets = map(PosixPath, targets)

    if pattern is None:
        return [SourceFile(None, path) for path in path_targets]

    return list(
        chain.from_iterable(
            Codebase.from_path(path).find_sources(ignore_list, pattern)
            for path in path_targets
        )
    )


def pairs(seq: Sequence[T]) -> Generator[Tuple[T, T], None, None]:
    seq_len = len(seq)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            yield (seq[i], seq[j])


class LineMatch(NamedTuple):
    left: VLine
    right: VLine


CodebasePair = Tuple[Codebase, Codebase]
SourceFilePair = Tuple[SourceFile, SourceFile]

FileMatchesMapping = Mapping[CodebasePair, Mapping[SourceFilePair, List[LineMatch]]]


def group_by_file_pairs(matches: LineGroupMatches[LineGroup]) -> FileMatchesMapping:
    # a map from a pair of codebases to a map of pair of files
    # to list of pairs of matches
    codebases_map: FileMatchesMapping = defaultdict(lambda: defaultdict(list))
    for _linegroup, occurences in progressbar(matches):
        for line_a, line_b in pairs(occurences):
            base_a, base_b = line_a.codebase, line_b.codebase
            if base_a is base_b:
                continue

            pairmaker = base_a.pairmaker(base_b)
            codebase_pair = pairmaker(base_a, base_b)
            line_pair = pairmaker(line_a, line_b)
            file_pair = pairmaker(line_a.source_file, line_b.source_file)
            codebases_map[codebase_pair][file_pair].append(LineMatch(*line_pair))

    return {k: dict(v) for k, v in codebases_map.items()}


class LineRun:
    __slots__ = ("lines", "neighbors", "visited")

    def __init__(self) -> None:
        self.lines: List[VLine] = []
        self.neighbors: Set[LineRun] = set()
        self.visited: object = None

    def __repr__(self) -> str:
        return f"<LineRun {self.lines[0]} -> {self.lines[-1]}>"

    def get_run_cluster(self, flag: object) -> Generator["LineRun", None, None]:
        if self.visited is flag:
            return

        self.visited = flag
        yield self
        for neighbor in self.neighbors:
            yield from neighbor.get_run_cluster(flag)


class VLineCluster:
    __slots__ = ("lines", "runs_count")

    def __init__(self) -> None:
        # a list is used for performance, but is really is a set
        self.lines: List[VLine] = []
        self.runs_count: int = 0

    def vlength(self, window_size: int) -> int:
        return len(self.lines) + (window_size - 1) * self.runs_count

    def update(self, run: LineRun) -> None:
        self.runs_count += 1
        self.lines.extend(run.lines)

    def __repr__(self) -> str:
        suffix = ""
        if self.lines:
            suffix = f" {self.lines[0]}"
        return f"<VLineCluster #{len(self.lines)}{suffix}>"


def compute_runs(
    matches: List[LineMatch],
    run_map: Dict[VLine, LineRun],
    pair_i: int,
    window_size: int,
) -> List[LineRun]:
    """find runs of lines appearing in any match"""

    # the same item can be involved in more than a single pair
    # the set can be avoided by changing algorithms instead
    match_list = list({match[pair_i] for match in matches})
    # insane stuff can be done here
    # we may sort this O(file_size), which may not be a good idea,
    # given that most of the time, len(matches) <<< file_size.
    # we could store a list of file lines somewhere, and test if the
    # line has matched using a hash set.
    match_list.sort(key=lambda m: m.number)

    runs: List[LineRun] = []

    run = LineRun()
    prev_line_number: Optional[int] = None

    def end_run() -> None:
        nonlocal run
        if run.lines:
            runs.append(run)
            run = LineRun()

    for line in match_list:
        cur_line_number = line.number
        has_prev_line = prev_line_number is not None
        if (
            has_prev_line
            and cur_line_number > cast(int, prev_line_number) + window_size
        ):
            end_run()

        run_map[line] = run
        run.lines.append(line)
        prev_line_number = cur_line_number

    end_run()
    return runs


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


def rebuild_filemap(
    filemap: Mapping[CodebasePair, Mapping[SourceFilePair, TInput]],
    func: Callable[[CodebasePair, SourceFilePair, TInput], TOutput],
) -> Mapping[CodebasePair, Mapping[SourceFilePair, TOutput]]:
    res = {}
    for codebase_pair, file_map in progressbar(filemap.items()):
        file_res = {}
        for file_pair, matches in file_map.items():
            file_res[file_pair] = func(codebase_pair, file_pair, matches)
        res[codebase_pair] = file_res
    return res


ClusterMatch = Tuple[VLineCluster, VLineCluster]
FileClusterMapping = Mapping[CodebasePair, Mapping[SourceFilePair, List[ClusterMatch]]]


def group_lines(
    options: Namespace, codebases_map: FileMatchesMapping
) -> FileClusterMapping:
    """
    {
       (base_a, base_b): {
           (file_a, file_b): [
               (line_a, line_b)
           ]
       }
    }
    """

    def _group_lines(
        _codebase_pair: CodebasePair,
        file_pair: SourceFilePair,
        matches: List[LineMatch],
    ) -> List[ClusterMatch]:
        window_size = options.window_size
        left_file = file_pair[0]
        run_map: Dict[VLine, LineRun] = {}

        left_runs = compute_runs(matches, run_map, 0, window_size)
        right_runs = compute_runs(matches, run_map, 1, window_size)

        for left_line, right_line in matches:
            left_run = run_map[left_line]
            right_run = run_map[right_line]
            left_run.neighbors.add(right_run)
            right_run.neighbors.add(left_run)

        clusters = []
        # runs should be connected now, so it doesn't matter which side is
        # iterated over
        for run in left_runs:
            if run.visited is True:
                continue

            left_cluster = VLineCluster()
            right_cluster = VLineCluster()
            for cur_run in run.get_run_cluster(True):
                if cur_run.lines[0].source_file is left_file:
                    left_cluster.update(cur_run)
                else:
                    right_cluster.update(cur_run)

            assert left_cluster.lines and right_cluster.lines
            clusters.append((left_cluster, right_cluster))
        assert all(map(lambda c: c.visited is True, left_runs))
        assert all(map(lambda c: c.visited is True, right_runs))
        return clusters

    return rebuild_filemap(codebases_map, _group_lines)


CodebaseScoreMapping = Dict[Tuple[Codebase, Codebase], int]


def rate_line(line: VLine) -> int:
    return len(line.canonical_line)


def rate_cluster(window_size: int, cluster: VLineCluster) -> int:
    # how long the content of the lines is matters
    significance = sum(map(rate_line, cluster.lines))
    # but not as much as how many lines the cluster holds
    return cluster.vlength(window_size) ** 2 * significance


def rate_cluster_match(window_size: int, match: ClusterMatch) -> int:
    left_cluster, right_cluster = match
    cluster_rater = partial(rate_cluster, window_size)
    return sum(map(cluster_rater, (left_cluster, right_cluster)))


def rate_grouped_lines(
    codebases_map: FileClusterMapping, window_size: int
) -> CodebaseScoreMapping:

    res = {}
    cluster_match_rater = partial(rate_cluster_match, window_size)
    for codebase_pair, file_map in codebases_map.items():
        cluster_matches = chain.from_iterable(file_map.values())
        cluster_matches_score = sum(map(cluster_match_rater, cluster_matches))
        codebase_sizes = sum(map(Codebase.get_size, codebase_pair))
        # the bigger the codebase, the more false positives.
        score = 0
        if codebase_sizes:
            score = cluster_matches_score // codebase_sizes
        res[codebase_pair] = score

    return res


def codebase_pair_name(codebase_pair: CodebasePair) -> str:
    """Returns a string uniquely identifying a pair of codebases"""
    base_a, base_b = codebase_pair
    base_a_name = base_a.name.replace("-", "--")
    base_b_name = base_b.name.replace("-", "--")
    # underscores are mandatory to avoid conflicts
    # consider the pairs ("a", "-b") and ("a-", "b")
    return f"{base_a_name}_-_{base_b_name}"


def compress_ranges(range_starts: List[int], range_width: int) -> Generator[Tuple[int, int], None, None]:
    """Compresses consecutive sequences of ints into ranges

    >>> list(compress_ranges([1], 10))
    [(1, 11)]

    >>> list(compress_ranges([1, 2, 3, 13], 10))
    [(1, 13), (13, 23)]

    >>> list(compress_ranges([1, 2, 3, 12], 10))
    [(1, 22)]
    """
    ranges_iter = iter(range_starts)
    first_range = next(ranges_iter, None)
    if first_range is None:
        return

    begin: int = first_range
    last: int = first_range
    for item in ranges_iter:
        if item >= last + range_width:
            yield (begin, last + range_width)
            begin = item
        last = item

    yield (begin, last + range_width)


class ClusterMatchScore(NamedTuple):
    score: int
    cluster_match: ClusterMatch
    file_pair: SourceFilePair

    def json_dump(self, window_size: int) -> str:
        left_file, right_file = self.file_pair
        left, right = self.cluster_match
        left_lines = [vline.origin_line_number for vline in left.lines]
        right_lines = [vline.origin_line_number for vline in right.lines]
        left_lines.sort()
        right_lines.sort()
        return json.dumps(
            {
                "left_file": str(left_file),
                "right_file": str(right_file),
                "left": list(compress_ranges(left_lines, window_size)),
                "right": list(compress_ranges(right_lines, window_size)),
            }
        )


def log_clusters(
    log_dir_path: str, window_size: int, cluster_map: FileClusterMapping
) -> None:
    log_dir = Path(log_dir_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("logging per-codebase correlations")
    for codebase_pair, file_map in progressbar(cluster_map.items()):
        codebase_pair_file_path = log_dir / f"{codebase_pair_name(codebase_pair)}.json"
        # json file format:
        # [{"left_file": left_file, "right_file": right_file,
        #   "left": [1, 2], "right": [1, 2, 3]}]
        with codebase_pair_file_path.open("w") as codebase_pair_file:
            # rate clusters
            cluster_scores = []
            for file_pair, clusters in file_map.items():
                for cluster in clusters:
                    score = rate_cluster_match(window_size, cluster)
                    cluster_scores.append(ClusterMatchScore(score, cluster, file_pair))

            # sort by decreasing score
            cluster_scores.sort(key=lambda c: -c.score)
            codebase_pair_file.write("[")
            for cluster_i, cluster_score in enumerate(cluster_scores):
                if cluster_i:
                    codebase_pair_file.write(",")
                codebase_pair_file.write(cluster_score.json_dump(window_size))
            codebase_pair_file.write("]")


def cluster_matches(
    options: Namespace,
    source_files: Iterable[SourceFile],
    window_size: int,
    group_type: Type[LineGroup],
) -> FileClusterMapping:
    logger.info("correlating sources")
    matches = correlate_sources(source_files, window_size, group_type)

    logger.info("grouping codebases / files")
    file_pairs_matches = group_by_file_pairs(matches)

    logger.info("clustering lines")
    return group_lines(options, file_pairs_matches)


def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv[1:]

    options = _trish_parser().parse_args(args=args)
    source_files = find_sources(options.targets, options.ignore, options.pattern)
    window_size = options.window_size

    # the type isn't assigned to a variable in order to keep mypy happy
    # mypy doesn't seem to handle union types parametrizing typevars
    if options.unordered_line_group:
        clustered_matches = cluster_matches(
            options, source_files, window_size, frozenset
        )
    else:
        clustered_matches = cluster_matches(options, source_files, window_size, tuple)

    if options.clusters_log_dir is not None:
        log_clusters(options.clusters_log_dir, window_size, clustered_matches)

    logger.info("rating groups")
    scores = rate_grouped_lines(clustered_matches, window_size)

    for codebase_pair, score in scores.items():
        codebase_a, codebase_b = codebase_pair
        print(f"{score}\t{codebase_a.name}\t{codebase_b.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
