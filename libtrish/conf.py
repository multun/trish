from importlib import import_module
from abc import ABC, abstractmethod

from .stats import DefaultChalStats
from .canon import canonicalize
from .lines_meta import compute_meta
from .token import get_tokens, group_tokens, token_split_re

from .chal import evaluate_chal, get_pairs, evaluate_match, Challenge


class BaseTrishConf(ABC):
    @abstractmethod
    def canonicalize(self, groups):
        raise NotImplemented()

    @abstractmethod
    def group_tokens(self, tokens):
        raise NotImplemented()

    @abstractmethod
    def get_tokens(self, lines):
        raise NotImplemented()

    @abstractmethod
    def compute_meta(self):
        raise NotImplemented()

    @abstractmethod
    def get_pairs(self):
        raise NotImplemented()

    @abstractmethod
    def evaluate_chal(self):
        raise NotImplemented()

    @abstractmethod
    def evaluate_match(self):
        raise NotImplemented()

    @abstractmethod
    def exporter(self, chal, stats):
        raise NotImplemented()

class DefaultTrishConf(BaseTrishConf):
    Challenge = Challenge
    ChalStats = DefaultChalStats
    token_split_re = token_split_re
    ignored_tokens = set()
    breaking_tokens = set('\n')

    get_tokens = get_tokens
    group_tokens = group_tokens
    canonicalize = canonicalize

    get_pairs = get_pairs
    evaluate_chal = evaluate_chal
    evaluate_match = evaluate_match

    compute_meta = compute_meta

    def exporter(self, chal, stats):
        print(f'{str(stats)}\t{str(chal)}')


def import_conf(path):
    spath = path.split('.')
    return getattr(spath[-1], import_module('.'.join(spath[:-1])))
