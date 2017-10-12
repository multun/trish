from abc import ABC, abstractmethod
from collections import defaultdict


class BaseChalStats(ABC):
    @abstractmethod
    def register(self, conf):
        raise NotImplemented()

    @abstractmethod
    def finalize(self, hit):
        raise NotImplemented()


class DefaultChalStats(BaseChalStats):
    def __init__(self, conf):
        self.hits = []

    def register(self, hit):
        if hit > len(self.hits):
            for i in range(hit - len(self.hits)):
                self.hits.append(0)
        self.hits[hit - 1] += 1

    def finalize(self, conf):
        for i in range(len(self.hits)):
            if i < len(self.hits) - 1:
                self.hits[i] -= self.hits[i + 1]

    def weight(self, hit):
        return hit ** 2

    def __str__(self):
        return str(sum(self.weight(hit + 1) for hit in self.hits))
