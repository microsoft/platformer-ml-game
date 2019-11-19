# --------------------------------------------------------------------------------------------------
#  Copyright (c) 2018 Microsoft Corporation
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------------------------------

"""Module providing sampler implementations."""

import numpy as np

from .utils import assert_range
from .abc import Sampler


class UniformSampler(Sampler):
    """
    A sampler that samples values uniformly.
    """

    def __init__(self, initial_size=0):
        """
        Args:
            initial_size -- the initial size of the sampler
        """
        self._indices = list(range(0, initial_size))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        return 1.0 / len(self._indices)

    def get_probabilities(self, indices, start=0, end=None):
        start, end = assert_range((start, end), (0, len(self)))
        probabilities = []
        probability = 1.0 / (end - start)
        for index in indices:
            assert start <= index < end, "Index must be within range"
            probabilities.append(probability)

        return probabilities

    def resize(self, num_indices):
        self._indices.clear()
        self._indices.extend(range(0, num_indices))

    def reset(self):
        self._indices.clear()

    def append(self):
        self._indices.append(len(self._indices))

    def sample(self, prng, count, start=0, end=None, replace=True):
        start, end = assert_range((start, end), (0, len(self)))
        if not replace:
            assert end - start >= count, "Not enough values in range to sample"

        population = self._indices[start:end]
        return prng.choice(population, count, replace)


class PrioritizedSampler(Sampler):
    """Class which implements a priority sampler."""

    def __init__(self, default_priority=1, priorities=None):
        """
        Keyword Args:
            priorities -- the initial values of the distribution [None]
            default_priority -- the default priority value [1]
        """
        if priorities is None:
            self._priorities = []
        else:
            self._priorities = priorities

        self._indices = list(range(0, len(self._priorities)))
        self._total = sum(self._priorities)
        self._default_priority = default_priority

    def __len__(self):
        return len(self._priorities)

    def __getitem__(self, index):
        return self._priorities[index] / self._total

    def get_probabilities(self, indices, start=0, end=None):
        start, end = assert_range((start, end), (0, len(self)))
        priority_sum = sum([self._priorities[i] for i in range(start, end)])
        probabilities = []
        for index in indices:
            assert start <= index < end
            probabilities.append(self._priorities[index] / priority_sum)

        return probabilities

    @property
    def default_priority(self):
        """The default priority value."""
        return self._default_priority

    def set_priority(self, index, priority):
        """Sets the priority of an index.

        Args:
            index -- the index to update
            priority -- the new priority
        """
        if not priority:
            priority = self._default_priority

        self._total += priority - self._priorities[index]
        self._priorities[index] = priority

    def append_priority(self, priority):
        """Appends an index to the sampler with a specified priority.

        Args:
            priority -- the priority of the new index
        """
        if priority is None:
            priority = self._default_priority

        self._priorities.append(priority)
        self._indices.append(len(self._indices))
        self._total += priority

    def append(self):
        self.append_priority(self._default_priority)

    def resize_priority(self, priorities):
        """Resizes the sampler to match the provided priority values.

        Args:
            priorities -- the priorities to use for the indices
        """
        self._priorities = priorities
        self._indices.clear()
        self._indices.extend(range(0, len(priorities)))
        self._total = sum(priorities)

    def resize(self, num_indices):
        self.resize_priority([self._default_priority]*num_indices)

    def reset(self):
        self._priorities.clear()
        self._indices.clear()
        self._total = 0

    def sample(self, prng, count, start=0, end=None, replace=True):
        start, end = assert_range((start, end), (0, len(self)))

        if not replace:
            non_zero = sum([self._priorities[i] > 0 for i in range(start, end)])
            assert non_zero >= count, "Too few non-zero priorities to sample"

        population = self._indices[start:end]
        probabilities = np.array(self._priorities[start:end], dtype=np.float32)
        probabilities /= probabilities.sum()

        return prng.choice(population, count, replace, probabilities)


def get_sampler(mode: str = 'uniform', default_priority=1, initial_size=0, priorities=None):
    """Factory function for samplers.

    Args:
        size -- the initial number of supported indices for the sampler

    Keyword Args:
        mode -- the type of sampler ('uniform' or 'prioritized') ['uniform']
        default_priority -- default priority (for prioritized sampler) [1]
    """
    assert mode in ['uniform', 'prioritized']
    if mode == 'uniform':
        return UniformSampler(initial_size)
    elif mode == 'prioritized':
        return PrioritizedSampler(default_priority, priorities)
    else:
        raise ValueError('Unknown sampler mode %s.' % mode)
