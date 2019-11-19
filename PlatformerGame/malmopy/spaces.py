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

"""Module containing custom spaces used by malmopy environments."""

import numpy as np
import gym

from . import Command, History


class DiscreteCommand(gym.spaces.Discrete, Command):
    """Space which has a fixed number of text commands."""

    def __init__(self, verbs):
        """
        Args:
            verbs -- a list of command verbs
        """
        self._verbs = verbs
        super(DiscreteCommand, self).__init__(len(verbs))

    @property
    def verbs(self):
        return self._verbs

    def to_commands(self, sample_n):
        if self.contains(sample_n):
            sample_n = [sample_n]

        commands = []
        for sample in sample_n:
            assert self.contains(sample)
            commands.append(self._verbs[sample])

        return commands

    def __repr__(self):
        return "Command(verbs: {})".format(self.verbs)


class ContinuousTupleCommand(gym.spaces.Tuple, Command):
    """
    Command space which uses a combination of a fixed number of verbs and a continuous
    argument.
    """

    def __init__(self, verbs, low=-1.0, high=1.0):
        """
        Args:
            verbs -- a list of command verbs

        Keyword Args:
            low -- the minimum command argument [-1.0]
            high -- the maximum command argument [1.0]
        """
        self._verbs = verbs
        spaces = (gym.spaces.Discrete(len(verbs)), gym.spaces.Box(low, high, (1,), np.float32))
        super(ContinuousTupleCommand, self).__init__(spaces)

    @property
    def verbs(self):
        return self._verbs

    def to_commands(self, sample_n):
        if self.contains(sample_n):
            sample_n = [sample_n]

        commands = []
        for sample in sample_n:
            assert self.contains(sample)
            index, arg = sample
            commands.append("{0} {1:.2f}".format(self._verbs[index], arg[0]))

        return commands

    def __repr__(self):
        return "Command(verbs: {}, args: {})".format(self.verbs, self.spaces[1])


class ContinuousStepCommand(gym.spaces.Discrete, Command):
    """
    Command space which uses a combination of a fixed number of verbs and a
    discretized continuous argument.
    """

    def __init__(self, verbs, low=-1.0, high=1.0, steps=10):
        """
        Args:
            verbs -- a list of command verbs

        Keyword Args:
            low -- the minimum command argument [-1.0]
            high -- the maximum command argument [1.0]
            steps -- the number of discretized steps to use [10]
        """
        self._verbs = verbs
        self._low = low
        self._delta = (high - low)/steps
        self._dims = (len(verbs), steps)
        super(ContinuousStepCommand, self).__init__(np.prod(self._dims))

    @property
    def verbs(self):
        return self._verbs

    def to_commands(self, sample_n):
        if self.contains(sample_n):
            sample_n = [sample_n]

        commands = []
        for sample in sample_n:
            assert self.contains(sample)
            verb_index, arg_index = np.unravel_index(sample, self._dims)
            verb = self._verbs[verb_index]
            arg = self._low + arg_index*self._delta
            commands.append("{0} {1:.2f}".format(verb, arg))

        return commands

    def __repr__(self):
        steps = self._dims[1]
        high = self._low + steps*self._delta
        return "Command(verbs: {}, args: {})".format(self.verbs, (self._low, high, steps))


class StringCommand(gym.Space, Command):
    """Space which allows arbitrary string commands prepended with set verbs."""

    def __init__(self, verbs):
        """
        Args:
            verbs -- the verbs which must prepend the string commands
        """
        super(StringCommand, self).__init__((1,), 'U')
        self._verbs = verbs

    def sample(self):
        raise NotImplementedError

    def to_commands(self, sample_n):
        return list(sample_n)

    @property
    def verbs(self):
        return self._verbs

    def contains(self, x):
        if not x.dtype.kind in ('S', 'U'):
            return False

        startswith = np.zeros(x.shape, np.bool)
        for verb in self._verbs:
            startswith = startswith | np.char.startswith(x, verb)

        return startswith.all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "StringCommand(verbs: {})".format(self.verbs)


class BoxHistory(gym.spaces.Box, History):
    """Space which describes a history of Box observations."""

    def __init__(self, length, space):
        """
        Args:
            length -- the length of the history
            space -- a Box space
        """
        assert isinstance(space, gym.spaces.Box)
        self._length = length
        self._inner = space
        shape = (length, ) + space.shape
        dtype = space.dtype
        low = np.zeros(shape, dtype)
        high = np.zeros(shape, dtype)
        low[:] = space.low
        high[:] = space.high
        super(BoxHistory, self).__init__(low=low, high=high, dtype=dtype)

    @property
    def inner(self):
        return self._inner

    @property
    def length(self):
        return self._length


class MultiDiscreteHistory(gym.Space, History):
    """Space which describes a history of MultiDiscrete observations."""

    def __init__(self, length, space):
        """
        Args:
            length -- the length of the history
            space -- a MultiDiscrete space
        """
        assert isinstance(space, gym.spaces.MultiDiscrete)
        self._length = length
        self._inner = space
        shape = (length, space.nvec.size)
        self.nvec = np.zeros(shape, space.dtype)
        self.nvec[:] = space.nvec
        super(MultiDiscreteHistory, self).__init__(shape, space.dtype)

    @property
    def inner(self):
        return self._inner

    @property
    def length(self):
        return self._length

    def sample(self):
        return (gym.spaces.np_random.random_sample(self.shape) * self.nvec).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and (x < self.nvec).all() and x.dtype.kind in 'ui'

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]


class MultiBinaryHistory(gym.Space, History):
    """Space which describes a history of MultiBinary observations."""

    def __init__(self, length, space):
        """
        Args:
            length -- the length of the history
            space -- a MultiBinary space
        """
        assert isinstance(space, gym.spaces.MultiBinary)
        self._length = length
        self._inner = space
        super(MultiBinaryHistory, self).__init__((length, space.n), space.dtype)

    @property
    def inner(self):
        return self._inner

    @property
    def length(self):
        return self._length

    def sample(self):
        return gym.spaces.np_random.randint(0, 2, self.shape).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and ((x == 0) | (x == 1)).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]


class DiscreteHistory(gym.Space, History):
    """Space which describes a history of Discrete observations."""

    def __init__(self, length, space):
        """
        Args:
            length -- the length of the history
            space -- a Discrete space
        """
        assert isinstance(space, gym.spaces.Discrete)
        self._length = length
        self._inner = space
        super(DiscreteHistory, self).__init__((length, 1), space.dtype)

    @property
    def inner(self):
        return self._inner

    @property
    def length(self):
        return self._length

    def sample(self):
        return gym.spaces.np_random.randint(self._inner.n, size=self.shape).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and (x < self._inner.n).all() and x.dtype.kind in 'ui'

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]


def to_history_space(length, space):
    """Converts a space to a history space.

    Args:
        length -- the length of the history
        space -- the space from which observations are drawn

    Returns a space which represents a history of observations drawn from the original space
    """
    if isinstance(space, gym.spaces.Box):
        return BoxHistory(length, space)

    if isinstance(space, gym.spaces.MultiDiscrete):
        return MultiDiscreteHistory(length, space)

    if isinstance(space, gym.spaces.MultiBinary):
        return MultiBinaryHistory(length, space)

    if isinstance(space, gym.spaces.Discrete):
        return DiscreteHistory(length, space)

    raise NotImplementedError
