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

"""Model which provides a class which encapsulates a library of demonstrated trajectories."""

from argparse import Namespace

import gym
import numpy as np

from .episode import EpisodicMemory
from .batches import TrajectoryBatch, batch_dtype
from .. import get_sampler, PrioritizedSampler

class TrajectoryLibrary(object):
    """
    This class contains multiple trajectories, which are completed episodes, and
    makes it possible to sample them for training.
    """

    @staticmethod
    def add_args_to_parser(parser):
        """Add arguments to the parser for library creation

        Keyword Args:
            num_trajectories -- The number of trajectories to store in the library [500]
            sampler -- The type of sampler to use for memory sampling ["prioritized"]
            default_priority -- The default priority of a prioritized sampler [1.0]
        """
        parser.add_argument("--num_trajectories", type=int, default=500,
                            help="The number of trajectories to store in the library")
        parser.add_argument("--sampler", type=str, default="prioritized",
                            choices=["uniform", "prioritized"],
                            help="The type of sampler to use for memory sampling")
        parser.add_argument("--default_priority", type=float, default=1.0,
                            help="The default priority of a prioritized sampler")

    @staticmethod
    def create(args, observation_space, action_space):
        """Create a Trajectory library"""
        assert isinstance(args, Namespace)
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)
        if args.sampler == "uniform":
            from .. import UniformSampler
            sampler = UniformSampler()
        elif args.sampler == "prioritized":
            sampler = PrioritizedSampler(args.default_priority)
        else:
            raise NotImplementedError

        return TrajectoryLibrary(observation_space, action_space, args.num_trajectories, sampler)

    def __init__(self, observation_space, action_space, capacity=None, sampler=None):
        """
        Args:
            observation_space -- the space from which observations will be drawn
            action_space -- the space from which actions will be drawn

        Keyword Args:
            capacity -- the maximum number of trajectories to store [None]
            sampler -- the sampler to use [None]
        """
        self._trajectories = []
        self._active = EpisodicMemory(observation_space, action_space)
        if sampler is None:
            self._sampler = get_sampler()
        else:
            self._sampler = sampler

        self._observation_space = observation_space
        self._action_space = action_space
        self._current_index = None
        self._capacity = capacity

    @property
    def observation_space(self):
        """Space from which observations in the stored trajectories are drawn"""
        return self._observation_space

    @property
    def action_space(self):
        """Space from which actions in the store trajectories are drawn"""
        return self._action_space

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, key):
        return self._trajectories[key]

    @property
    def current(self):
        """The current trajectory."""
        return self._trajectories[self._current_index]

    @property
    def active(self):
        """The active (under construction) trajectory."""
        return self._active

    @property
    def is_full(self):
        """Whether the library is at maximum capacity."""
        if self._capacity:
            return len(self._trajectories) >= self._capacity

        return False

    @property
    def capacity(self):
        """The capacity of the library.

        Description:

        Once the library reaches this value, it will begin to overwrite old entries.
        None indicates the library will keep growing as new trajectories are added.
        """
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = value
        if self._capacity:
            if len(self._trajectories) > self._capacity:
                del self._trajectories[self._capacity:]

    def _move_next(self):
        if not self.is_full:
            self._current_index = len(self._trajectories)
            self._trajectories.append(self._active)
            self._active.index = self._current_index
            self._sampler.append()
        else:
            self._current_index = (self._current_index + 1) % self._capacity
            self._trajectories[self._current_index] = self._active
            self._active.index = self._current_index

        self._active = EpisodicMemory(self._observation_space, self._action_space)

    def complete_trajectory(self, observation, priority):
        """Append the terminal state with fictive action and reward.

        Args:
            observation -- the final observation
            priority -- the priority for the completed trajectory
        """
        self._active.append(observation, self._action_space.sample(), 0, True)
        self._move_next()
        if isinstance(self._sampler, PrioritizedSampler):
            self._sampler.set_priority(self._current_index, priority)

    def sample(self, prng, count):
        """Sample trajectories from the library.

        Args:
            prng -- pseudo-random number generator
            count -- the number of trajectories to sample

        Returns [trajectory, ...] list of trajectories
        """
        indices = self._sampler.sample(prng, count, replace=len(self) < count)
        return [self._trajectories[index] for index in indices]


    def minibatch(self, prng, count):
        """Create an aligned trajectory minibatch for use in training.

        Args:
            prng -- pseudo-random number generator
            count -- the number of trajectories to sample

        Returns a TrajectoryBatch object
        """
        return TrajectoryBatch(self.sample(prng, count))

    def get_probabilities(self, batch):
        """Get the sample probabilities for the trajectories in the provided batch.

        Args:
            batch -- a TrajectoryBatch

        Returns a probability per trajectory in the batch of that index being sampled
        """
        indices = [trajectory.index for trajectory in batch.trajectories]
        return self._sampler.get_probabilities(indices)

    def save(self, file):
        """Save the trajectories to the disk.

        Args:
            file -- a file object or the path to the file on disk
        """
        np.save(file, np.array([trajectory.to_ndarray() for trajectory in self._trajectories]))

    def load(self, file):
        """Load the trajectories from the disk.

        Args:
            file -- a file object or the path to the file on disk
        """
        self._trajectories = []
        for array in np.load(file):
            episode = EpisodicMemory(self.observation_space, self.action_space)
            episode.from_ndarray(array, None)
            self._trajectories.append(episode)

        self._current_index = 0

    def move_to_buffer(self, num_transitions=None):
        """Create a transition buffer from the stored trajectories.

        Description:
            This method moves some number of the stored trajectories to a buffer
            object in a way which minimizes memory overhead.

        Keyword Args:
            num_transitions -- the number of transitions to move. [None]
                               ('None' indicates to move all trajectories.)
        """
        max_size = 0
        for trajectory in reversed(self._trajectories):
            max_size += len(trajectory)
            if num_transitions and max_size > num_transitions:
                break

        buffer = np.empty(max_size, batch_dtype(self.observation_space, self.action_space))
        index = 0
        while index < max_size:
            trajectory = self._trajectories.pop()
            end = index + len(trajectory)
            buffer[index:end] = trajectory.to_ndarray()
            index = end

        return buffer

    @property
    def num_transitions(self):
        """The total number of transitions in all trajectories in the library."""
        return sum([len(trajectory) for trajectory in self._trajectories])
