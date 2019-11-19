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

"""Module containing a class which implements a replay memory"""

import logging

import numpy as np
import gym

from .. import Memory
from ..samplers import PrioritizedSampler, get_sampler
from .batches import Batch, batch_dtype


class ReplayMemory(Memory):
    #pylint: disable=too-many-public-methods
    """
    Simple representation of agent memories
    """

    ALPHA = 0.6
    EPSILON = 0.001
    EPSILON_DEMO = 1.0

    def __init__(self, observation_space, action_space, sampler=None,
                 is_demo=False, max_size=None, buffer=None):
        """
        Description:
            When creating a replay memory you must either provide an existing buffer
            or specify a maximum size.

        Args:
            observation_space -- the space of the stored observations
            action_space -- the space of the stored actions

        Keyword Args:
            max_size -- the maximum size of the replay memory
            sampler -- the sampler to use when creating minibatches
            is_demo -- whether this is a demonstration memory
            buffer -- the existing buffer to use
        """
        #pylint: disable=too-many-arguments
        assert ((max_size is not None and buffer is None) or
                (max_size is None and buffer is not None)), "max_size or buffer must be specified"
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)
        self._observation_space = observation_space
        self._action_space = action_space
        if sampler is None:
            self._sampler = get_sampler()
        else:
            self._sampler = sampler

        self._is_demo = is_demo
        self._pos = 0
        self._logger = logging.getLogger(self.__class__.__name__)

        # Generate internal structured array shape
        if buffer is None:
            self._count = 0
            self._max_size = max_size
            self._buffer = np.empty(max_size, batch_dtype(observation_space, action_space))
        else:
            assert observation_space.contains(buffer['states'][0])
            assert action_space.contains(buffer['actions'][0])
            self._buffer = buffer
            self._count = len(buffer)
            self._max_size = self._count
            self._sampler.resize(self._count)

        self._total_reward = 0
        self._total_terminals = 0

    def to_ndarray(self):
        return self._buffer[:self._count]

    def __getitem__(self, key):
        return self._buffer[key]

    def reset(self):
        self._buffer[...] = 0
        self._count = 0
        self._sampler.reset()

    def from_ndarray(self, array, sampler):
        assert self._observation_space.contains(array['states'][0])
        assert self._action_space.contains(array['actions'][0])
        self._buffer = array
        self._pos = 0
        self._count = len(array)
        if sampler:
            assert len(sampler) == len(self)
            self._sampler = sampler
        else:
            self._sampler = get_sampler(initial_size=len(self))

        return self

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def size(self):
        """The maximum size of the replay memory."""
        return self._max_size

    def __len__(self):
        return self._count

    @property
    def is_full(self):
        return self._count >= self._max_size

    @property
    def reward_per_step(self) -> float:
        """Average reward per set"""
        return self._total_reward / len(self)

    @property
    def is_demo(self):
        """Whether this memory contains demonstration data"""
        return self._is_demo

    @property
    def reward_per_episode(self) -> float:
        """Average per-episode reward"""
        return self._total_reward / max(1, self._total_terminals)

    def append(self, observation, action, reward, done, priority=None):
        assert self._observation_space.contains(observation)
        assert self._action_space.contains(action)

        # If the current position is occupied by an existing sample
        if self._count == self._max_size:
            self._total_reward -= self._buffer[self._pos]['rewards']
            self._total_terminals -= self._buffer[self._pos]['terminals']

        self._buffer[self._pos] = (observation, action, reward, done)

        if isinstance(self._sampler, PrioritizedSampler):
            if self._count < self._max_size:
                self._sampler.append_priority(priority)
            else:
                self._sampler.set_priority(self._pos, priority)
        else:
            if self._count < self._max_size:
                self._sampler.append()

        self._total_reward += self._buffer[self._pos]['rewards']
        self._total_terminals += self._buffer[self._pos]['terminals']

        self._count = min(self._max_size, self._count + 1)
        self._pos = (self._pos + 1) % self._max_size

    def get_state(self, index: int) -> np.ndarray:
        """Return the specified state

        Args:
            index -- State's index

        Returns:
            the state at the specified index
        """
        index %= self._max_size
        return self._buffer['states'][index]

    def get_action(self, index: int) -> int:
        """Return the specified action

        Args:
            index -- Action's index

        Returns:
            the action at the specified index
        """

        if self._max_size == 0:
            raise IndexError('Index %d out of bounds' % index)

        index %= self._max_size
        return self._buffer['actions'][index]

    def get_reward(self, index: int) -> float:
        """Return the specified reward

        Args:
            index -- Reward's index

        Returns:
            the rewards at the specified index
        """
        if self._max_size == 0:
            raise IndexError('Index %d out of bounds' % index)

        index %= self._max_size
        return self._buffer['rewards'][index]

    def get_terminal(self, index: int) -> int:
        """Return the specified terminal flag

        Args:
            index -- Flag's index

        Returns:
            the terminal state at the index
        """
        if self._max_size == 0:
            raise IndexError('Index %d out of bounds' % index)

        index %= self._max_size
        return self._buffer['terminals'][index]

    def minibatch(self, prng, count):
        indices = self._sampler.sample(prng, count, 0, self._count - 1, replace=False)
        values = self._buffer[indices]

        actions, rewards, terminals = values['actions'], values['rewards'], values['terminals']
        # NOTE this is surprisingly expensive if TemporalMemory is used under the hood
        # should either find a way to make it faster or have a different implementation in
        # Temporal memory to reflect its added sampling cost.
        pre_states = np.array([self.get_state(i) for i in indices])
        post_states = np.array([self.get_state(i + 1) for i in indices])

        return Batch(pre_states, actions, rewards, terminals, post_states, indices)

    def get_probabilities(self, indices):
        return self._sampler.get_probabilities(indices, 0, self._count)

    def get_minibatch_probability(self, batch):
        """Returns the probabilities for the samples in a batch.

        Args:
            batch -- a batch produced by this memory

        Returns a list of probabilities per sample in the batch
        """
        return self._sampler.get_probabilities(batch.indices, 0, self._count)

    def set_priorities(self, indices, priorities):
        """Set the new priorities to data in the buffer.

        Description:
            Gives updated priorities to values in the memory. Should be called before
            altering the buffer using append

        Args:
            indices -- the indices to update
            priorities -- 1D numpy array or list of numbers
        """
        epsilon = ReplayMemory.EPSILON_DEMO if self._is_demo else ReplayMemory.EPSILON
        priorities = (np.abs(priorities) + epsilon) ** ReplayMemory.ALPHA
        for index, priority in zip(indices, priorities):
            try:
                self._sampler.set_priority(index, priority)
            except AttributeError:
                break

    def save(self, file):
        self._logger.info("%s: save to %s", self, file)
        np.savez_compressed(file,
                            count=self._count,
                            states=self._buffer['states'][:self._count],
                            actions=self._buffer['actions'][:self._count],
                            rewards=self._buffer['rewards'][:self._count],
                            terminals=self._buffer['terminals'][:self._count])

    def load(self, file):
        self._logger.info("%s: load from %s", self, file)
        info = np.load(file)
        states = info['states']
        actions = info['actions']
        rewards = info['rewards']
        terminals = info['terminals']
        assert self.observation_space.contains(states[0])
        assert self.action_space.contains(actions[0])
        num_saved = info['count']
        if num_saved > self._max_size:
            fmt = ("Not enough room in memory to load all trajectories"
                   " (only loading %d of %d total)")
            self._logger.warning(fmt, self._max_size, num_saved)
            num_saved = self._max_size

        self._buffer[0:num_saved]['states'] = states
        self._buffer[0:num_saved]['actions'] = actions
        self._buffer[0:num_saved]['rewards'] = rewards
        self._buffer[0:num_saved]['terminals'] = terminals
        self._count = num_saved

    def __repr__(self):
        representation = '{}(max_size={}, obs_space={}, act_space={}, sampler={})'
        return representation.format(self.__class__.__name__, self._max_size,
                                     self._observation_space, self._action_space, self._sampler)
