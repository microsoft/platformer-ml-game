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

"""Module containing classes providing episodic memories."""

import numpy as np
import gym

from .. import Memory
from ..samplers import get_sampler, PrioritizedSampler
from .batches import Batch, batch_dtype


class EpisodicMemory(Memory):
    """
    A simple episode-based memory. Adds new experiences until the end of the episode, and
    is then reset.
    """

    def __init__(self, observation_space, action_space, sampler=None):
        """
        Args:
            observation_space -- the space for the stored observations
            action_space -- the space for the stored actions

        Keyword Args:
            sampler -- the sampler to use
        """
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)
        self._observation_space = observation_space
        self._action_space = action_space
        if sampler is None:
            self._sampler = get_sampler()
        else:
            self._sampler = sampler

        self._index = -1
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminals = []

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def index(self):
        """An optional index used to identify the episode"""
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def is_full(self):
        return False

    def __len__(self) -> int:
        """
        Return the number of sequences hold by this buffer
        :return: Int >= 0
        """
        return len(self._states)

    def __getitem__(self, key):
        return self._states[key], self._actions[key], self._rewards[key], self._terminals[key]

    def __iter__(self):
        for step in zip(self._states, self._actions, self._rewards, self._terminals):
            yield step

    @property
    def values(self):
        """Returns all of the recorded values."""
        return Batch(
            np.ascontiguousarray(self._states, dtype='float32'),
            np.ascontiguousarray(self._actions),
            np.ascontiguousarray(self._rewards, dtype='float32'),
            np.ascontiguousarray(self._terminals),
            False
        )

    def minibatch(self, prng, count):
        indices = self._sampler.sample(prng, count, 0, -1, replace=False)
        pre_states = []
        post_states = []
        actions = []
        rewards = []
        terminals = []
        for index in indices:
            pre_states.append(self._states[index])
            post_states.append(self._states[index + 1])
            actions.append(self._actions[index])
            rewards.append(self._rewards[index])
            terminals.append(self._terminals[index])

        return Batch(
            np.ascontiguousarray(pre_states),
            np.ascontiguousarray(actions),
            np.ascontiguousarray(rewards, dtype='float32'),
            np.ascontiguousarray(terminals),
            post_states=post_states,
            indices=np.ascontiguousarray(indices)
        )

    def get_probabilities(self, indices):
        return self._sampler.get_probabilities(indices, 0, len(self))

    def append(self, observation, action, reward, done, priority=None):
        assert self._observation_space.contains(observation)
        assert self._action_space.contains(action)
        self._states.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminals.append(done)
        if isinstance(self._sampler, PrioritizedSampler):
            self._sampler.append_priority(priority)
        else:
            self._sampler.append()

    def reset(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._terminals.clear()
        self._sampler.reset()

    def to_ndarray(self):
        result = np.empty(len(self), batch_dtype(self.observation_space, self.action_space))
        for i, step in enumerate(self):
            result[i] = step

        return result

    def from_ndarray(self, array, sampler):
        assert self.observation_space.contains(array['states'][0])
        assert self.action_space.contains(array['actions'][0])
        self._states = array['states'].tolist()
        self._actions = array['actions'].tolist()
        self._rewards = array['rewards'].tolist()
        self._terminals = array['terminals'].tolist()
        if sampler:
            assert len(sampler) == len(self)
            self._sampler = sampler
        else:
            self._sampler = get_sampler(initial_size=len(self))

        return self

    def save(self, file):
        np.savez_compressed(file, states=self._states, actions=self._actions,
                            rewards=self._rewards, terminals=self._terminals)

    def load(self, file):
        loaded = np.load(file)
        self._states = loaded['states']
        self._actions = loaded['actions']
        self._rewards = loaded['rewards']
        self._terminals = loaded['terminals']
