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

"""Module containing classes to facilitate temporal (i.e. n-step) learning"""

import numpy as np

from .batches import NStepBatch
from .replay import ReplayMemory
from .. import History


class TemporalMemory(ReplayMemory):
    """
    Temporal memories adds a new dimension to store N previous samples (t, t-1, t-2, ..., t-N)
    when sampling from memories
    """

    def __init__(self, observation_space, action_space, unflicker=False,
                 sampler=None, is_demo=False, max_size=None, buffer=None):
        #pylint: disable=too-many-arguments
        """
        Args:
            observation_space -- the space of the stored observations
            action_space -- the space of the stored actions

        Keyword Args:
            history_length -- Length of the visual memories (n previous frames)
                              included with each state [4]
            unflicker -- Indicate if we need to compute the difference between
                         consecutive frames [False]
            max_size -- the maximum size of the replay memory
            sampler -- the sampler to use when creating minibatches
            is_demo -- whether this is a demonstration memory
            buffer -- the existing buffer to use
        """
        assert isinstance(observation_space, History)
        super(TemporalMemory, self).__init__(observation_space.inner, action_space,
                                             sampler, is_demo, max_size, buffer)

        self._unflicker = unflicker
        self._history_length = observation_space.length
        self._last_observation = np.zeros(observation_space.inner.shape)

    @property
    def unflicker(self):
        """
        Indicate if samples added to the replay memories are preprocessed
        by taking the maximum between current frame and previous one
        """
        return self._unflicker

    @property
    def history_length(self):
        """
        Visual memories length (ie. number of previous frames included for each sample)
        """
        return self._history_length

    def append(self, observation, action, reward, done, priority=None):
        assert observation.shape[0] == self._history_length
        observation = observation[-1]
        if self._unflicker:
            max_diff_buffer = np.maximum(self._last_observation, observation)
            self._last_observation = observation
            observation = max_diff_buffer

        super(TemporalMemory, self).append(observation, action, reward, done, priority)

        if done:
            if self._unflicker:
                self._last_observation.fill(0)

    def get_state(self, index: int):
        """Return the specified state with the visual memories

        Args:
            index -- State's index

        Return:
            Tensor[history_length, input_shape...]
        """
        if self.size == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            state = np.ascontiguousarray(
                self._buffer[(index - (history_length - 1)):index + 1, ...]['states'])
        else:
            indices = np.arange(index - self._history_length + 1, index + 1)
            state = np.ascontiguousarray(
                self._buffer.take(indices, mode='wrap')['states'])

        # Set to zero all frames the precede a terminal frame if that terminal frame is not the last
        # frame in the state, i.e. belongs to a previous episode.
        # The buffer stays unchanged given the behavior of np.ascontiguousarray.
        for i in range(history_length - 1):
            if self._buffer[index - self._history_length + 1 + i]['terminals'] == 1:
                state[0:i + 1, ...] = 0
        return state

    def get_discount_reward(self, index, nstep=1, gamma=0.99):
        """Output N-step discount reward

        Args:
            index -- the index for the discounted reward
            nstep -- the number of steps into the future
            gamma -- the discount factor

        Returns:
            discount N-step reward
        """
        if self.size == 0:
            raise IndexError('Index %d out of bounds' % index)

        index %= self.size
        reward = 0
        for i in range(nstep):
            reward = self._buffer['rewards'][index +
                                             nstep - 1 - i] + gamma * reward
        return reward

    def _sample(self, prng, count, nstep):
        """Performs rejection sampling to avoid sampling within the current episode.

        Args:
            prng -- the pseudo-random number generator
            count -- the number of samples
            nstep -- the number of steps being project into the future

        Returns:
            a list of sampled indices
        """
        window_min = self._pos - nstep + 1
        window_max = self._pos + self._history_length - 1
        total_before = window_min - self._history_length
        total_after = self._count - window_max - nstep
        assert count <= total_before + total_after, "Not enough elements to sample"

        # implementation using rejection sampling
        indices = set([])
        while len(indices) < count:
            samples = self._sampler.sample(prng, count, self._history_length - 1,
                                           self._count - nstep + 1, replace=False)
            for index in samples:
                # this disallows terminals within the sampled episode -
                # creates bias in long episodes and makes learning in
                # short episodes impossible!
                if index < window_min or index >= window_max:
                    indices.add(index)

        assert len(indices) == count
        return list(indices)

    def nstep_minibatch(self, prng, count, nstep=1, gamma=0.99):
        """Computes the minibatch incorporating including future states and rewards.

        Args:
            prng -- the pseudo-random number generator used for sampling
            count -- the number of items in the batch

        Keyword Args:
            nstep -- the number of steps into the future to use [1]
            gamma -- the discount used when calculating future rewards [0.99]

        Returns:
            an NStepBatch object
        """
        indices = self._sample(prng, count, nstep)
        values = self._buffer[indices]

        actions, rewards, terminals = values['actions'], values['rewards'], values['terminals']
        pre_states = np.array([self.get_state(i) for i in indices])
        post_states = np.array([self.get_state(i + 1) for i in indices])
        nstep_rewards = np.array(
            [self.get_discount_reward(i, nstep, gamma) for i in indices])
        nstep_states = np.array(
            [self.get_state(i + nstep) for i in indices])

        return NStepBatch(pre_states, actions, rewards, terminals, post_states,
                          indices, nstep, nstep_states, nstep_rewards)

    def get_minibatch_probability(self, batch):
        if isinstance(batch, NStepBatch):
            return self.get_nstep_minibatch_probability(batch)

        return super(TemporalMemory, self).get_minibatch_probability(batch)

    def get_nstep_minibatch_probability(self, batch: NStepBatch):
        """Gets the sample probability of the members of an n-step batch.

        Args:
            batch -- the batch to use when computing probability
        """
        count, history_len = self._count, self._history_length
        return self._sampler.get_probabilities(batch.indices, history_len - 1,
                                               count - batch.nstep + 1)

    def __repr__(self):
        format_str = ('{}(max_size={}, observation_space={}, action_space={}, history_length={},'
                      ' unflicker={}, sampler={})')
        return format_str.format(self.__class__.__name__, self._max_size, self._observation_space,
                                 self._action_space, self._history_length, self._unflicker,
                                 self._sampler)
