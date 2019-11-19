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

"""
Module containing classes which provide batches of observations, actions, rewards and terminals.
"""

import numpy as np


def batch_dtype(observation_space, action_space):
    """Returns a batch dtype for the provided observation and action spaces.

    Args:
        observation_space -- the observation space
        action_space -- the action space

    Returns a complex numpy dtype for the batch
    """
    states_dtype = np.dtype((observation_space.dtype, observation_space.shape))
    actions_dtype = np.dtype((action_space.dtype, action_space.shape))
    return np.dtype([('states', states_dtype), ('actions', actions_dtype),
                     ('rewards', 'f4'), ('terminals', 'i1')], align=True)


class Batch(object):
    """Class encapsulating a batch of experience data."""

    def __init__(self, pre_states, actions, rewards, terminals, post_states=None, indices=None):
    #pylint: disable=too-many-arguments
        """
        Args:
            pre_states -- the states that were observed
            actions -- the actions that were taken
            rewards -- the rewards that were given
            terminals -- whether the states were terminal states

        Keyword Args:
            post_states -- the states after the actions were taken
            indices -- the indices of the data (in the memory)
        """
        self._pre_states = pre_states
        self._actions = actions
        self._rewards = rewards
        self._terminals = terminals
        self._post_states = post_states
        self._indices = indices

    def __len__(self):
        """Returns the number of items in the batch."""
        return len(self._pre_states)

    def random_crop(self, count, prng):
        """Randomly crops the batch to the size provided.

        Description:
            Selects a random offset and crops the batch. If `count` is greater than the size
            of the batch, the full batch is returned.

        Args:
            count -- the desired size of the random crop
            prng -- the pseudo-random number generator
        """
        if count is None or count >= len(self):
            return self

        offset = prng.randint(len(self) - count)
        post_states = self._post_states[offset:offset+count] if self._post_states else None
        indices = self._indices[offset:offset+count] if self._indices else None
        return Batch(
            self._pre_states[offset:offset+count],
            self._actions[offset:offset+count],
            self._rewards[offset:offset+count],
            self._terminals[offset:offset+count],
            post_states,
            indices
        )

    @staticmethod
    def to_array_and_type(array, dtype):
        """May convert the provided array to a numpy array of the specified if necessary.

        Args:
            array -- the array to standardize
            dtype -- the type to convert to

        Returns:
            the (if necessary converted) array
        """
        if not isinstance(array, np.ndarray):
            array = np.ascontiguousarray(array, dtype=dtype)
        elif array.dtype != dtype:
            array = array.astype(dtype)

        return array

    def standardize(self):
        """Ensures that all arrays are of the correct underlying types.

        Description:
            Sets all the underlying array types to standard classes and types.
        """
        self._pre_states = Batch.to_array_and_type(self._pre_states, np.float32)
        self._post_states = Batch.to_array_and_type(self._post_states, np.float32)
        self._rewards = Batch.to_array_and_type(self._rewards, np.float32)
        self._terminals = Batch.to_array_and_type(self._terminals, np.float32)
        self._actions = Batch.to_array_and_type(self._actions, np.int8)
        self._indices = Batch.to_array_and_type(self._indices, np.int32)

    def to_gpu_chainer(self, gpu_device):
        """Converts the arrays to Chainer GPU format.

        Args:
            gpu_device -- the GPU device to use
        """
        import chainer.cuda as cuda
        with gpu_device:
            self._pre_states = cuda.to_gpu(self._pre_states)
            self._actions = cuda.to_gpu(self._actions)
            self._post_states = cuda.to_gpu(self._post_states)
            self._rewards = cuda.to_gpu(self._rewards)
            self._terminals = cuda.to_gpu(self._terminals)

    def to_dict_list(self):
        """Converts the batch to a list of dict objects.

        Returns [{state:, action:, reward:, next_state:, next_action:, is_state_terminal:}]
        """
        dicts = []
        for pre_state, action, reward, post_state, is_state_terminal in zip(self._pre_states,
                                                                            self._actions,
                                                                            self._rewards,
                                                                            self._post_states,
                                                                            self._terminals):
            dicts.append({
                "state": pre_state,
                "action": action,
                "reward": reward,
                "next_state": post_state,
                "next_action": action,
                "is_state_terminal": is_state_terminal
            })

        return dicts

    @property
    def states(self):
        """The states that were observed (alias for pre_states)"""
        return self.pre_states

    @property
    def pre_states(self):
        """The states that were observed"""
        return self._pre_states

    @property
    def actions(self):
        """The actions that were taken"""
        return self._actions

    @property
    def rewards(self):
        """The rewards that were given"""
        return self._rewards

    @property
    def terminals(self):
        """Whether the states were terminal states"""
        return self._terminals

    @property
    def post_states(self):
        """The states after the actions were taken"""
        return self._post_states

    @property
    def indices(self):
        """The indices of the batch elements (in the memory)"""
        return self._indices


class NStepBatch(Batch):
    """A batch which also incorporates n-step states and rewards"""

    def __init__(self, pre_states, actions, rewards, terminals,
                 post_states, indices, nstep, nstep_states, nstep_rewards):
        #pylint: disable=too-many-arguments
        """
        Args:
            pre_states -- the states that were observed
            actions -- the actions that were taken
            rewards -- the rewards that were given
            terminals -- whether the states were terminal states
            post_states -- the states after the actions were taken
            indices -- the indices of the data (in the memory)
            nstep -- the number of steps in the future
            nstep_states -- the states after n steps
            nstep_rewards -- the (discounted) rewards after n steps
        """
        super(NStepBatch, self).__init__(pre_states, actions,
                                         rewards, terminals, post_states, indices)
        self._nstep = nstep
        self._nstep_states = nstep_states
        self._nstep_rewards = nstep_rewards

    def random_crop(self, count, prng):
        """Randomly crops the batch to the size provided.

        Description:
            Selects a random offset and crops the batch. If `count` is greater than the size
            of the batch, the full batch is returned.

        Args:
            count -- the desired size of the random crop
        """
        if count > len(self):
            return self

        offset = prng.randrange(len(self) - count)
        return NStepBatch(
            self._pre_states[offset:offset+count],
            self._actions[offset:offset+count],
            self._rewards[offset:offset+count],
            self._terminals[offset:offset+count],
            self._post_states[offset:offset+count],
            self._indices[offset:offset+count],
            self._nstep,
            self._nstep_states[offset:offset+count],
            self._nstep_rewards[offset:offset+count]
        )

    def to_gpu_chainer(self, gpu_device):
        super(NStepBatch, self).to_gpu_chainer(gpu_device)
        import chainer.cuda as cuda
        with gpu_device:
            self._nstep_states = cuda.to_gpu(self._nstep_states)
            self._nstep_rewards = cuda.to_gpu(self._nstep_rewards)

    def standardize(self):
        super(NStepBatch, self).standardize()
        self._nstep_states = Batch.to_array_and_type(self._nstep_states, np.float32)
        self._nstep_rewards = Batch.to_array_and_type(self._nstep_rewards, np.float32)

    @property
    def nstep(self):
        """The number of steps in the future"""
        return self._nstep

    @property
    def nstep_states(self):
        """The states after n steps"""
        return self._nstep_states

    @property
    def nstep_rewards(self):
        """The rewards after n steps"""
        return self._nstep_rewards


class TrajectoryBatch(object):
    """Class encapsulating a batch of EpisodicMemory objects encoding trajectories."""

    def __init__(self, trajectories):
        from .episode import EpisodicMemory
        for trajectory in trajectories:
            assert isinstance(trajectory, EpisodicMemory)

        self._trajectories = list(sorted(trajectories, key=len, reverse=True))
        observation_space = trajectories[0].observation_space
        action_space = trajectories[0].action_space
        self._count = len(self._trajectories)
        self._length = len(self._trajectories[0])
        self._logprobs = []
        self._values = []
        for trajectory in self._trajectories:
            self._logprobs.append([None]*len(trajectory))
            self._values.append([None]*len(trajectory))

        self._states_shape = observation_space.shape
        self._actions_shape = action_space.shape

    def step_batches(self, gpu_device=None):
        """Enumerates over the trajectories, returning optimal size batches for each time step.

        Keyword Args:
            gpu_device -- a cupy gpu device, used to move the batch to the GPU [None]

        Returns (step, batch)
        """
        for step in range(self._length):
            trajectories = []
            for trajectory in self._trajectories:
                if len(trajectory) <= step:
                    break

                trajectories.append(trajectory)

            num_trajectories = len(trajectories)
            pre_states = np.zeros((num_trajectories, ) + self._states_shape, np.float32)
            actions = np.zeros((num_trajectories, ) + self._actions_shape, np.int32)
            rewards = np.zeros(num_trajectories, np.float32)
            terminals = np.zeros(num_trajectories, np.int8)
            for i, trajectory in enumerate(trajectories):
                pre_states[i], actions[i], rewards[i], terminals[i] = trajectory[step]

            batch = Batch(pre_states, actions, rewards, terminals)

            if gpu_device:
                batch.to_gpu_chainer(gpu_device)

            yield step, batch

    def set_logprobs_and_values(self, step, logprobs, values):
        """Sets the log probabilities and values for all trajectories at the provided step.

        Description:
            This method is meant to be used in conjuction with the batches returned by
            `dynamic_batches`, returning the logprobs and values in the same order.

        Args:
            step -- the step in the trajectory
            logprobs -- the log probability of the actions for the trajectories at that step
            values -- the values for the trajectories at that step
        """
        length = len(logprobs)
        assert length == len(values)
        assert len(self._trajectories[length-1]) > step

        for i, logprob in enumerate(logprobs):
            self._logprobs[i][step] = logprob

        for i, value in enumerate(values):
            self._values[i][step] = value

    def __getitem__(self, key):
        """Accessor for the trajectory information.

        Args:
            key -- the trajectory key

        Returns (trajectory, logprobs, values)
        """
        return self._trajectories[key], self._logprobs[key], self._values[key]

    def __len__(self):
        """The number of trajectories in the batch."""
        return self._count

    def __iter__(self):
        """Iterates over the trajectory information."""
        for item in zip(self._trajectories, self._logprobs, self._values):
            yield item

    @property
    def trajectories(self):
        """The trajectories in this batch."""
        return self._trajectories
