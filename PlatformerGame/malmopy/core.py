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

"""Module containing the shared, basic core functionality"""

from abc import ABC, abstractmethod
from enum import IntEnum
import logging
import os
import csv
import json
from collections import deque

import gym
import numpy as np

from .abc import FeedForwardModel, Visualizable, Explorable
from .utils import np_random, Graph, Node


class QType(IntEnum):
    """Different types of Q networks"""

    DQN = 0
    DoubleDQN = 1


class QFunctionApproximator(ABC):
    """Abstract base class for classes which approximate the Q function."""

    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space -- The space of input observations
            action_space -- The space of output actions
        """
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)

        self._observation_space = observation_space
        self._action_space = action_space
        self.seed()

    def seed(self, seed=None):
        """Seeds the pseudo-random number generator of the approximator.

        Args:
            seed -- the value used to seed the PRNG

        Returns the seed
        """
        self.np_random, seed = np_random(seed)
        return seed

    @property
    def observation_space(self):
        """The space of observations upon which the agent can act."""
        return self._observation_space

    @property
    def action_space(self):
        """The space to which actions returned by the agent will belong"""
        return self._action_space

    @abstractmethod
    def present_batch(self, memory, minibatch_size):
        """Present a batch to the model.

        Args:
            memory -- a memory object
            minibatch_size -- the size of minibatch to sample
        """

    @abstractmethod
    def train_model(self):
        """Train the model."""

    @abstractmethod
    def update_target(self):
        """Update the target."""

    @abstractmethod
    def compute(self, observations, is_training):
        """Compute the approximate q-values for all actions given the provided observations.

        Args:
            observations -- the observations
            is_training -- whether the system is in training (as opposed to evaluation)

        Returns the q values for each observation for every action in the action space
        """

    @abstractmethod
    def save(self, path):
        """Save the function to the specified path

        Args:
            path -- path to a location on the disk
        """

    @abstractmethod
    def load(self, path):
        """Load the function from the specified path

        Args:
            path -- path to a location on the disk
        """

class TrajectoryLearner(ABC):
    """Abstract base class for classes which learn a policy from trajectories."""

    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space -- The space of input observations
            action_space -- The space of output actions
        """
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)

        self._observation_space = observation_space
        self._action_space = action_space
        self.seed()

    def seed(self, seed=None):
        """Seeds the pseudo-random number generator of the approximator.

        Args:
            seed -- the value used to seed the PRNG

        Returns the seed
        """
        self.np_random, seed = np_random(seed)
        return seed

    @property
    def observation_space(self):
        """The space of observations upon which the agent can act."""
        return self._observation_space

    @property
    def action_space(self):
        """The space to which actions returned by the agent will belong"""
        return self._action_space

    @abstractmethod
    def train_on_policy(self, batch, weights):
        """Trains the learner on-policy using the provided trajectories.

        Args:
            batch -- a TrajectoryBatch of trajectories from the current policy
            weights -- per-trajectory weights
        """

    @abstractmethod
    def train_off_policy(self, batch, weights):
        """Trains the learner off-policy using the provided batch of trajectories.

        Args:
            batch -- a TrajectoryBatch of previous trajectories
            weights -- per-trajectory weights
        """

    @abstractmethod
    def select_action(self, observation):
        """Selects an action for the given observation.

        Args:
            observation -- the current observation from the environment

        Returns an action from the action space according to the current policy
        """

    @abstractmethod
    def save(self, path):
        """Save the learner to the specified path

        Args:
            path -- path to a location on the disk
        """

    @abstractmethod
    def load(self, path):
        """Load the learner from the specified path

        Args:
            path -- path to a location on the disk
        """


class FeedForwardModelQueue(object):
    """
    Class which encapsulates a queue of models and manages processing data through all of them.
    """

    def __init__(self, num_targets, reduce=None):
        """
        Args:
            num_targets -- the maximum number of targets to store

        Keyword Args:
            reduce -- the reduction function to use on model outputs [None]
        """
        self._models = deque(maxlen=num_targets)
        self._reduce = reduce

    def enqueue(self, model):
        """Enqueue a new model.

        Args:
            model -- the model to enqueue.
        """
        assert isinstance(model, FeedForwardModel)
        self._models.append(model)

    def compute(self, inputs):
        """Computes the output values for the inputs.

        Args:
            inputs -- the current inputs

        Returns the result of compute the outputs from each model with an optional reduction
        """
        if self._reduce:
            outputs = self._reduce([model(inputs) for model in self._models])
        else:
            outputs = self._models[-1](inputs)

        return outputs

    def __call__(self, inputs):
        return self.compute(inputs)


class FileRecorder(object):
    """Saves key-values pairs into a local file as a CSV."""

    def __init__(self, output_path, keys=None):
        """
        Args:
            output_path -- the path to the output file

        Keyword Args:
            keys -- a dictionary of metadata keys written to the top of the file [{}]

        """
        self._logger = logging.getLogger(__name__)
        self._logger.info("Opening file %s", output_path)
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        self._file = open(output_path, 'w', newline='')
        if not keys:
            self._keys = {}
        else:
            self._keys = keys

        self._count = 0
        self._csv = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

    def init(self, columns):
        """Indicate the labels for the values which the recorder will write to file.

        Args:
            columns -- a list of column names
        """
        self._keys["columns"] = columns
        self._file.write('key\n%s\n' % json.dumps(self._keys))
        self._csv = csv.DictWriter(self._file, columns)
        self._csv.writeheader()

    def record(self, values):
        """Save values to the file and increment the counter.

        Args:
            values -- tuple of items that can be converted to string using str
                      these should be in the same order as the columns passed
                      to init()
        """
        assert self._csv
        content = dict(zip(self._keys["columns"], map(str, values)))
        self._csv.writerow(content)
        self._count += 1


class AgentMode(IntEnum):
    """Enumeration of different agent modes."""
    Warmup = 0
    Training = 1
    Evaluation = 2


class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space -- The space of observations upon which the agent can act
            action_space -- The space to which actions returned by the agent will belong
        """
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)

        self._observation_space = observation_space
        self._action_space = action_space
        self._mode = AgentMode.Warmup
        self.seed()

    def seed(self, seed=None):
        """Seed the pseudo-random number generator for the agent.

        Keyword Args:
            seed -- the seed to use. [None]

        Returns the seed used (i.e. if None is passed one will be generated)
        """
        self.np_random, seed = np_random(seed)
        return seed

    @property
    def mode(self):
        """The current mode of the Agent (see `AgentMode`)."""
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def observation_space(self):
        """The space of observations upon which the agent can act."""
        return self._observation_space

    @property
    def action_space(self):
        """The space to which actions returned by the agent will belong"""
        return self._action_space

    @property
    def unwrapped(self):
        """Completely unwrap this agent.

        Returns the base non-wrapped Agent instance
        """
        return self

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    @abstractmethod
    def act(self, observation):
        """Determine an action based upon the provided observation.

        Args:
            observation -- a sample from the supported observation space

        Returns a sample from the action space
        """

    @abstractmethod
    def observe(self, pre_observation, action, reward, post_observation, done):
        """Lets the agent observe a sequence of observation => action => observation, reward.

        Args:
            pre_observation -- the observation before the action was taken
            action -- the action which was taken
            reward -- the reward which was given
            post_observation -- the observation after the action was taken
            done -- whether the environment was in a terminal state after the action was taken
        """

    @abstractmethod
    def save(self, path):
        """Save the agent to the specified path

        Args:
            path -- path to a location on the disk
        """

    @abstractmethod
    def load(self, path):
        """Load the agent from the specified path

        Args:
            path -- path to a location on the disk
        """


class AgentWrapper(Agent):
    """
    Class which wraps an existing agent, allowing it to intercept the act and observe methods
    as needed to add additional behavior.
    """

    def __init__(self, agent):
        assert isinstance(agent, Agent)
        super(AgentWrapper, self).__init__(agent.observation_space, agent.action_space)
        self._agent = agent

    @property
    def mode(self):
        return self._agent.mode

    @mode.setter
    def mode(self, value):
        self._agent.mode = value

    @property
    def agent(self):
        """The wrapped agent."""
        return self._agent

    @property
    def unwrapped(self):
        return self._agent.unwrapped

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self._agent)

    def __repr__(self):
        return str(self)

    @abstractmethod
    def act(self, observation):
        pass

    @abstractmethod
    def observe(self, pre_observation, action, reward, post_observation, done):
        pass

    def save(self, path):
        self._agent.save(path)

    def load(self, path):
        self._agent.load(path)


class ActWrapper(AgentWrapper):
    """Agent wrapper which only wraps the act method."""

    @abstractmethod
    def act(self, observation):
        pass

    def observe(self, pre_observation, action, reward, post_observation, done):
        self.agent.observe(pre_observation, action, reward, post_observation, done)


class ObserveWrapper(AgentWrapper):
    """Agent wrapper which only wraps the observe method."""

    def act(self, observation):
        return self.agent.act(observation)

    @abstractmethod
    def observe(self, pre_observation, action, reward, post_observation, done):
        pass


class VisualizableWrapper(gym.Wrapper, Visualizable):
    """Class which exposes the metrics members of all wrapped environments."""

    def __init__(self, env):
        """
        Args:
            env -- an environment which implements Visualizable at some level
                   of wrapping.
        """
        super(VisualizableWrapper, self).__init__(env)
        self._visualizable = []
        # recursively unwrap the environment, gathering any metrics as we go
        while isinstance(env, gym.Wrapper):
            if isinstance(env, Visualizable):
                self._visualizable.append(env)

            env = env.env

        if isinstance(env, Visualizable):
            self._visualizable.append(env)

    @property
    def metrics(self):
        result = []
        for env in self._visualizable:
            result.extend(env.metrics)

        return result

    def reset(self, **kwargs): #pylint: disable=E0202
        return self.env.reset()

    def step(self, action): #pylint: disable=E0202
        return self.env.step(action)


class VisualizableAgentWrapper(AgentWrapper, Visualizable):
    """Class which exposes the metrics members of all wrapped agents"""

    def __init__(self, agent):
        super(VisualizableAgentWrapper, self).__init__(agent)
        self._visualizable = []
        # recursively unwrap the agent, gathering any metrics as we go
        while isinstance(agent, AgentWrapper):
            if isinstance(agent, Visualizable):
                self._visualizable.append(agent)

            agent = agent.agent

        if isinstance(agent, Visualizable):
            self._visualizable.append(agent)

    @property
    def metrics(self):
        result = []
        for agent in self._visualizable:
            result.extend(agent.metrics)

        return result

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, pre_observation, action, reward, post_observation, done):
        self.agent.observe(pre_observation, action, reward, post_observation, done)


class ExplorableGraph(Graph):
    """Graph wrapping for an Explorable environment."""

    class Node(Node):
        """Node class for use by ExplorableGraph"""

        def __init__(self, action, observation, key):
            self.action = action
            self.observation = observation
            self._key = key

        def matches(self, other):
            return self.key == other.key

        @property
        def key(self):
            return self._key

        def __repr__(self):
            return "{}=>{}".format(self.action, self.observation)

    def __init__(self, env):
        assert isinstance(env.unwrapped, Explorable)
        self._env = env.unwrapped

    def neighbors(self, node):
        neighbors = []
        pre_observation = node.observation
        for action in self._env.available_actions(pre_observation):
            post_observation, _, _ = self._env.try_step(pre_observation, action)
            dist = self._env.distance(post_observation, pre_observation)
            if not np.isclose(dist, 0.0):
                neighbors.append(ExplorableGraph.Node(action,
                                                      post_observation,
                                                      self._env.hash(post_observation)))

        return neighbors

    def distance(self, node0, node1):
        return self._env.distance(node0.observation, node1.observation)

    @property
    def nodes(self):
        raise NotImplementedError
