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

"""Module containing simpler agent implementations."""

from argparse import ArgumentParser, Namespace

import gym
import numpy as np

from .. import Agent, Visualizable, ScalarSummary


class ConstantAgent(Agent):
    """
    Class for non-learning agents, such as Nash or random strategies
    Expects to be given a dictionary where keys are states and values
    are probability distributions over actions.  See kuhnpoker_multiagent.py
    for an example.
    """

    def __init__(self, observation_space, action_space, strategies):
        assert observation_space.shape is not None
        assert isinstance(action_space, gym.spaces.Discrete)
        super(ConstantAgent, self).__init__(observation_space, action_space)
        self._strategies = strategies

    def act(self, observation):
        action_probs = self._strategies[tuple(observation)]
        new_action = self.np_random.choice(self.action_space.n, p=action_probs)
        return new_action

    def observe(self, pre_observation, action, reward, post_observation, done):
        pass

    def save(self, path):
        np.savez_compressed(path, strategies=self._strategies)

    def load(self, path):
        loaded = np.load(path)
        self._strategies = loaded['strategies']


class SequenceAgent(Agent):
    """An agent that executes a fixed sequence of actions."""

    def __init__(self, observation_space, action_space, sequence, delay_between_action=0):
        super(SequenceAgent, self).__init__(observation_space, action_space)

        self._sequence = sequence
        self._index = 0
        self._delay = delay_between_action

    def act(self, observation):
        if self._delay > 0:
            from time import sleep
            sleep(self._delay)

        action = self._sequence[self._index]
        self._index += 1
        if self._index == len(self._sequence):
            self._index = 0

        return action

    def observe(self, pre_observation, action, reward, post_observation, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class RandomAgent(Agent):
    """Agent which ignores the observation space and simply samples from the action space."""

    def act(self, observation):
        return self.action_space.sample()

    def observe(self, pre_observation, action, reward, post_observation, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class TabularQAgent(Agent, Visualizable):
    """Agent which uses tabular Q learning."""

    @staticmethod
    def add_args_to_parser(parser):
        """Add arguments for the Tabular Q-learning hyperparameters.

        Description:
            This method adds the following arguments:

            learning_rate -- The rate at which Q values should be updated [0.1]
            discount -- The discount factor for future rewards [0.95]
            resume -- Path to a saved model on the disk to resume [None]
        """
        assert isinstance(parser, ArgumentParser)
        group = parser.add_argument_group("Tabular Q Agent")
        group.add_argument("--learning_rate", type=float, default=0.1,
                           help="The rate at which Q values should be updated")
        group.add_argument("--discount", type=float, default=0.95,
                           help="The discount factor for future rewards")
        group.add_argument("--resume", type=str, default=None,
                           help="Path to a saved model on the disk to resume")

    @staticmethod
    def create(args, observation_space, action_space):
        """Create a Tabular Q Agent from the arguments."""
        assert isinstance(args, Namespace)
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)
        agent = TabularQAgent(observation_space, action_space, args.learning_rate, args.discount)
        if args.resume:
            agent.load(args.resume)

        return agent

    def __init__(self, observation_space, action_space, learning_rate, discount=0.95):
        """
        Args:
            observation_space -- must be of type gym.spaces.Discrete
            action_space -- must be of type gym.spaces.Discrete
            learning_rate -- the rate at which Q values should be updated

        Keyword Args:
            discount -- The discount factor for future rewards [0.95]
        """
        assert isinstance(observation_space, gym.spaces.Discrete)
        assert isinstance(action_space, gym.spaces.Discrete)
        super(TabularQAgent, self).__init__(observation_space, action_space)

        self._learning_rate = learning_rate
        self._discount_factor = discount
        self._qtable = np.zeros((observation_space.n, action_space.n), np.float32)
        self._experience = ScalarSummary("TabularQ/Experience")

    @property
    def qtable(self):
        """The learned Q-table"""
        return self._qtable

    @property
    def metrics(self):
        return [self._experience]

    def act(self, observation):
        q_values = self._qtable[observation]
        max_action = np.max(q_values)
        top_q_indices = np.where(q_values == max_action)[0]
        return self.np_random.choice(top_q_indices).astype(self.action_space.dtype)

    def observe(self, pre_observation, action, reward, post_observation, done):
        qvalue = self._qtable[pre_observation, action]
        if done:
            update = reward - qvalue
        else:
            max_next = np.max(self._qtable[post_observation])
            update = reward + self._discount_factor*max_next - qvalue

        qvalue += self._learning_rate * update
        self._qtable[pre_observation, action] = qvalue
        self._experience.add(np.sum(np.abs(self._qtable)))

    def save(self, path):
        np.savez_compressed(path, qtable=self.qtable)

    def load(self, path):
        loaded = np.load(path)
        self._qtable = loaded['qtable']


class MultiSimultaneousAgent(object):
    """These define base class for all simultaneous multi agents"""

    def __init__(self):
        self._agent_id = 0

    @property
    def multi_agent_id(self):
        """Getter for the (unique) id of the agent"""
        return self._agent_id

    @property
    def name(self):
        """Getter for the (unique) human readable name of this agent"""
        return self.__class__.__name__ + '_' + str(self._agent_id)

    @multi_agent_id.setter
    def multi_agent_id(self, agent_id):
        """Setter for the (unique) id of the agent"""
        self._agent_id = agent_id
