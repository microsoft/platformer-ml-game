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

"""Module containing classes pertaining to a Q Learner agent."""

from argparse import ArgumentParser, Namespace

import numpy as np

from .. import Agent, AgentMode, QFunctionApproximator, Memory, Visualizable, ScalarSummary,\
    each_episode
from ..memories import ReplayMemory


class QLearnerAgent(Agent, Visualizable):
    """
    Class which implements a learning agent that uses a learned Q function approximator to
    determine its actions.
    """

    class Parameters(object):
        """Hyperparameters for training Q Learning agents."""

        def __init__(self, args=None):
            self.train_frequency = 1
            self.target_update_frequency = 50
            self.minibatch_size = 32
            self.resume = None
            self.create_memory = lambda obs, act: ReplayMemory(obs, act, max_size=1000)

            if args:
                self.train_frequency = args.train_frequency
                self.target_update_frequency = args.target_update_frequency
                self.minibatch_size = args.minibatch_size
                self.resume = args.resume
                self.create_memory = lambda obs, act: Memory.create(args, obs, act)

    @staticmethod
    def add_args_to_parser(parser):
        """Adds the arguments for the hyperparameters to the argument parser.

        Args:
            parser -- an instance of ArgumentParser

        Description:
            This method adds the following arguments:

            minibatch_size -- the number of items in a minibatch [32]
            train_frequency -- How often (in steps) training should occur [1]
            target_update_frequency -- How often (in steps) the targets should be updated [50]
            resume -- A path to a model to use to resume the model [None]
        """
        assert isinstance(parser, ArgumentParser)
        group = parser.add_argument_group("Q-Learner")
        Memory.add_args_to_parser(group)
        group.add_argument('--minibatch_size', type=int, default=32,
                           help='Size of the minibatch.')
        group.add_argument('--train_frequency', type=int, default=1,
                           help='How often (in steps) training should occur')
        group.add_argument('--target_update_frequency', type=int, default=50,
                           help='How often (in steps) the target should be updated')
        group.add_argument("--resume", type=str, default=None,
                           help="Path to a checkpoint to resume from")


    def __init__(self, qfunction, params):
        """
        Args:
            qfunction -- an instance of QFunctionApproximator
            params -- an instance of QLearnerAgent.Parameters or Namespace
        """
        if isinstance(params, Namespace):
            params = QLearnerAgent.Parameters(params)

        assert isinstance(params, QLearnerAgent.Parameters)
        assert isinstance(qfunction, QFunctionApproximator)
        super(QLearnerAgent, self).__init__(qfunction.observation_space, qfunction.action_space)
        self.memory = params.create_memory(qfunction.observation_space, qfunction.action_space)
        assert isinstance(self.memory, Memory)
        self.qfunction = qfunction
        self._train_frequency = params.train_frequency
        self._target_update_frequency = params.target_update_frequency
        self.minibatch_size = params.minibatch_size
        self.steps = 0
        if params.resume:
            self.load(params.resume)

    @property
    def metrics(self):
        result = []
        if isinstance(self.qfunction, Visualizable):
            result.extend(self.qfunction.metrics)

        if isinstance(self.memory, Visualizable):
            result.extend(self.memory.metrics)

        return result

    def act(self, observation):
        qvalues = self.qfunction.compute(observation, self.mode == AgentMode.Training)
        return np.argmax(qvalues)

    def observe(self, pre_observation, action, reward, post_observation, done):
        self.memory.append(pre_observation, action, reward, done)
        if self.mode == AgentMode.Training:
            self.steps += 1
            if self.steps % self._train_frequency == 0:
                self.learn()

            if self.steps % self._target_update_frequency == 0:
                self.qfunction.update_target()

    def learn(self):
        """Update the function approximator using the data in the memory."""
        self.qfunction.present_batch(self.memory, self.minibatch_size)
        self.qfunction.train_model()

    def save(self, path):
        self.qfunction.save(path)

    def load(self, path):
        self.qfunction.load(path)


class ImitationQLearnerAgent(QLearnerAgent):
    """
    Base class for imitation Q-learner. The main difference from
    QLearnerAgent is the access to a demonstration memory. A
    demonstration_ratio scheduler is added to represent the importance
    of demonstration set along the training process.
    """

    class Parameters(QLearnerAgent.Parameters):
        """Hyperparameters for training an ImitationQLearner agent"""

        def __init__(self, args=None):
            super(ImitationQLearnerAgent.Parameters, self).__init__(args)
            self.demonstration_ratio = lambda step: 0.0

            if args:
                start, end = args.demo_ratio_range
                if start != end:
                    dur = args.demo_ratio_duration
                    demo_step = (end - start) / dur
                    def _linear_demo_ratio(step):
                        if step < dur:
                            return start + step*demo_step

                        return end

                    self.demonstration_ratio = _linear_demo_ratio
                else:
                    self.demonstration_ratio = lambda step: start

    @staticmethod
    def add_args_to_parser(parser):
        """Adds arguments for the hyperparameters to the parser.

        Description:
            This method adds the following arguments:

            demo_ratio_range -- [start, end] the range of the linear interpolation [[1.0, 0.1]]
            demo_ratio_duration -- the length of the linear interpolation in steps [1000]
        """
        QLearnerAgent.add_args_to_parser(parser)
        group = parser.add_argument_group("Imitation Q-Learner")
        group.add_argument("--demo_ratio_range", type=float, nargs=2, default=[1.0, 0.1],
                           help="the starting value for the demonstration ratio")
        group.add_argument("--demo_ratio_duration", type=int, default=1000,
                           help="the duration of the linear transform (in steps)")

    def __init__(self, qfunction, demonstrations, params):
        """
        Args:
            qfunction -- an instance of QFunctionApproximator
            demonstrations -- an instance of Memory containing demonstrations (can be None)
            params -- an instance of ImitationQLearnerAgent.Parameters or Namespace
        """
        if isinstance(params, Namespace):
            params = ImitationQLearnerAgent.Parameters(params)

        assert isinstance(params, ImitationQLearnerAgent.Parameters)
        if demonstrations:
            assert demonstrations.is_demo

        super(ImitationQLearnerAgent, self).__init__(qfunction, params)
        self._demonstrations = demonstrations
        if self._demonstrations:
            self._demonstration_ratio = params.demonstration_ratio
        else:
            self._demonstration_ratio = lambda x: 0

        self._demonstration_ratio_summary = ScalarSummary("Imitation/Demonstration_Ratio",
                                                          trigger=each_episode())

    @property
    def metrics(self):
        result = [self._demonstration_ratio_summary]
        result.extend(super(ImitationQLearnerAgent, self).metrics)
        return result

    def learn(self):
        demonstration_ratio = self._demonstration_ratio(self.steps)
        self._demonstration_ratio_summary.add(demonstration_ratio)
        demo_size = int(self.minibatch_size * demonstration_ratio)
        replay_size = self.minibatch_size - demo_size

        if replay_size > 0:
            self.qfunction.present_batch(self.memory, replay_size)
        if demo_size > 0:
            self.qfunction.present_batch(self._demonstrations, demo_size)

        self.qfunction.train_model()
