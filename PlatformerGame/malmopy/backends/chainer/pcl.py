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

"""Module containing classes to enable using Chainer as a back-end for neural net based agents."""

from argparse import ArgumentParser, Namespace

import gym
import numpy as np
import chainer
from chainer.backends import cuda
from chainer import optimizers
from chainer import Chain, ChainList
import chainer.functions as F

from ... import FeedForwardModel, Visualizable, PolicyAndValueModel, TrajectoryLearner,\
    MeanSummary, each_epoch
from .core import create_optimizer, add_optimizer_args, add_model_args, create_model, ChainerMLP


class ChainerPCLSeparateModel(PolicyAndValueModel, Chain):
    """PCL model which uses separate q and value function approximators."""

    def __init__(self, q_model, value_model, tau=1):
        """
        Args:
            q_model -- a FeedForwardModel which models the Q function
            value_model -- a FeedForwardModel which models the value function

        Keyword Args:
            tau -- the scaling value [1]
        """
        assert isinstance(q_model, FeedForwardModel)
        assert isinstance(q_model, (Chain, ChainList))
        assert isinstance(value_model, FeedForwardModel)
        assert isinstance(value_model, (Chain, ChainList))
        super(ChainerPCLSeparateModel, self).__init__()
        self._tau = tau
        with self.init_scope():
            self.q_model = q_model
            self.value_model = value_model

    @property
    def observation_space(self):
        return self.q_model.input_space

    @property
    def action_space(self):
        return self.q_model.output_space

    def compute(self, observations):
        pi_out = F.log_softmax(self.q_model(observations) / self._tau)
        value_out = self.value_model(observations)
        return pi_out, value_out


class ChainerPCLUnifiedModel(PolicyAndValueModel, Chain):
    """PCL model which uses a single model to approximate the Q and value functions."""

    def __init__(self, q_model, tau=1):
        """
        Args:
            q_model -- a FeedForwardModel which models the Q function

        Keyword Args:
            tau -- the scaling parameter [1]
        """
        assert isinstance(q_model, FeedForwardModel)
        assert isinstance(q_model, (Chain, ChainList))
        super(ChainerPCLUnifiedModel, self).__init__()
        self._tau = tau
        with self.init_scope():
            self.q_model = q_model

    @property
    def observation_space(self):
        return self.q_model.input_space

    @property
    def action_space(self):
        return self.q_model.output_space

    def compute(self, observations):
        q_values = self.q_model(observations)
        values_out = F.expand_dims(self._tau * F.logsumexp(q_values / self._tau, axis=1), axis=1)
        values = F.broadcast_to(values_out, q_values.shape)
        pi_out = (q_values - values) / self._tau
        return pi_out, values_out


def add_pcl_model_args(parser):
    """Add the arguments to create PCL models"""
    add_model_args(parser)
    parser.add_argument("--pcl", type=str, default="unified",
                        choices=["unified", "separate"],
                        help="Whether to use a unified or separate PCL model")


def create_pcl_model(args, observation_space, action_space):
    """Create the PCL model from the arguments"""
    if args.pcl == "unified":
        q_model = create_model(args, observation_space, action_space)
        return ChainerPCLUnifiedModel(q_model, args.tau)

    if args.pcl == "separate":
        q_model = create_model(args, observation_space, action_space)
        value_model = create_model(args, observation_space,
                                   gym.spaces.Box(-1, 1, (1,), np.float32))
        return ChainerPCLSeparateModel(q_model, value_model, args.tau)

    raise NotImplementedError


class ChainerPCLLearner(TrajectoryLearner, Visualizable):
    """Chainer implementation of Path Consistency Learning (https://arxiv.org/abs/1702.08892)"""

    class Parameters(object):
        """Parameters for the PCL learner"""

        def __init__(self, args=None):
            self.policy_loss_coefficient = 1.0
            self.value_loss_coefficient = 0.5
            self.normalize_loss_by_steps = True
            self.device_id = -1
            self.discount = 0.99
            self.tau = 1e-2
            self.optimizer = optimizers.SGD()
            self.rollout_length = 10
            def _default_create_model(obs, act):
                return ChainerPCLUnifiedModel(ChainerMLP(obs, act, (64, 64)), 1e-2)

            self.create_model = _default_create_model

            if args:
                self.policy_loss_coefficient = args.policy_loss_coef
                self.value_loss_coefficient = args.value_loss_coef
                self.device_id = args.device_id
                self.discount = args.discount
                self.tau = args.tau
                self.rollout_length = args.rollout_length
                self.optimizer = create_optimizer(args)
                self.create_model = lambda obs, act: create_pcl_model(args, obs, act)

    @staticmethod
    def add_args_to_parser(parser):
        """Add arguments for the hyperparameters of this class.

        Args:
            parser -- instance of ArgumentParser

        Description:
            This method adds the following arguments:

            policy_loss_coef -- Coefficient for the policy consistency loss [1.0]
            value_loss_coef -- Coefficient for the value consistency loss [0.5]
            device_id -- GPU device on which to run the experiments [-1]
            discount -- Discount factor (gamma) [0.99]
            tau -- entropy regularizer value [1e-2]
            rollout_length -- The rollout length to use in consistency calculations [10]
        """
        assert isinstance(parser, ArgumentParser)
        group = parser.add_argument_group("Chainer PCL")
        add_optimizer_args(group)
        add_pcl_model_args(group)
        group.add_argument("--policy_loss_coef", type=float, default=1.0,
                           help="Coefficient for the policy consistency loss")
        group.add_argument("--value_loss_coef", type=float, default=0.5,
                           help="Coefficient for the value consistency loss")
        group.add_argument("--device_id", type=int, default=-1,
                           help="GPU device on which to run the experiments.")
        group.add_argument("--discount", type=float, default=0.99,
                           help="Discount factor (gamma)")
        group.add_argument("--tau", type=float, default=1e-2,
                           help="entropy regularizer value")
        group.add_argument("--rollout_length", type=int, default=10,
                           help="The rollout length to use in consistency calculations")

    def __init__(self, observation_space, action_space, params):
        """
        Args:
            params -- a ChainerPCLLearner.Parameters or Namespace instance
        """
        if isinstance(params, Namespace):
            params = ChainerPCLLearner.Parameters(params)

        model = params.create_model(observation_space, action_space)

        assert isinstance(model, PolicyAndValueModel)
        assert isinstance(model, (Chain, ChainList))
        super(ChainerPCLLearner, self).__init__(observation_space, action_space)

        self._model = model
        if params.device_id > -1:
            self._gpu_device = cuda.get_device_from_id(params.device_id)
            self._gpu_device.use()
            self._model.to_gpu(self._gpu_device)
        else:
            self._gpu_device = None

        self._xp = self._model.xp
        self._discount = params.discount
        self._policy_loss_coefficient = params.policy_loss_coefficient
        self._value_loss_coefficient = params.value_loss_coefficient
        self._tau = params.tau
        self._rollout_length = params.rollout_length
        self._normalize_loss_by_steps = params.normalize_loss_by_steps
        self._optimizer = params.optimizer
        self._optimizer.setup(self._model)
        self._actions = np.arange(self._model.action_space.n)
        self._loss_summary = MeanSummary("ChainerPCL/Mean Loss", trigger=each_epoch())
        self._on_policy_loss_summary = MeanSummary("ChainerPCL/Mean On Policy Loss",
                                                   trigger=each_epoch())
        self._off_policy_loss_summary = MeanSummary("ChainerPCL/Mean Off Policy Loss",
                                                    trigger=each_epoch())

    @property
    def metrics(self):
        return [self._loss_summary, self._on_policy_loss_summary, self._off_policy_loss_summary]

    @property
    def observation_space(self):
        return self._model.observation_space

    @property
    def action_space(self):
        return self._model.action_space

    def train_on_policy(self, batch, weights):
        self._model.cleargrads()
        loss = self.compute_batch_loss(batch, weights)
        self._loss_summary.add(loss)
        self._on_policy_loss_summary.add(loss)
        self._optimizer.update()

    def train_off_policy(self, batch, weights):
        self._model.cleargrads()
        loss = self.compute_batch_loss(batch, weights)
        self._loss_summary.add(loss)
        self._off_policy_loss_summary.add(loss)
        self._optimizer.update()

    def compute_batch_loss(self, batch, weights):
        """Compute gradients on a list of trajectories.

        Args:
            batch -- a TrajectoryBatch
            weights -- a list of weights for trajectories in the batch

        Returns a loss value
        """

        weights = self._xp.array(weights)
        for step, step_batch in batch.step_batches(self._gpu_device):
            policies, values = self._model(step_batch.states)
            values *= (1 - step_batch.terminals).reshape(values.shape)
            logprobs = F.select_item(policies, step_batch.actions)
            batch.set_logprobs_and_values(step, logprobs, values)

        losses = []
        for trajectory, logprobs, values in batch:
            losses.append(self.compute_trajectory_loss(trajectory, logprobs, values))

        losses = F.stack(losses)
        loss = F.average(losses * weights)
        loss.backward()

        return np.asscalar(cuda.to_cpu(loss.data))

    def _rollout_consistency(self, log_probs, rewards, first_value, last_value, length):
        log_probs = F.stack(log_probs)
        weights = (self._discount ** self._xp.arange(length)).astype(self._xp.float32)
        discounted_sum_rewards = self._xp.sum(weights*rewards, keepdims=True)
        discounted_sum_log_probs = F.sum(weights*log_probs, keepdims=True)
        last_value *= self._discount ** length

        value_rewards = last_value - first_value + discounted_sum_rewards

        # ensure pi backprop only goes through log_probs
        policy_consistency = (value_rewards.data -
                              self._tau * discounted_sum_log_probs)

        # ensure values backprop only goes through values
        value_consistency = (value_rewards -
                             self._tau * discounted_sum_log_probs.data)

        return policy_consistency, value_consistency

    def compute_trajectory_loss(self, trajectory, logprobs, values):
        """Compute the loss for a single trajectory.

        Args:
            trajectory -- the trajectory
            logprobs -- the log probabilities of the actions taken
            values -- the computed per-state values

        Returns the per-trajectory losses
        """

        rewards = self._xp.array(trajectory.values.rewards)
        policy_losses = []
        value_losses = []
        steps = len(trajectory) - 1
        for start in range(0, steps):
            end = min(start + self._rollout_length, steps)
            policy_cons, value_cons = self._rollout_consistency(logprobs[start:end],
                                                                rewards[start:end],
                                                                values[start],
                                                                values[end],
                                                                end - start)

            policy_losses.append(policy_cons ** 2)
            value_losses.append(value_cons ** 2)

        policy_loss = F.sum(F.stack(policy_losses)) / 2
        value_loss = F.sum(F.stack(value_losses)) / 2

        policy_loss /= self._tau

        policy_loss *= self._policy_loss_coefficient
        value_loss *= self._value_loss_coefficient

        if self._normalize_loss_by_steps:
            policy_loss /= steps
            value_loss /= steps

        return policy_loss + value_loss

    def prepare_observation(self, observation):
        """Prepares a single observation, moving it to the GPU and batching it if needed.

        Args:
            observation -- a single observation

        Returns a batched observation that (if needed) is on the GPU
        """
        if observation.ndim == len(self.observation_space.shape):
            observation = np.expand_dims(observation, 0)

        if self._gpu_device:
            observation = cuda.to_gpu(observation, self._gpu_device)

        return observation

    def select_action(self, observation):
        observation = self.prepare_observation(observation)

        policy, _ = self._model(observation)
        policy = cuda.to_cpu(F.exp(policy).data[0])
        return np.random.choice(self._actions, 1, p=policy)[0]

    def save(self, path):
        chainer.serializers.save_npz(path, self._model)

    def load(self, path):
        chainer.serializers.load_npz(path, self._model)
