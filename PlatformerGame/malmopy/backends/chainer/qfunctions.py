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

import copy
from argparse import ArgumentParser, Namespace

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import optimizers
from chainer import Chain, ChainList
import chainer.functions as F

from ... import QFunctionApproximator, FeedForwardModel, Visualizable, QType, ScalarSummary,\
    FeedForwardModelQueue, each_episode
from .core import ChainerMLP, add_model_args, add_optimizer_args, create_optimizer, create_model


class ChainerTemporalDifferenceLoss(Visualizable):
    """Computes temporal difference loss."""

    def __init__(self, gamma, nstep, weight, label, qtype=QType.DQN):
        assert isinstance(qtype, QType)
        self._qtype = qtype
        self._discount = gamma
        self._nstep = nstep
        self._weight = weight
        tag = "Loss/N_Step_{}" if nstep > 0 else "Loss/TD_{}"
        self._loss_summary = ScalarSummary(tag.format(label), trigger=each_episode())

    @property
    def metrics(self):
        return [self._loss_summary]

    @property
    def weight(self):
        """The weight for this loss"""
        return self._weight

    def _dqn(self, targets, terminals, post_states, rewards):
        return F.reshape(
            (self._discount ** self._nstep) * (1 - terminals) *
            F.max(targets(post_states), axis=1) + rewards,
            (-1, 1)
        ).data

    def _double_dqn(self, model, targets, terminals, post_states, rewards):
        max_a = F.argmax(model(post_states), axis=1)
        q_targets = F.select_item(targets(post_states), max_a)
        result = F.reshape(((self._discount ** self._nstep) * (1 - terminals)) *
                           q_targets + rewards, (-1, 1))
        return result.data

    def __call__(self, model, target, batch, weights=None):
        if self._nstep > 1:
            post_states = batch.nstep_states
            rewards = batch.nstep_rewards
        else:
            post_states = batch.post_states
            rewards = batch.rewards

        if self._qtype == QType.DQN:
            q_targets = self._dqn(target, batch.terminals, post_states, rewards)
        elif self._qtype == QType.DoubleDQN:
            q_targets = self._double_dqn(model, target, batch.terminals, post_states, rewards)
        else:
            raise NotImplementedError

        q_subset = F.reshape(F.select_item(model(batch.pre_states), batch.actions), (-1, 1))
        loss = F.huber_loss(q_subset, q_targets, 1.0)
        self._loss_summary.add(np.asscalar(cuda.to_cpu(F.average(loss).data)))
        if weights is not None:
            loss *= weights

        return loss


class ChainerLargeMarginLoss(Visualizable):
    """Implementation of large margin loss"""

    def __init__(self, margin_size, weight, label):
        self._margin_size = margin_size
        self._weight = weight
        self._loss_summary = ScalarSummary("Loss/Large_Margin_{}".format(label),
                                           trigger=each_episode())

    @property
    def weight(self):
        """The weight for this loss"""
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def metrics(self):
        return [self._loss_summary]

    def __call__(self, model, _, batch, weights=None):
        q_values = model(batch.pre_states)
        q_subset = F.reshape(F.select_item(q_values, batch.actions), (-1, 1))
        margin = model.xp.ones(q_values.data.shape, dtype=model.xp.float32) * self._margin_size
        margin[model.xp.arange(margin.shape[0]), batch.actions] = 0
        loss = F.reshape(F.max(q_values + margin, axis=1), (-1, 1)) - q_subset
        self._loss_summary.add(np.asscalar(cuda.to_cpu(F.average(loss).data)))
        if weights is not None:
            # Chainer raises a broadcast error if loss and weight aren't the same shape
            loss = F.squeeze(loss)
            loss *= weights

        return loss


class ChainerQFunction(QFunctionApproximator, Visualizable):
    """
    Class that encapsulates a Q-Function approximator which is implemented using a Chainer
    neural net.
    """

    class Parameters(object):
        """Hyperparameters for training Chainer Q Functions"""

        def __init__(self, args=None):
            self.device_id = -1
            self.optimizer = optimizers.SGD()
            self.nstep = 1
            self.nstep_discount = 0.99
            self.discount = 0.99
            self.lambda_nstep = 1.0
            self.qtype = QType.DQN
            self.create_model = lambda obs, act: ChainerMLP(obs, act, (64, 64))

            if args:
                self.device_id = args.device_id
                self.discount = args.discount
                self.nstep = args.nstep
                self.nstep_discount = args.nstep_discount
                self.lambda_nstep = args.lambda_nstep
                self.optimizer = create_optimizer(args)
                self.create_model = lambda obs, act: create_model(args, obs, act)

                if args.qtype == "dqn":
                    self.qtype = QType.DQN
                elif args.qtype == "ddqn":
                    self.qtype = QType.DoubleDQN
                else:
                    raise NotImplementedError

    @staticmethod
    def add_args_to_parser(parser):
        """Adds arguments for the hyperparameters to the argument parser.

        Args:
            parser -- an instance of ArgumentParser

        Description:
            This method adds the following arguments:

            device_id -- GPU device on which to run the experiments [-1]
            nstep -- A value larger than 1 indicates to use nstep return. [1]
            nstep_discount -- Discount factor used in nstep return. [0.99]
            lambda_nstep -- Coefficient used for nstep TD loss. [1.0]
            discount -- Discount factor (gamma) [0.99]
            qtype -- The type of Q network to use ["dqn"]
        """
        assert isinstance(parser, ArgumentParser)
        group = parser.add_argument_group("Chainer Q-Function")
        add_optimizer_args(group)
        add_model_args(group)
        group.add_argument('-d', '--device_id', type=int, default=-1,
                           help='GPU device on which to run the experiments.')
        group.add_argument('--nstep', type=int, default=1,
                           help='A value larger than 1 indicates to use nstep return.')
        group.add_argument('--nstep_discount', type=float, default=0.99,
                           help='Discount factor used in nstep return.')
        group.add_argument('--lambda_nstep', type=float, default=1.0,
                           help='Coefficient used for nstep TD loss.')
        group.add_argument('--discount', type=float, default=0.99,
                           help='Discount factor (gamma).')
        group.add_argument('--qtype', type=str, default="dqn",
                           choices=['dqn', 'ddqn'], help="The type of Q network to use")

    def __init__(self, observation_space, action_space, params):
        """
        Args:
            observation_space -- the space that observations will be drawn from
            action_space -- the space of actions this agent should produce
            params -- an instance of ChainerQFunction.Parameters or Namespace
        """
        if isinstance(params, Namespace):
            params = ChainerQFunction.Parameters(params)

        model = params.create_model(observation_space, action_space)
        assert isinstance(model, FeedForwardModel)
        assert isinstance(model, (Chain, ChainList))
        assert isinstance(params, ChainerQFunction.Parameters)
        super(ChainerQFunction, self).__init__(observation_space, action_space)
        self.losses = {
            "td_eval": ChainerTemporalDifferenceLoss(params.discount, 1, 1, "Eval", params.qtype)
        }

        self._nstep = params.nstep
        if self._nstep > 1:
            self._nstep_discount = params.nstep_discount
            self.losses["nstep_eval"] = ChainerTemporalDifferenceLoss(params.discount, params.nstep,
                                                                      params.lambda_nstep, "Eval",
                                                                      params.qtype)

        self.model = model
        if params.device_id > -1:
            self.gpu_device = cuda.get_device_from_id(params.device_id)
            self.gpu_device.use()
            self.model.to_gpu(self.gpu_device)
        else:
            self.gpu_device = None

        self.target = self.copy_model()

        self._optimizer = params.optimizer
        self._optimizer.setup(self.model)
        self._optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(2e-5))

        self.model.cleargrads()
        self._loss = 0
        self._num_examples = 0
        self._loss_summary = ScalarSummary("Mean Loss")

    def copy_model(self):
        """Returns a copy of the current model with the same parameters."""
        model_copy = copy.deepcopy(self.model)
        model_copy.copyparams(self.model)
        if self.gpu_device:
            model_copy.to_gpu(self.gpu_device)

        return model_copy

    @property
    def metrics(self):
        metrics = [self._loss_summary]
        for loss in self.losses.values():
            metrics.extend(loss.metrics)

        return metrics

    def sample_batch(self, memory, size):
        """Samples a batch from the memory.

        Args:
            memory -- the memory
            size -- the size of the batch to sample

        Returns:
            (batch, weights, loss_keys) the batch of samples from the memory, standardized and
            moved to the GPU, the weights if there are any (None otherwise) and the loss keys
            to use
        """
        if self._nstep > 1:
            batch = memory.nstep_minibatch(self.np_random, size, self._nstep,
                                           self._nstep_discount)
        else:
            batch = memory.minibatch(self.np_random, size)

        batch.standardize()

        if self.gpu_device:
            batch.to_gpu_chainer(self.gpu_device)

        loss_keys = filter(lambda key: "eval" in key, self.losses.keys())
        return batch, None, loss_keys

    def present_batch(self, memory, minibatch_size):
        batch, weights, loss_keys = self.sample_batch(memory, minibatch_size)
        loss, losses = self.compute_loss(batch, weights, loss_keys)
        memory.set_priorities(batch.indices, losses)
        self._loss += loss
        self._num_examples += minibatch_size

    def train_model(self):
        assert self._num_examples
        self._loss_summary.add(cuda.to_cpu(self._loss.data) / self._num_examples)
        self._loss.backward()
        self._optimizer.update()
        self.model.cleargrads()
        self._loss = 0
        self._num_examples = 0

    def compute_loss(self, batch, weights, loss_keys):
        """Computes the loss for the batch.

        Args:
            batch -- a training batch
            weights -- the per-sample weights to use when totalling the loss
            loss_keys -- the keys of the losses to use

        Returns (loss, TD loss vector)
        """
        loss = 0.0
        td_losses = None
        for key in loss_keys:
            losses = self.losses[key](self.model, self.target, batch, weights)
            if "td" in key:
                td_losses = cuda.to_cpu(losses.data)
            loss += self.losses[key].weight * F.sum(losses)

        return loss, td_losses

    def update_target(self):
        self.target.copyparams(self.model)

    def prepare_observations(self, observations):
        """Prepare observations for processing.

        Description:
            This method will turn a single observation into a batch by expanding its dimensions.
            It will also move the observations to the GPU device if required.

        Returns observations ready for processing
        """
        if observations.ndim == len(self.model.input_space.shape):
            observations = np.expand_dims(observations, 0)

        if self.gpu_device:
            observations = cuda.to_gpu(observations, self.gpu_device)

        return observations

    def compute(self, observations, is_training=False):
        observations = self.prepare_observations(observations)
        return cuda.to_cpu(self.model(observations).data)

    def save(self, path):
        chainer.serializers.save_npz(path, self.model)

    def load(self, path):
        chainer.serializers.load_npz(path, self.model)
        self.update_target()


class ChainerAveragedQFunction(ChainerQFunction):
    """
    Averaged DQN

    Paper: Averaged-DQN: Variance Reduction and Stabilization for Deep
            Reinforcement Learning by Anschel et al.

    This is a convenience class for Averaged DQN, altering parameters appropriately.
    """

    class Parameters(ChainerQFunction.Parameters):
        """Hyperparameters for training a Chainer averaged Q function"""

        def __init__(self, args=None):
            super(ChainerAveragedQFunction.Parameters, self).__init__(args)
            self.num_targets = 10
            if args:
                self.num_targets = args.num_targets

    @staticmethod
    def add_args_to_parser(parser):
        """Add arguments for the hyperparameters for this class.

        Args:
            parser -- an instance of ArgumentParser

        Description:
            This methods adds the following argument:

            num_targets -- Number of target networks [10]
        """
        ChainerQFunction.add_args_to_parser(parser)
        group = parser.add_argument_group("Chainer Averaged Q-Function")
        group.add_argument("--num_targets", type=int, default=10,
                           help="Number of target networks")

    def __init__(self, observation_space, action_space, params):
        """
        Args:
            observation_space -- the space that observations will be drawn from
            action_space -- the space of actions this agent should produce
            params -- an instance of ChainerAverageQFunction.Parameters or Namespace
        """
        if isinstance(params, Namespace):
            params = ChainerAveragedQFunction.Parameters(params)

        assert isinstance(params, ChainerAveragedQFunction.Parameters)
        assert params.num_targets > 1
        super(ChainerAveragedQFunction, self).__init__(observation_space, action_space, params)
        self.target = FeedForwardModelQueue(params.num_targets,
                                            lambda inputs: np.mean(inputs, axis=1))
        self.update_target()

    def update_target(self):
        self.target.enqueue(self.copy_model())

    def compute(self, observations, is_training=False):
        observations = self.prepare_observations(observations)

        if is_training:
            return super(ChainerAveragedQFunction, self).compute(observations, is_training)

        return cuda.to_cpu(self.target(observations).data)


class ChainerQFunctionFromDemonstration(ChainerQFunction):
    """
    Q Function approximator that learns both from experience and from demonstrations.
    """

    class Parameters(ChainerQFunction.Parameters):
        """Hyperparameters for training a Chainer Q function that learns from demonstration"""

        def __init__(self, args=None):
            super(ChainerQFunctionFromDemonstration.Parameters, self).__init__(args)
            self.lambda_supervised = 1.0
            self.margin_size = 0.8
            self.prioritized_replay = False
            self.beta = lambda step: min(0.4 + step*.01, 1.0)

            if args:
                self.lambda_supervised = args.lambda_supervised
                self.margin_size = args.margin_size
                self.prioritized_replay = args.prioritized_replay
                beta_step = (1.0 - args.beta_start) / args.beta_duration
                self.beta = lambda step: min(args.beta_start + step*beta_step, 1.0)

    @staticmethod
    def add_args_to_parser(parser):
        """Adds arguments for the hyperparameters for this class.

        Args:
            parser -- instance of ArgumentParser

        Description:
            This method adds the following arguments:

            lambda_supervised -- Coefficient for the supervised large-margin loss [1.0]
            margin_size -- Margin size for large margin loss [0.8]
            prioritized_replay -- Whether to used prioritized replay weights when training [False]
            beta_start -- Initial beta value for prioritized replay (max is always 1.0) [0.4]
            beta_duration -- Duration of the beta linear interpolation in off-policy samples [32]
        """
        ChainerQFunction.add_args_to_parser(parser)
        group = parser.add_argument_group("Chainer Q-Function From Demonstration")
        group.add_argument("--lambda_supervised", type=float, default=1.0,
                           help="Coefficient for the supervised large-margin loss")
        group.add_argument("--margin_size", type=float, default=0.8,
                           help="Margin size for large margin loss")
        group.add_argument("--prioritized_replay", action='store_true',
                           help="Whether to used prioritized replay weights when training")
        group.add_argument("--beta_start", type=float, default=0.4,
                           help="Initial beta value for prioritized replay (max is always 1.0)")
        group.add_argument("--beta_duration", type=int, default=32,
                           help="Duration of the beta linear interpolation in off-policy samples")

    def __init__(self, observation_space, action_space, params):
        """
        Args:
            observation_space -- the space that observations will be drawn from
            action_space -- the space of actions this agent should produce
            params -- an instance of ChainerQFunctionFromDemonstration.Parameters or Namespace
        """
        if isinstance(params, Namespace):
            params = ChainerQFunctionFromDemonstration.Parameters(params)

        assert isinstance(params, ChainerQFunctionFromDemonstration.Parameters)
        super(ChainerQFunctionFromDemonstration, self).__init__(observation_space, action_space,
                                                                params)
        self.losses["td_demo"] = ChainerTemporalDifferenceLoss(params.discount, 1, 1,
                                                               "Demo", params.qtype)
        self.losses["lm_demo"] = ChainerLargeMarginLoss(params.margin_size,
                                                        params.lambda_supervised, "Demo")
        if params.nstep > 1:
            self.losses["nstep_demo"] = ChainerTemporalDifferenceLoss(params.discount, params.nstep,
                                                                      params.lambda_nstep, "Demo",
                                                                      params.qtype)
        self._beta = params.beta
        self._prioritized_replay = params.prioritized_replay
        self._beta_step = 0

    def sample_batch(self, memory, size):
        batch, weights, loss_keys = super(ChainerQFunctionFromDemonstration,
                                          self).sample_batch(memory, size)
        if self._prioritized_replay:
            probabilities = np.array(memory.get_minibatch_probability(batch), np.float32)
            beta = self._beta(self._beta_step)
            weights = (len(memory) * probabilities) ** -beta
            weights = self.model.xp.array(weights)
            weights /= weights.max()
            self._beta_step += 1

        if memory.is_demo:
            loss_keys = filter(lambda key: "demo" in key, self.losses.keys())

        return batch, weights, loss_keys
