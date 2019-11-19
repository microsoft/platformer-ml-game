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

import logging

import gym
import numpy as np
from chainer import optimizers
from chainer import Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

from ... import FeedForwardModel


def get_in_channels(input_space, in_channels):
    """Determine (or check) the input channels.

    Args:
        input_space -- the input gym.Space
        in_channels -- the provided input channels. None indicates to infer the value

    Returns the number of input channels
    """
    logger = logging.getLogger(__name__)
    assert input_space.shape is not None
    if in_channels:
        assert in_channels == input_space.shape[0]
    else:
        if len(input_space.shape) == 2:
            in_channels = 1
        else:
            in_channels = input_space.shape[0]

        logger.warning("Inferring input channel count of %d from shape %s", in_channels,
                       str(input_space.shape))

    return in_channels


class ChainerMLP(FeedForwardModel, ChainList):
    """
    Class encapsulating a simple Multi-layer Perceptron with a configurable
    number of hidden layers.
    """

    def __init__(self, input_space, output_space, hidden_layer_sizes=(512, 512, 512),
                 activation=F.relu, zero_bias=False):
        """
        Args:
            input_space -- the space that inputs will be drawn from
            output_space -- the space that outputs will be drawn from.

        Keyword Args:
            hidden_layer_sizes -- the number of units in each hidden layer in order
                                  [(512, 512, 512)]
            activation -- the activation function used on all layers [relu]
            zero_bias -- whether the bias should be initialized to zero [False]
        """
        assert isinstance(input_space, gym.Space)
        assert input_space.shape is not None
        if isinstance(output_space, gym.spaces.Discrete):
            out_units = output_space.n
        elif isinstance(output_space, gym.spaces.Box):
            assert len(output_space.shape) == 1
            out_units = output_space.shape[0]
        else:
            raise NotImplementedError

        self._input_space = input_space
        self._output_space = output_space
        self._num_hidden = len(hidden_layer_sizes)
        assert self._num_hidden > 0

        initial_weights = I.HeUniform()
        initial_bias = None if zero_bias else I.Uniform(1e-4)
        links = []
        for units in hidden_layer_sizes:
            links.append(L.Linear(None, units, False, initial_weights, initial_bias))

        links.append(L.Linear(None, out_units, False, initial_weights, initial_bias))
        self._activation = activation

        super(ChainerMLP, self).__init__(*links)
        self(np.zeros((1, ) + input_space.shape, input_space.dtype))

    @property
    def input_space(self):
        return self._input_space

    @property
    def output_space(self):
        return self._output_space

    def compute(self, inputs):
        var = inputs
        for i in range(self._num_hidden):
            var = self._activation(self[i](var))

        return self[self._num_hidden](var)


class ChainerDQN(FeedForwardModel, Chain):
    """The neural net from the Nature paper"""

    def __init__(self, input_space, output_space, zero_bias=False, in_channels=None):
        super(ChainerDQN, self).__init__()
        assert isinstance(input_space, gym.Space)
        assert input_space.shape is not None
        if isinstance(output_space, gym.spaces.Discrete):
            out_units = output_space.n
        elif isinstance(output_space, gym.spaces.Box):
            assert len(output_space.shape) == 1
            out_units = output_space.shape[0]
        else:
            raise NotImplementedError

        self._input_space = input_space
        self._output_space = output_space
        initial_bias = None if zero_bias else I.Uniform(1e-4)
        in_channels = get_in_channels(input_space, in_channels)
        with self.init_scope():
            self.conv0 = L.Convolution2D(in_channels, 32, ksize=8, stride=4,
                                         initialW=I.HeUniform(), initial_bias=initial_bias)
            self.conv1 = L.Convolution2D(None, 64, ksize=4, stride=2,
                                         initialW=I.HeUniform(), initial_bias=initial_bias)
            self.conv2 = L.Convolution2D(None, 64, ksize=3, stride=1,
                                         initialW=I.HeUniform(), initial_bias=initial_bias)
            self.fc0 = L.Linear(None, 512, initialW=I.HeNormal(scale=0.01),
                                initial_bias=initial_bias)
            self.fc1 = L.Linear(None, out_units, initialW=I.HeNormal(scale=0.01),
                                initial_bias=initial_bias)

        self(np.zeros((1, ) + input_space.shape, input_space.dtype))

    @property
    def input_space(self):
        return self._input_space

    @property
    def output_space(self):
        return self._output_space

    def compute(self, inputs):
        hidden = F.relu(self.conv0(inputs))
        hidden = F.relu(self.conv1(hidden))
        hidden = F.relu(self.conv2(hidden))
        hidden = F.relu(self.fc0(hidden))
        return self.fc1(hidden)


class ChainerSmallDQN(FeedForwardModel, Chain):
    """The neural net from the original arXiv paper"""

    def __init__(self, input_space, output_space, zero_bias=False, in_channels=None):
        super(ChainerSmallDQN, self).__init__()
        assert isinstance(input_space, gym.Space)
        assert input_space.shape is not None
        if isinstance(output_space, gym.spaces.Discrete):
            out_units = output_space.n
        elif isinstance(output_space, gym.spaces.Box):
            assert len(output_space.shape) == 1
            out_units = output_space.shape[0]
        else:
            raise NotImplementedError

        self._input_space = input_space
        self._output_space = output_space
        initial_bias = None if zero_bias else I.Uniform(1e-4)
        in_channels = get_in_channels(input_space, in_channels)
        with self.init_scope():
            self.conv0 = L.Convolution2D(in_channels, 16, ksize=8, stride=4,
                                         initialW=I.HeUniform(), initial_bias=initial_bias)
            self.conv1 = L.Convolution2D(None, 32, ksize=4, stride=2,
                                         initialW=I.HeUniform(), initial_bias=initial_bias)
            self.fc0 = L.Linear(None, 256, initialW=I.HeNormal(scale=0.01),
                                initial_bias=initial_bias)
            self.fc1 = L.Linear(None, out_units, initialW=I.HeNormal(scale=0.01),
                                initial_bias=initial_bias)

        self(np.zeros((1, ) + input_space.shape, input_space.dtype))

    @property
    def input_space(self):
        return self._input_space

    @property
    def output_space(self):
        return self._output_space

    def compute(self, inputs):
        hidden = F.relu(self.conv0(inputs))
        hidden = F.relu(self.conv1(hidden))
        hidden = F.relu(self.fc0(hidden))
        return self.fc1(hidden)


def add_optimizer_args(parser, **kwargs):
    """Adds the arguments for the standard optimizers."""
    parser.add_argument('--opt', type=str, default=kwargs.get("opt", "sgd"),
                        choices=['sgd', 'adam'], help="Optimizer algorithm")
    parser.add_argument('--learning_rate', type=float, default=kwargs.get("learning_rate", 0.0001),
                        help="The learning rate (alpha for Adam")
    parser.add_argument('--adam_beta1', type=float, default=kwargs.get("adam_beta1", 0.9),
                        help="Beta1 parameter for Adam")
    parser.add_argument('--adam_beta2', type=float, default=kwargs.get("adam_beta2", 0.999),
                        help="Beta2 parameter for Adam")
    parser.add_argument('--adam_eps', type=float, default=kwargs.get("adam_eps", 1e-8),
                        help="Epsilon parameter for Adam")

def create_optimizer(args):
    """Creates an optimizer object from the provided arguments."""
    if args.opt == "sgd":
        return optimizers.SGD(args.learning_rate)
    elif args.opt == "adam":
        return optimizers.Adam(args.learning_rate, args.adam_beta1, args.adam_beta2, args.adam_eps)
    else:
        raise NotImplementedError


def add_model_args(parser):
    """Adds the arguments for the model."""
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "dqn", "small_dqn"],
                        help="The type of neural net model to use.")
    parser.add_argument("--hidden_layer_sizes", type=int, nargs='+', default=[64, 64],
                        help="indicates the number of units in the hidden layers")
    parser.add_argument("--zero_bias", action="store_true",
                        help="Whether to have a zero bias initialization")


def create_model(args, observation_space, action_space):
    """Creates a model from the arguments."""
    if args.model == "mlp":
        return ChainerMLP(observation_space, action_space, tuple(args.hidden_layer_sizes))

    if args.model == "dqn":
        return ChainerDQN(observation_space, action_space, args.zero_bias)

    if args.model == "small_dqn":
        return ChainerSmallDQN(observation_space, action_space, args.zero_bias)

    raise NotImplementedError
