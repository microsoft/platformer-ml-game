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

"""Module containing the duck types and simple ABCs"""

from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from enum import IntEnum

import gym

from .utils import check_methods


class DriverExtension(ABC):
    """Abstract base class for a Driver extension."""

    def __init__(self, trigger, name=None):
        """
        Args:
            trigger -- the trigger that determines the extension should run

        Keyword Args:
            name -- the name of the extension [None]
        """
        self.name = name if name else type(self).__name__
        self.trigger = trigger

    def __call__(self, driver):
        if self.trigger(driver):
            self._internal_call(driver)

    def finalize(self):
        """Called when the driver exits"""

    @abstractmethod
    def _internal_call(self, driver):
        """Called by the driver at various points during execution."""
        raise NotImplementedError()

    @property
    def metrics(self):
        """Override to add metrics"""
        return []


class Explorable(ABC):
    """Duck type definition for a simulation which allows exploration."""

    __slots__ = ()

    @abstractmethod
    def try_step(self, observation, action):
        """Tries taking an action (without actually modifying the environment).

        Description:
            This method can be used for trying out various actions, primarily for use in building
            demonstrators.

        Args:
            observation -- the observation indicating the starting state
            action -- the action to take

        Returns the result of taking that action
        """

    @property
    @abstractmethod
    def goal(self):
        """The current goal or target observation for the simulation."""

    @abstractmethod
    def available_actions(self, observation):
        """The set of actions to take given an observation.

        Args:
            observation -- the observation to predicate actions upon

        Returns a set of samples from the action space appropriate to the observation
        """

    @abstractmethod
    def distance(self, observation0, observation1):
        """The distance between two observations.

        Description:
            The distance computed here should be useful for search algorithms, i.e. monotonic,
            obeys the triangle inequality.

        Args:
            observation0 -- a first observation from the simulation
            observation1 -- a second observation from the simulation

        Returns a single scalar value representing the distance between the two observations
        """

    @abstractmethod
    def hash(self, observation):
        """Create a unique hash key for the provided observation.

        Args:
            observation -- the observation sample

        Returns a unique hash key
        """

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Explorable:
            return check_methods(class_obj, "try_step", "hash", "goal", "available_actions", "distance")

        return NotImplemented


class FeedForwardModel(ABC):
    """
    Duck type definition for a class which provides a feed forward model
    that maps one space to another.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def input_space(self):
        """The gym.Space from which inputs will be drawn"""

    @property
    @abstractmethod
    def output_space(self):
        """The gym.Space from which outputs will be drawn"""

    @abstractmethod
    def compute(self, inputs):
        """Computes the outputs for the provided inputs.

        Args:
            inputs -- a vector of inputs drawn from the input space

        Returns a vector of mapped outputs for each input
        """

    def __call__(self, inputs):
        return self.compute(inputs)

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is FeedForwardModel:
            return check_methods(class_obj, "input_space", "output_space", "compute", "__call__")

        return NotImplemented


class PolicyAndValueModel(ABC):
    """
    Duck type definition for a model that produces a policy and value output for
    an observation.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def observation_space(self):
        """The gym.Space from which inputs will be drawn"""

    @property
    @abstractmethod
    def action_space(self):
        """The gym.Space from which outputs will be drawn"""

    @abstractmethod
    def compute(self, observations):
        """Computes the policy and value values for the provided observations.

        Args:
            observations -- a vector of observations drawn from the observation space

        Returns (policy, values) where the policy is a vector over actions and each value is
        the value for that observation.
        """

    def __call__(self, observations):
        return self.compute(observations)

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is PolicyAndValueModel:
            return check_methods(class_obj, "observation_space", "action_space", "compute",
                                 "__call__")

        return NotImplemented


class Evaluator(ABC):
    """
    Duck type definition for a simple version of Driver. It runs a pre-trained agent
    on a specific environment, which can also be different from the one the agent was
    trained on as long as the input shape matches.
    """

    __slots__ = ()

    @abstractmethod
    def run(self, training_epoch, env, agent):
        """Runs the evaluation.

        Args:
            training_epoch -- integer indicating the status of the training
            env -- the environment
            agent -- the agent to evaluate

        Returns:
            list containing the metrics from the evaluation in the order matching
            get_metric_names.
        """

    @abstractmethod
    def get_metric_names(self):
        """Returns a list containing the metric names.

        Generates the list of metric names that will be generated by the evaluator.
        """

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Evaluator:
            return check_methods(class_obj, "run", "get_metric_names")

        return NotImplemented


class Explorer(ABC):
    """Duck type for classes which can be used as explorers"""

    @staticmethod
    def add_args_to_parser(arg_parser):
        """Adds arguments to create an explorer to the parser.

        Description:
            This method adds the following arguments:

            explorer -- the type of explorer ["constant"]
            constant_epsilon -- The value of epsilon for the constant explorer [0.1]
            linear_range -- the range of values for the linear explorer [(0.9, 0.1)]
            linear_duration -- the duration of the linear interpolation in steps [1000]
        """
        assert isinstance(arg_parser, ArgumentParser)
        group = arg_parser.add_argument_group("Explorer")
        group.add_argument("--explorer", type=str, default=0.1,
                           choices=["constant", "linear"],
                           help="The type of exploration strategy to use")
        group.add_argument("--constant_epsilon", type=float, default=0.1,
                           help="The value of epsilon for the constant explorer")
        group.add_argument("--linear_range", nargs=2, type=float, default=[0.9, 0.1],
                           help="max_epsilon,min_epsilon for linear interpolation")
        group.add_argument("--linear_duration", type=int, default=1000,
                           help="The duration of the linear interpolation in steps")

    @staticmethod
    def create(args):
        """Creates an Explorer object given the provided arguments."""
        assert isinstance(args, Namespace)
        if args.explorer == "constant":
            from .explorers import ConstantExplorer
            return ConstantExplorer(args.constant_epsilon)

        if args.explorer == "linear":
            eps_max, eps_min = args.linear_range
            eps_min_time = args.linear_duration

            if args.resume:
                from .explorers import ConstantExplorer
                return ConstantExplorer(eps_min)

            from .explorers import LinearEpsilonGreedyExplorer
            return LinearEpsilonGreedyExplorer(eps_max, eps_min, eps_min_time)

        raise NotImplementedError


    __slots__ = ()

    def __call__(self, step, action_space):
        return self.explore(step, action_space)

    @abstractmethod
    def is_exploring(self, step):
        """Returns whether the agent is exploring or exploiting.

        Args:
            step -- the current action step

         Returns:
            True when exploring, False when exploiting
        """

    @abstractmethod
    def explore(self, step, action_space):
        """Generate an exploratory action

        Args:
            step -- the current step in the simulation
            action_space -- the action space of the environment
        """

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Explorer:
            return check_methods(class_obj, "is_exploring", "explore", "__call__")

        return NotImplemented


class Memory(ABC):
    """
    Abstract base class for representations of agent's memories
    """

    @staticmethod
    def add_args_to_parser(parser):
        """Adds arguments for creating a sampler to the parser.

        Description:
            This method adds the following arguments:

            memory -- the type of memory ["replay"]
            num_replay_steps -- the maximum size of the memory [1000]
            sampler -- the sampler type for the memory ["uniform"]
            default_priority -- the default priority for the prioritized sampler [1.0]
            temporal_unflicker -- whether temporal memory should help with flickering [False]
        """
        parser.add_argument("--memory", type=str, default="replay",
                            choices=["replay", "episodic", "temporal"],
                            help="The type of memory to create")
        parser.add_argument("--num_replay_steps", type=int, default=1000,
                            help="The maximum size of the memory (for replay and temporal)")
        parser.add_argument("--sampler", type=str, default="uniform",
                            choices=["uniform", "prioritized"],
                            help="The type of sampler to use for memory sampling")
        parser.add_argument("--default_priority", type=float, default=1.0,
                            help="The default priority of a prioritized sampler")
        parser.add_argument("--temporal_unflicker", action="store_true",
                            help="Whether to unflicker samples over time (temporal only)")

    @staticmethod
    def create(args, observation_space, action_space):
        """Creates a new memory object from the arguments."""
        assert isinstance(args, Namespace)
        assert isinstance(observation_space, gym.Space)
        assert isinstance(action_space, gym.Space)
        if args.sampler == "uniform":
            from .samplers import UniformSampler
            sampler = UniformSampler()
        elif args.sampler == "prioritized":
            from .samplers import PrioritizedSampler
            sampler = PrioritizedSampler(args.default_priority)
        else:
            raise NotImplementedError

        if args.memory == "replay":
            from .memories import ReplayMemory
            return ReplayMemory(observation_space, action_space, sampler,
                                max_size=args.num_replay_steps)

        if args.memory == "temporal":
            from .memories import TemporalMemory
            return TemporalMemory(observation_space, action_space, args.temporal_unflicker, sampler,
                                  max_size=args.num_replay_steps)

        if args.memory == "episodic":
            from .memories import EpisodicMemory
            return EpisodicMemory(observation_space, action_space, sampler)

        raise NotImplementedError

    __slots__ = ()

    @property
    @abstractmethod
    def observation_space(self):
        """The space of observations stored in this memory"""

    @property
    @abstractmethod
    def action_space(self):
        """The space of actions stored in this memory"""

    @abstractmethod
    def __getitem__(self, key):
        """Returns the memory element at the provided key"""

    @abstractmethod
    def __len__(self):
        """Number of elements currently stored in the memories"""

    @abstractmethod
    def reset(self):
        """Resets the memory, removing all transitions."""

    @property
    def is_full(self):
        """Whether the memory is full."""

    @abstractmethod
    def minibatch(self, prng, count):
        """Generate a minibatch.

        Description:
            Generates a random minibatch with the number of samples specified by the size parameter.

        Args:
            prng -- the pseudo-random number generator used to determine the random batch indices
            count -- number of samples in the minibatch

        Returns:
            Batch object containing samples
        """

    @abstractmethod
    def get_probabilities(self, indices):
        """Returns the sampling probability per index.

        Description:
            This method returns the probability (for each index) of that transition being sampled
            by the `minibatch` method.

        Args:
            indices -- the indices to use when computing probabilities

        Returns:
            a list of probabilities per index
        """

    @abstractmethod
    def append(self, observation, action, reward, done, priority):
        """Appends the specified values to the memory.

        Args:
            observation -- The observation to append (should have the same shape as defined at
                           initialization time)
            action -- An integer representing the action done
            reward -- A float representing the reward received for doing this action
            done -- A boolean specifying if this state is a terminal (episode has finished)
            priority -- Priority of the observation in the buffer
        """

    def __repr__(self) -> str:
        """Returns a string representation of the memory"""
        return '{}(length={}, obs_space={}, act_space={})'.format(self.__class__.__name__,
                                                                  len(self),
                                                                  self.observation_space,
                                                                  self.action_space)

    def to_ndarray(self):
        """Returns the memory as a complex dtype ndarray"""

    def from_ndarray(self, array, sampler):
        """Sets the values in the memory using a complex dtype ndarray.

        Args:
            array -- an array containing 'states', 'actions', 'rewards' and 'terminals'
            sampler -- a sampler to use for the array. If None, a default sampler will be used.

        Returns a reference to this object
        """

    @abstractmethod
    def save(self, file):
        """Save the memory to the specified file or path

        Args:
            file -- either a file object or a path to a location on the disk
        """

    @abstractmethod
    def load(self, file):
        """Load the memory from the specified file or path

        Args:
            path -- either a file object or a path to a location on the disk
        """

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Memory:
            return check_methods(class_obj, "observation_space", "action_space", "__len__",
                                 "minibatch", "append", "save", "load", "to_ndarray",
                                 "from_ndarray", "is_full")

        return NotImplemented


class StateBuilder(ABC):
    """
    Duck type definition for classes which map environment state into another representation.
    """

    __slots__ = ()

    @abstractmethod
    def build(self, environment):
        """Build and return an observation.

        Description:
            Builds and returns an observation of shape `shape` that reflects
            the current observable state of the environment.

        Args:
            environment -- the environment to observe

        Returns:
            an observation of shape `shape`
        """

    @property
    @abstractmethod
    def space(self):
        """Return the space of observations generated by this `StateBuilder`"""

    def __call__(self, environment):
        """Calls the `build()` method.

        Args:
            environment -- the environment to observe

        Returns:
            an observation of shape `shape`
        """
        return self.build(environment.unwrapped)

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is StateBuilder:
            return check_methods(class_obj, "space", "build", "__call__")

        return NotImplemented


class Command(ABC):
    """Space which assigns explicit text commands to different actions."""

    __slots__ = ()

    @property
    @abstractmethod
    def verbs(self):
        """The list of commands"""

    @abstractmethod
    def to_commands(self, sample_n):
        """Converts a set of action space samples to text commands.

        Args:
            sample_n -- a list of samples or a single sample
        """

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Command:
            return check_methods(class_obj, "verbs", "to_commands")

        return NotImplemented


class History(ABC):
    """Space which is formed by stacking observations from another space."""

    __slots__ = ()

    @property
    @abstractmethod
    def inner(self):
        """The space of the observations in this history."""

    @property
    @abstractmethod
    def length(self):
        """The length of the history."""

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is History:
            return check_methods(class_obj, "inner", "length")

        return NotImplemented


class MetricKind(IntEnum):
    """Enumeration containing different kinds of metrics"""
    SCALAR = 0
    IMAGE = 1
    HISTOGRAM = 2
    DISTRIBUTION = 3


class Metric(ABC):
    """Duck type definition for classes which represent a metric that can be visualized."""
    __slots__ = ()

    @property
    @abstractmethod
    def shape(self):
        """The n-dimensional shape of the metric"""

    @property
    @abstractmethod
    def value(self):
        """The value of the metric"""

    @property
    @abstractmethod
    def kind(self):
        """The kind of metric"""

    @property
    def ndim(self):
        """The number of dimensions in the metric"""
        return len(self.shape)

    @abstractmethod
    def reset(self):
        """Resets the metric to a default value"""

    @abstractmethod
    def add(self, value):
        """Add a new value to the metric.

        Args:
            value -- a value to add to the metric
        """

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Metric:
            return check_methods(class_obj, "shape", "value", "kind",
                                 "reset", "__lshift__", "ndim")

        return NotImplemented


class Visualizable(ABC):
    """Duck type defition for classes which can be visualized."""

    __slots__ = ()

    @property
    @abstractmethod
    def metrics(self):
        """A list of metrics for this visualizable entity"""

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Visualizable:
            return check_methods(class_obj, "metrics")

        return NotImplemented


class Visualizer(ABC):
    """Duck type definition for a class which enables visualization."""

    __slots__ = ()

    @abstractmethod
    def add_summary(self, summary, step):
        """Adds a summary to the Visualizer.

        Args:
            summary - the summary object (implements Summary)

        Keyword args:
            step - the step associated with the summary [0]
        """

    @abstractmethod
    def set_metrics(self, metrics):
        """Initialize the visualizer.

        Args:
            metrics -- the metrics that will be used with this visualizer
        """

    @abstractmethod
    def is_supported(self, summary):
        """Returns whether this summary is supported by this visualizer.

        Args:
            summary - the summary object (implements Summary)

        Returns:
            Whether this visualizer supports this kind of summary
        """

    @abstractmethod
    def close(self):
        """Closes the visualizer and cleans up any active resources."""

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Visualizer:
            return check_methods(class_obj, "add_summary", "is_supported")

        return NotImplemented


class Sampler(ABC):
    """
    Duck type definition for a sampler
    """

    __slots__ = ()

    @abstractmethod
    def __len__(self):
        """Number of items known to the sampler"""

    @abstractmethod
    def __getitem__(self, index):
        """Returns the probability of the index being sampled.

        Args:
            index -- the index in the sampler
        """

    @abstractmethod
    def append(self):
        """Appends an index to the sampler."""

    @abstractmethod
    def get_probabilities(self, indices, start, end):
        """Get the bounded probabilities of a list of indices being selected.

        Args:
            indices -- the indices to query
            start -- the start index
            end -- the end index (exclusive)

        Returns:
            a list containing the likelihood each index would be sampled
        """

    @abstractmethod
    def sample(self, prng, count, start, end, replace):
        """Samples a range of indices with or without replacement.

        Args:
            prng -- the pseudo-random number generator
            count -- the number of indices to sample
            start -- the starting value in the range
            end -- the end value in the range (exclusive)
            replace -- whether to sample with replacement

        Returns:
            a list of sampled indices
        """

    @abstractmethod
    def resize(self, num_indices):
        """Resize the sampler to support the specified number of indicies.

        Description:
            The sampler will be resized to match the number of indices. If the sampler tracks
            priority, then the priority will be set to the default value for all indices.

        Args:
            num_indices -- the desired number of indices in the sampler
        """

    @abstractmethod
    def reset(self):
        """Resets the sampler, clearing all values."""

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return '{}(size={})'.format(self.__class__.__name__, len(self))

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Sampler:
            return check_methods(class_obj, "__len__", "__getitem__", "append", "sample",
                                 "reset", "get_probabilities")

        return NotImplemented


class EpsilonFunction(ABC):
    """Duck type definition for classes which can compute a step-based epsilon."""

    __slots__ = ()

    @abstractmethod
    def epsilon(self, step):
        """Computes the episilon at the provided step.

        Args:
            step -- the step in the learning process

        Returns the epsilon value
        """

    def __call__(self, step):
        return self.epsilon(step)

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is EpsilonFunction:
            return check_methods(class_obj, "__call__", "epsilon")

        return NotImplemented
   