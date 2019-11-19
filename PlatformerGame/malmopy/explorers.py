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

"""Module containing explorer classes"""

from numpy import random as np_random

from .summaries import ScalarSummary
from .triggers import each_step
from .abc import Explorer, EpsilonFunction, Visualizable

class ConstantEpsilon(EpsilonFunction):
    """Epsilon function which returns a constant value regardless of step."""

    def __init__(self, epsilon):
        """
        Args:
            epsilon -- the constant epsilon value
        """
        self._epsilon = epsilon

    def epsilon(self, step):
        return self._epsilon


class LinearEpsilon(EpsilonFunction):
    """
    This function uses linear interpolation between epsilon_max and epsilon_min
    to linearly anneal epsilon as a function of the current episode.

    3 cases exist:
        - If 0 <= episode < eps_min_time then epsilon = interpolator(episode)
        - If episode >= eps_min_time then epsilon then epsilon = eps_min
        - Otherwise epsilon = eps_max
    """

    def __init__(self, eps_max, eps_min, eps_min_time):
        """
        Args:
            eps_max -- the maximum epsilon value
            eps_min -- the minimum epsilon value
            eps_min_time -- the number of steps until epsilon is at its minimum
        """
        assert eps_max > eps_min
        assert eps_min_time > 0

        self._eps_min_time = eps_min_time
        self._eps_min = eps_min
        self._eps_max = eps_max

        self._delta = -(eps_max - eps_min) / eps_min_time

    def epsilon(self, step):
        """The epsilon value at a specific step.

        Args:
            step -- the step during training
        """
        if step < 0:
            return self._eps_max

        if step > self._eps_min_time:
            return self._eps_min

        return self._delta * step + self._eps_max


class EpsilonGreedyExplorer(Explorer, Visualizable):
    """Explorer which determines whether to explore by sampling from a Bernoulli distribution."""

    def __init__(self, epsilon_function):
        """
        Args:
            epsilon_function -- an instance of EpsilonFunction
        """
        assert isinstance(epsilon_function, EpsilonFunction)
        self._epsilon = epsilon_function
        self._epsilon_summary = ScalarSummary("EpsilonGreedy/Epsilon", each_step())

    @property
    def metrics(self):
        return [self._epsilon_summary]

    def is_exploring(self, step):
        epsilon = self._epsilon(step)
        self._epsilon_summary.add(epsilon)
        return np_random.binomial(1, epsilon)

    def explore(self, step, action_space):
        return action_space.sample()

class ConstantExplorer(EpsilonGreedyExplorer):
    """Explorer which explores with a constant probability."""

    def __init__(self, epsilon):
        """
        Args:
            epsilon -- the probability that the agent will explore
        """
        super(ConstantExplorer, self).__init__(ConstantEpsilon(epsilon))


class LinearEpsilonGreedyExplorer(EpsilonGreedyExplorer):
    """Explorer which uses a LinearEpsilon function."""

    def __init__(self, eps_max, eps_min, eps_min_time):
        """
        Args:
            eps_max -- the maximum epsilon value
            eps_min -- the minimum epsilon value
            eps_min_time -- the number of steps until epsilon is at its minimum
        """
        epsilon_function = LinearEpsilon(eps_max, eps_min, eps_min_time)
        super(LinearEpsilonGreedyExplorer, self).__init__(epsilon_function)
