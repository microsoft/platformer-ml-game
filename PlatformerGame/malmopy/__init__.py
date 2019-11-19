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
MalmoPy is a library which provides reinforcement learning code for use in experimenation and
research. It includes integrations with multiple deep learning frameworks and simulation
environments and reference implementations of a variety of reinforcement learning algorithms. It
also provides abstract classes and implementation definitions to enable easy extending of
functionality.
"""

import logging

from .abc import MetricKind, Metric, Visualizable, Visualizer, Evaluator, Explorer, Memory,\
    StateBuilder, Sampler, FeedForwardModel, PolicyAndValueModel, History, Command,\
    EpsilonFunction, Explorable, DriverExtension
from .core import FileRecorder, FeedForwardModelQueue, Agent, AgentMode, AgentWrapper,\
    QFunctionApproximator, QType, TrajectoryLearner, ActWrapper, ObserveWrapper,\
    VisualizableAgentWrapper, VisualizableWrapper, ExplorableGraph
from .driver import Driver, simple_driver
from .evaluators import EpisodicEvaluator, StepEvaluator
from .explorers import ConstantExplorer, LinearEpsilonGreedyExplorer, ConstantEpsilon
from .explorers import LinearEpsilon, EpsilonGreedyExplorer
from .samplers import UniformSampler, PrioritizedSampler, get_sampler
from .multidriver import MultiagentDriver, MultiagentWrapper
from .extensions import EvaluationExtension, CheckpointExtension, VisualiseNumActionsExtension,\
    VisualizerExtension, WritePngsExtension, ThroughputExtension
from .summaries import MinSummary, MedianSummary, MaxSummary, SumSummary, VarianceSummary,\
    ImageSummary, OnDemandImageSummary, ScalarSummary, Summary, MeanSummary, HistogramSummary
from .triggers import IntervalTrigger, LimitTrigger, AlwaysTrigger, NeverTrigger,\
    ElapsedTimeTrigger, each_step, each_episode, each_epoch, IntervalUnit
from .spaces import DiscreteCommand, ContinuousTupleCommand, ContinuousStepCommand,\
    to_history_space, StringCommand
from . import memories, agents, utils

FORMAT = '%(levelname)s\t%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
