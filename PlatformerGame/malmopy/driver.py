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

"""Module containing the Driver class which runs experiments."""

from argparse import ArgumentParser, Namespace
import os
from datetime import datetime

import gym

from . import AgentMode, Agent, FileRecorder, VisualizableAgentWrapper, VisualizableWrapper
from .extensions import VisualizerExtension, EvaluationExtension, CheckpointExtension,\
    DriverExtension, WritePngsExtension
from .triggers import each_episode, each_epoch, IntervalTrigger, IntervalUnit, LimitTrigger
from .evaluators import StepEvaluator, EpisodicEvaluator


def get_trigger(freq):
    """Creates a frequency trigger."""
    if freq == "episode":
        return each_episode()

    if freq == "epoch":
        return each_epoch()

    if freq == "never":
        return None

    raise NotImplementedError


class Driver(object):
    """The Driver class runs an experiment, guiding the agent to interact with the environment."""

    EVAL_ONLY = -1

    class Parameters(object):
        """Parameters for the Driver"""

        def _add_evaluator(self, logdir, args):
            if args.eval_file:
                record_path = os.path.join(logdir, args.eval_file)
                recorder = FileRecorder(record_path, vars(args))
            else:
                recorder = None

            if args.eval == "step":
                evaluator = StepEvaluator(args.eval_steps, recorder)
            elif args.eval == "episodic":
                evaluator = EpisodicEvaluator(args.eval_episodes, recorder)
            elif args.eval == "none":
                evaluator = None
            else:
                raise NotImplementedError

            if evaluator:
                trigger = get_trigger(args.eval_freq)
                self.extensions.append(EvaluationExtension(trigger, evaluator))


        def _add_visualizer(self, logdir, args):
            if args.viz == "tensorboard":
                from .visualizers.tensorboard import TensorboardVisualizer
                visualizer = TensorboardVisualizer(logdir)
            elif args.viz == "hdf5":
                from .visualizers import HDF5Visualizer
                visualizer = HDF5Visualizer(logdir)
            elif args.viz == "none":
                visualizer = None
            else:
                raise NotImplementedError

            if visualizer:
                trigger = get_trigger(args.viz_freq)
                self.extensions.append(VisualizerExtension(trigger, visualizer))

        def __init__(self, args=None, logdir_postfix=None):
            """
            Keyword Args:
                args -- a Namespace object
                logdir_postfix -- a postfix to attach to the end of the user logdir. If not
                                  specified, a timestamp will be used.
            """
            self.epoch_length = 250000
            self.stop_trigger = LimitTrigger(40, IntervalUnit.Epoch)
            self.train_after = 32
            self.extensions = []

            if args:
                self.epoch_length = args.epoch_length

                if args.limit_unit == "step":
                    self.stop_trigger = LimitTrigger(args.num_units, IntervalUnit.Step)
                elif args.limit_unit == "episode":
                    self.stop_trigger = LimitTrigger(args.num_units, IntervalUnit.Episode)
                elif args.limit_unit == "epoch":
                    self.stop_trigger = LimitTrigger(args.num_units, IntervalUnit.Epoch)
                else:
                    raise NotImplementedError

                self.train_after = args.train_after
                if not logdir_postfix:
                    logdir_postfix = datetime.utcnow().strftime("%y-%m-%dT%H-%M-%S")

                if args.logdir:
                    logdir = os.path.join(args.logdir, logdir_postfix)
                    self._add_visualizer(logdir, args)
                else:
                    logdir = None

                self._add_evaluator(logdir, args)

                if args.pngdir:
                    pngdir = os.path.join(logdir, args.pngdir)
                    trigger = IntervalTrigger(args.png_freq, IntervalUnit.Episode,
                                              once_per_unit=False)
                    self.extensions.append(WritePngsExtension(trigger, pngdir))

                trigger = get_trigger(args.checkpoint_freq)
                if trigger:
                    self.extensions.append(CheckpointExtension(trigger, logdir))


    @staticmethod
    def add_args_to_parser(parser):
        """Adds arguments for the Driver to the parser.

        Args:
            parser -- instance of ArgumentParser

        Description:
            This method adds the following arguments:

            epoch_length -- The length of an epoch in steps [250000]
            limit_unit -- The unit that determines when to stop the experiment ["epoch"]
            num_units -- The number of units before the experiment should stop [40]
            train_after -- Number of steps after which to start training [32]
            viz -- Visualizer to use for metrics data ["none"]
            viz_freq -- Frequency at which the visualizer updates ["episode"]
            logdir -- Directory for metrics and evaluation results [""]
            pngdir -- Directory for writing PNG renderings of the environment [None]
            eval -- Type of evaluation to perform ["none"]
            eval_file -- File for writing the evaluation results [""]
            eval_freq -- Frequency at which evaluation is performed ["epoch"]
            eval_steps -- The number of steps to run evaluation [500]
            eval_episodes -- The number of episodes to run evaluation [10]
            checkpoint_freq -- The frequency to create checkpoints ["never"]
        """
        assert isinstance(parser, ArgumentParser)
        group = parser.add_argument_group("Driver")
        group.add_argument("--epoch_length", type=int, default=250000,
                           help="The number of steps in an epoch")
        group.add_argument("--limit_unit", type=str, default="epoch",
                           choices=["step", "episode", "epoch"],
                           help="The unit that determines when to stop the experiment")
        group.add_argument("--num_units", type=int, default=40,
                           help="The number of units before the experiment should stop")
        group.add_argument("--train_after", type=int, default=32,
                           help="Number of steps after which to start training")
        group.add_argument("--viz", type=str, default="none",
                           choices=["none", "hdf5", "tensorboard"],
                           help="Visualizer to use for metrics data")
        group.add_argument("--viz_freq", type=str, default="episode",
                           choices=["episode", "epoch"],
                           help="Frequency at which the visualizer updates")
        group.add_argument("--logdir", type=str, default=None,
                           help="Directory for metrics and evaluation results")
        group.add_argument("--pngdir", type=str, default=None,
                           help="Directory for writing PNG renderings of the environment")
        group.add_argument("--png_freq", type=int, default=5000,
                           help="Frequency (in episodes) which with PNGs should be written")
        group.add_argument("--eval", type=str, default="none",
                           choices=["none", "step", "episodic"],
                           help="Type of evaluation to perform")
        group.add_argument("--eval_freq", type=str, default="epoch",
                           choices=["episode", "epoch"],
                           help="Frequency at which evaluation is performed")
        group.add_argument("--eval_steps", type=int, default=500,
                           help="Number of evaluation steps")
        group.add_argument("--eval_episodes", type=int, default=10,
                           help="Number of evaluation episodes")
        group.add_argument("--eval_file", type=str, default=None,
                           help="Record file for evaluator")
        group.add_argument("--checkpoint_freq", type=str, default="never",
                           choices=["never", "episode", "epoch"],
                           help="Frequency at which to create checkpoints")

    def __init__(self, agent, environment, params):
        if isinstance(params, Namespace):
            params = Driver.Parameters(params)

        assert isinstance(params, Driver.Parameters)
        assert isinstance(agent, Agent)
        assert isinstance(environment, gym.Env)

        self.agent = VisualizableAgentWrapper(agent)
        self.environment = VisualizableWrapper(environment)
        self._stop_trigger = params.stop_trigger
        self._epoch_len = params.epoch_length
        self._train_after = params.train_after
        self._extensions = params.extensions

        self._step = 0
        self._episode = 0
        self._epoch = 0
        self._last_action = None

    @property
    def metrics(self):
        """
        Return all the metrics managed by this driver in addition to
        the one managed by the current agent and environment.
        :return: [Summary] or None if no metrics
        """
        metrics = []
        if self.agent.metrics:
            metrics.extend(self.agent.metrics)

        if self.environment.metrics:
            metrics.extend(self.environment.metrics)

        extension_metrics = [ext.metrics for ext in self._extensions]
        for ext_metrics in extension_metrics:
            if ext_metrics:
                metrics.extend(ext_metrics)

        return metrics

    def metric_value(self, tag):
        """Returns the value of the specified metric.

        Args:
            tag -- the tag for the metric

        Returns the current metric value
        """
        for metric in self.metrics:
            if metric.tag == tag:
                return metric.value

        raise IndexError("Unknown tag: " + tag)

    @property
    def current_step(self):
        """The current step in the experiment"""
        return self._step

    @property
    def episode(self):
        """The current episode in the experiment"""
        return self._episode

    @property
    def epoch(self):
        """The current epoch in the experiment"""
        return self._epoch

    @property
    def epoch_length(self):
        """The number of steps in an epoch"""
        return self._epoch_len

    def extend(self, extension):
        """Extends the Driver by adding an extension"""
        assert isinstance(extension, DriverExtension)
        self._extensions.append(extension)

    def reset_env(self):
        """Resets the environment."""
        self._episode += 1
        self._epoch = self._step // self._epoch_len
        for ext in self._extensions:
            ext(self)

        state = self.environment.reset()
        return state

    @property
    def last_action(self):
        """The last action performed by the agent."""
        return self._last_action

    def run(self):
        """Runs the experiment."""
        try:
            pre_observation = self.reset_env()
            while self._stop_trigger(self):
                # loop
                if self._train_after == Driver.EVAL_ONLY:
                    self.agent.mode = AgentMode.Evaluation
                elif self._step < self._train_after:
                    self.agent.mode = AgentMode.Warmup
                else:
                    self.agent.mode = AgentMode.Training

                action = self.agent.act(pre_observation)

                self._last_action = action
                post_observation, reward, done, _ = self.environment.step(action)

                if self.agent.mode != AgentMode.Evaluation:
                    self.agent.observe(pre_observation, action, reward, post_observation, done)

                if done:
                    pre_observation = self.reset_env()
                    self._last_action = None
                else:
                    pre_observation = post_observation
                    # call extensions
                    for ext in self._extensions:
                        ext(self)

                self._step += 1

        finally:
            for ext in self._extensions:
                ext.finalize()


def simple_driver(agent, env, num_epochs=5, train_after=32, epoch_length=100, eval_steps=500):
    """Creates a simple driver.

    Description:
        The simple driver runs an agent through an environment and evaluates it each
        epoch.

    Args:
        env -- the environment to use
        agent -- the agent to run

    Keyword Args:
        num_epochs -- the number of epochs to run
        train_after -- the number of steps before the agent begins to train
        epoch_length -- the number of steps per epoch
        eval_steps -- the number of steps for evaluation

    Returns a simple Driver object
    """
    params = Driver.Parameters()
    params.epoch_length = epoch_length
    params.train_after = train_after
    params.stop_trigger = LimitTrigger(num_epochs, IntervalUnit.Epoch)
    evaluator = StepEvaluator(eval_steps)
    params.extensions.append(EvaluationExtension(each_epoch(), evaluator))
    return Driver(agent, env, params)
