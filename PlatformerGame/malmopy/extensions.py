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

"""Module containing several Driver Extensions"""

import logging
import time
import os

from . import Evaluator, DriverExtension
from .triggers import AlwaysTrigger, NeverTrigger, IntervalTrigger, each_episode, each_epoch,\
    IntervalUnit
from .summaries import MedianSummary, ScalarSummary, SumSummary


class VisualizerExtension(DriverExtension):
    """Extension which integrates a Visualizer into the Driver"""

    def __init__(self, trigger, visualizer, name=None, reset_on_trigger=True):
        super(VisualizerExtension, self).__init__(trigger, name)

        self._visualizer = visualizer
        self._reset_on_trigger = reset_on_trigger
        self._first = True

    def _internal_call(self, driver):
        metrics = driver.metrics

        if metrics is not None:
            if self._first:
                self._visualizer.set_metrics(metrics)
                self._first = False

            # Plot metrics
            for metric in metrics:
                if metric.trigger(driver):
                    if self._visualizer.is_supported(metric):
                        self._visualizer.add_summary(metric, driver.current_step)
                    else:
                        logging.warning("%s is not supported by %s",
                                        type(metric).__name__,
                                        type(self._visualizer).__name__)

                    if self._reset_on_trigger:
                        metric.reset()

    def finalize(self):
        self._visualizer.close()


class WritePngsExtension(DriverExtension):
    """Extension which incorporates writing PNG images into the Driver."""

    def __init__(self, trigger, output_dir, name=None):
        super(WritePngsExtension, self).__init__(trigger, name)

        self._output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self._frame_no = 0
        self._last_episode = -1

    def _internal_call(self, driver):
        from PIL import Image
        assert "rgb_array" in driver.environment.metadata["render_modes"]
        rgb = driver.environment.render("rgb_array")
        img = Image.fromarray(rgb, "RGB")

        if self._last_episode != driver.episode:
            self._frame_no = 0
            self._last_episode = driver.episode
        fname = "ep{:03d}_step{:05d}.png".format(driver.episode, self._frame_no)
        self._frame_no += 1
        img.save(os.path.join(self._output_dir, fname))


class CheckpointExtension(DriverExtension):
    """Extension which integrates agent checkpoint saving into the Driver."""

    def __init__(self, trigger, output_dir, name=None):
        super(CheckpointExtension, self).__init__(trigger, name)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _internal_call(self, driver):
        path = os.path.join(self.output_dir, "checkpoint_{}.npz".format(driver.epoch))
        logging.info('Creating checkpoint at %s', path)
        driver.agent.save(path)


class ThroughputExtension(DriverExtension):
    """Extension which adds throughput measurement to the Driver."""

    def __init__(self, trigger):
        super(ThroughputExtension, self).__init__(trigger)
        self._previous_step = 0
        self._previous_time = time.perf_counter()
        self._throughput = MedianSummary(
            'Diagnostics/Median steps per second', trigger=each_episode())

    def _internal_call(self, driver):
        current_time = time.perf_counter()
        step_diff = driver.current_step - self._previous_step
        time_diff = current_time - self._previous_time

        self._previous_step = driver.current_step
        self._previous_time = current_time

        # print('Current throughput: %d step/s' % (step_diff / time_diff))
        self._throughput.add(step_diff / time_diff)

    @property
    def metrics(self):
        return self._throughput


class VisualiseNumActionsExtension(DriverExtension):
    """
    Extension which adds visualization of the numbers if different discrete actions to the Driver
    """

    def __init__(self, action_name_map):
        super(VisualiseNumActionsExtension, self).__init__(AlwaysTrigger())

        self._actions_map = {}
        for key in action_name_map:
            self._actions_map[key] = SumSummary(
                'Actions/Num_' + action_name_map[key] + '_Actions', trigger=each_episode())

    def _internal_call(self, driver):
        try:
            action = driver.last_action
            if (action is not None) and action in self._actions_map:
                self._actions_map[action].add(1)

        except AttributeError as ex:
            print(
                'Warning: driver has no last_action attribute - will not try to count actions.')
            print('(' + ex + ')')
            self.trigger = NeverTrigger()

    @property
    def metrics(self):
        return [self._actions_map[k] for k in self._actions_map]


class EvaluationExtension(DriverExtension):
    """Adds model evaluation after epoch into the Driver"""

    def __init__(self, trigger, evaluator: Evaluator):
        """

        :param trigger: must be each_epoch
        :param evaluator: Evaluator object (with a recorder)
        """
        assert isinstance(trigger, IntervalTrigger)
        assert trigger.unit == IntervalUnit.Epoch

        super(EvaluationExtension, self).__init__(trigger, name='Evaluation')
        self._evaluator = evaluator
        assert isinstance(self._evaluator, Evaluator)

        if trigger.unit == IntervalUnit.Epoch:
            new_trigger = each_epoch()
        else:
            raise ValueError(trigger)
        self._metrics = [ScalarSummary(type(
            evaluator).__name__+"/"+name, new_trigger) for name in evaluator.get_metric_names()]

    def _internal_call(self, driver):
        metric_values = self._evaluator.run(driver.epoch, driver.environment, driver.agent)
        for metric in driver.metrics:
            metric.reset()

        assert len(metric_values) == len(
            self._metrics), "Mismatch in the number of values returned"
        for metric, metric_value in zip(self._metrics, metric_values):
            metric.add(metric_value)

    @property
    def metrics(self):
        return self._metrics
