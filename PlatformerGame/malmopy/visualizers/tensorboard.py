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

"""Module containing a visualizer built on top of TensorBoard"""

import os

import tensorflow as tf
import numpy as np

from .. import Visualizer, MetricKind


class TensorboardVisualizer(Visualizer):
    """
    Visualize the generated results in Tensorboard
    """

    def __init__(self, logdir, graph=None):
        super(TensorboardVisualizer, self).__init__()
        gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.01)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        assert logdir is not None, "logdir cannot be None"
        assert isinstance(logdir, str), "logdir should be a string"

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if graph is None:
            graph = self._session.graph

        self._train_writer = tf.summary.FileWriter(logdir=logdir, graph=graph, flush_secs=30)

    def set_metrics(self, metrics):
        pass

    def _scalar(self, metric, step=0):
        """Log a scalar summary"""

        summary = tf.Summary(value=[
            tf.Summary.Value(tag=metric.tag, simple_value=metric.value)
        ])
        self._add_event(summary, step)

    def _image(self, metric, step=0):
        from PIL import Image
        from io import BytesIO
        output = BytesIO()
        img = Image.fromarray(metric.value, mode='RGB')
        img.save(output, format="png")
        contents = output.getvalue()
        output.close()
        image = tf.Summary.Image(
            height=img.height,
            width=img.width,
            colorspace=3,
            encoded_image_string=contents
        )
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=metric.tag, image=image)
        ])
        self._add_event(summary, step)

    def _histogram(self, metric, step=0):
        values = np.array(metric.raw_values)
        histo = tf.HistogramProto(
            min=np.amin(values),
            max=np.amax(values),
            num=len(values),
            sum=np.sum(values),
            sum_squares=np.sum(values**2),
            bucket_limit=metric.value['bin_edges'][1:].astype(np.float64),
            bucket=metric.value['bucket'].astype(np.float64)
        )
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=metric.tag, histo=histo)
        ])
        self._add_event(summary, step)

    def is_supported(self, summary):
        return summary.kind != MetricKind.DISTRIBUTION

    def close(self):
        if self._train_writer is not None:
            self._train_writer.close()

        self._session.close()

    def add_summary(self, summary, step):
        if summary.kind == MetricKind.SCALAR:
            self._scalar(summary, step)
        elif summary.kind == MetricKind.HISTOGRAM:
            self._histogram(summary, step)
        elif summary.kind == MetricKind.IMAGE:
            self._image(summary, step)
        else:
            raise ValueError

    def _add_event(self, summary, index=0):
        if summary is not None:
            self._train_writer.add_summary(summary, global_step=index)
