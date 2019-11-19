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

"""Module containing core visualizer classes."""

import os

import numpy as np
import h5py

from .. import Visualizer


class HDF5Visualizer(Visualizer):
    """Visualizer which stores summaries in HDF5 format with SWMR support."""

    def __init__(self, logdir, active_steps=1, flush_frequency=10):
        """
        Args:
            path -- path to the HDF5 file on disk

        Keyword Args:
            active_steps -- the number of active steps at any one time [1]
            flush_frequency -- how often to flush to disk [10]
        """
        self._steps = {}
        assert logdir is not None, "logdir cannot be None"
        assert isinstance(logdir, str), "logdir should be a string"

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self._path = os.path.join(logdir, "metrics.h5")
        self._active_steps = active_steps
        self._flush_frequency = flush_frequency
        self._dtypes = {}
        self._file = h5py.File(self._path, 'w', libver='latest')
        self._datasets = {}

    def _add_dataset(self, metric):
        value = np.array(metric.value)
        dtype = np.dtype((value.dtype, value.shape))
        self._dtypes[metric.tag] = dtype
        dataset_dtype = np.dtype([('step', np.int64), ('value', dtype)], align=True)
        dataset = self._file.create_dataset(metric.tag.replace('/', '_'), shape=(1,),
                                            dtype=dataset_dtype, chunks=True, maxshape=(None,))
        dataset.attrs['kind'] = metric.kind.name
        default = np.empty(1, dataset_dtype)
        default['step'] = -1
        dataset[0] = default
        self._datasets[metric.tag] = dataset

    def set_metrics(self, metrics):
        for metric in metrics:
            self._add_dataset(metric)

        assert not self._file.swmr_mode
        self._file.swmr_mode = True
        assert self._file.swmr_mode

    def add_summary(self, summary, step):
        if not step in self._steps:
            self._steps[step] = {}

        value = np.array(summary.value)
        dtype = np.dtype((value.dtype, value.shape))
        if summary.tag not in self._datasets:
            self._add_dataset(summary)

        if dtype != self._dtypes[summary.tag]:
            message = ("All summaries with the same tag must have the same dtype "
                       "{}: ({} != {})".format(summary.tag, dtype, self._dtypes[summary.tag]))
            raise AssertionError(message)

        self._steps[step][summary.tag] = value
        if len(self._steps) > self._active_steps + self._flush_frequency:
            self.flush()

    def _group_by_tag(self, steps):
        summaries_by_tag = {}
        for step in steps:
            summaries = self._steps[step]
            for tag in summaries:
                if tag not in summaries_by_tag:
                    summaries_by_tag[tag] = {}

                summaries_by_tag[tag][step] = summaries[tag]

        return summaries_by_tag

    def flush(self, complete=False):
        """Flushes the summaries to disk.

        Keyword Args:
            complete -- whether to flush everything (including the latest steps)
        """
        steps = list(sorted(self._steps.keys()))
        if not complete:
            steps = steps[:-self._active_steps]

        summaries_by_tag = self._group_by_tag(steps)
        for tag in summaries_by_tag:
            summaries = summaries_by_tag[tag]
            dataset = self._datasets[tag]
            index = len(dataset)
            if index == 1 and dataset[0, 'step'] == -1:
                count = len(summaries)
                index = 0
            else:
                count = index + len(summaries)

            dataset.resize((count, ))

            rows = np.empty(len(summaries), dtype=dataset.dtype)
            for i, step in enumerate(summaries):
                rows[i]['step'] = step
                rows[i]['value'] = summaries[step]

            dataset[index:] = rows
            dataset.flush()

        for step in steps:
            del self._steps[step]

    def close(self):
        self.flush(True)
        self._file.close()

    def is_supported(self, summary):
        return True
