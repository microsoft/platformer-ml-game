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

"""Module containing summaries for use in visualizers"""

import numpy as np

from .triggers import each_episode
from . import Metric, MetricKind


class Summary(object):
    """Summary object encapsulating information to be written by a visualizer."""

    _ID_GENERATOR = 0

    def __init__(self, tag=None, trigger=each_episode()):
        """
        Keyword Args:
            tag -- the name of the summary, used as a name in the visualizer
            trigger -- the trigger that determines the summary should be written
        """
        if tag is None:
            tag = '%s_%d' % (str(self.__class__), Summary._ID_GENERATOR)
            Summary._ID_GENERATOR += 1
        else:
            tag = tag.replace(' ', '_')

        self.trigger = trigger
        self.tag = tag


class ScalarSummary(Summary, Metric):
    """A summary that stores a single scalar value."""

    def __init__(self, tag=None, trigger=each_episode(), default=0.0):
        """
        Keyword Args:
            tag -- the name of the summary, used as a name in the visualizer
            trigger -- the trigger that determines the summary should be written
            default -- the default scalar value
        """
        super(ScalarSummary, self).__init__(tag, trigger)

        self._default = default
        self._value = default

    @property
    def shape(self):
        return 1

    @property
    def kind(self):
        return MetricKind.SCALAR

    def reset(self):
        self._value = self._default

    def add(self, value):
        self._value = float(value)
        return self

    @property
    def value(self):
        return self._value


class AggregationSummary(Summary, Metric):
    """Base class for summaries which are reductions of aggregated samples."""

    def __init__(self, tag=None, trigger=each_episode()):
        """
        Keyword Args:
            tag -- the name for the summary [None]
            trigger -- determines when the summary will fire [each_episode]
        """
        super(AggregationSummary, self).__init__(tag, trigger)

        self._agg = []

    @property
    def value(self):
        if not self._agg:
            return 0.0

        return self._agg_val()

    @property
    def kind(self):
        return MetricKind.SCALAR

    @property
    def shape(self):
        return 1

    def reset(self):
        self._agg = []

    def add(self, value):
        self._agg.append(float(value))
        return self

    def _agg_val(self):
        raise NotImplementedError()


class MedianSummary(AggregationSummary):
    """Summary which records the median of a list"""

    def _agg_val(self):
        return np.median(self._agg)


class MeanSummary(AggregationSummary):
    """Summary which records the mean of a list"""

    def _agg_val(self):
        return np.mean(self._agg)


class MinSummary(AggregationSummary):
    """Summary which records the minimum of a list"""

    def _agg_val(self):
        return np.amin(self._agg)


class MaxSummary(AggregationSummary):
    """Summary which records the maximum of a list"""

    def _agg_val(self):
        return np.amax(self._agg)


class SumSummary(AggregationSummary):
    """Summary which records the sum of a list"""

    def _agg_val(self):
        return np.sum(self._agg)


class VarianceSummary(AggregationSummary):
    """Summary which records the variance of a list"""

    def __init__(self, tag=None, trigger=each_episode(), stddev=False):
        super(VarianceSummary, self).__init__(tag, trigger)

        self._stddev = bool(stddev)

    def _agg_val(self):
        if self._stddev:
            return np.std(self._agg)

        return np.var(self._agg)


class ImageSummary(Summary, Metric):
    """Summary which records an image."""

    def __init__(self, width, height, tag=None, trigger=each_episode()):
        """
        Args:
            width -- the width of the image in pixels
            height -- the height of the image in pixels

        Keyword Args:
            tag -- identifier for the summary
            trigger -- how often the summary should be written
        """
        assert width > 0, "width should be > 0"
        assert height > 0, "height should be > 0"

        super(ImageSummary, self).__init__(tag, trigger)

        self._width = width
        self._height = height
        self._buffer = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def kind(self):
        return MetricKind.IMAGE

    @property
    def width(self):
        """Width of the image in pixels"""
        return self._width

    @property
    def height(self):
        """Height of the image in pixels"""
        return self._height

    def to_array(self):
        """Converts the image to a numpy array"""
        return self._buffer

    @property
    def shape(self):
        return self._buffer.shape

    def reset(self):
        self._buffer[...] = 0

    def add(self, value):
        self._buffer[...] = value

        return self

    @property
    def value(self):
        return np.copy(self._buffer)


class OnDemandImageSummary(ImageSummary):
    """Saves an image as a numpy array"""

    def __init__(self, width, height, builder, tag=None, trigger=each_episode()):
        """
        Args:
            width -- the width of the image in pixels
            height -- the height of the image in pixels

        Keyword Args:
            tag -- identifier for the summary
            trigger -- how often the summary should be written
        """
        super(OnDemandImageSummary, self).__init__(width, height, tag, trigger)
        self._builder = builder

    def add(self, value):
        raise NotImplementedError

    @property
    def value(self):
        self._buffer[...] = self._builder()
        return np.copy(self._buffer)


class HistogramSummary(Summary, Metric):
    """Summary which builds a histogram from samples."""

    def __init__(self, bins, value_range=None, tag=None, trigger=each_episode()):
        """
        Args:
            bins -- the number of bins in the histogram

        Keyword Args:
            range -- the range of the histogram values [None]
            tag -- the name for the summary [None]
            trigger -- determines when the summary will fire [each_episode]
        """
        super(HistogramSummary, self).__init__(tag, trigger)

        self._bins = bins
        self._range = value_range
        self._values = []

    @property
    def value(self):
        hist, bin_edges = np.histogram(self._values, self._bins, self._range)
        hist_dtype = np.dtype((hist.dtype, hist.shape))
        bin_edges_dtype = np.dtype((bin_edges.dtype, bin_edges.shape))
        dtype = np.dtype([('hist', hist_dtype), ('bin_edges', bin_edges_dtype)])
        value = np.array(1, dtype)
        value['hist'] = hist
        value['bin_edges'] = bin_edges
        return value

    @property
    def bins(self):
        """The number of bins in the histogram"""
        return self._bins

    @property
    def range(self):
        """The range of values in the histogram"""
        if self._range:
            return self._range

        return (np.amin(self._values), np.amax(self._values))

    @property
    def raw_values(self):
        """The raw values used to build the histogram."""
        return self._values

    @property
    def kind(self):
        return MetricKind.HISTOGRAM

    @property
    def shape(self):
        return self._bins

    def reset(self):
        self._values.clear()

    def add(self, value):
        assert np.isscalar(value)

        self._values.append(value)
        return self
