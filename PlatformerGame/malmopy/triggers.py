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

"""Module containing various trigger functions"""

from enum import Enum, unique


@unique
class IntervalUnit(Enum):
    """Unit measured by the trigger."""

    Step = 'step'
    Episode = 'episode'
    Epoch = 'epoch'


class IntervalTrigger(object):
    """Base class for triggers which measure a unit on the driver to determine when to fire."""

    NEVER_TRIGGERED = -1

    def __init__(self, interval, unit, once_per_unit=True):
        """
        Args:
            interval -- the interval to wait
            unit -- the unit to track

        Keyword Args:
            once_per_unit -- Whether to only fire once per unit interval
        """
        assert isinstance(unit, IntervalUnit), 'unit should be IntervalUnit'

        self._previous = IntervalTrigger.NEVER_TRIGGERED
        self.once_per_unit = once_per_unit
        self.interval = max(1, interval)
        self.unit = unit

    def __call__(self, driver):
        if self.unit == IntervalUnit.Step:
            current = driver.current_step
        elif self.unit == IntervalUnit.Episode:
            current = driver.episode
        else:
            current = driver.epoch

        if current % self.interval == 0:
            if self.once_per_unit:
                if current != self._previous:
                    self._previous = current
                    return True

                return False

            return True

        return False


class LimitTrigger(object):
    """Trigger which fires until a limit is reached."""

    def __init__(self, limit, unit):
        """
        Args:
            limit -- the limit to reach
            unit -- the unit to track
        """
        self.limit = max(1, limit)
        self.unit = unit

    def __call__(self, driver):
        if self.unit == IntervalUnit.Step:
            current = driver.current_step
        elif self.unit == IntervalUnit.Episode:
            current = driver.episode
        else:
            current = driver.epoch

        return current < self.limit


class AlwaysTrigger(object):
    """Trigger which always fires."""

    def __call__(self, driver):
        return True


class NeverTrigger(object):
    """Trigger which never fires."""

    def __call__(self, driver):
        return False


class ElapsedTimeTrigger(object):
    """Trigger which fires once a certain amount of time has passed."""

    def __init__(self, seconds):
        """
        Args:
            seconds -- the number of seconds to wait
        """
        assert seconds > 0, 'seconds has to be > 0 (got %d)' % seconds

        from time import time
        self._now = time
        self._seconds = seconds
        self._limit = self._now() + seconds

    @property
    def seconds(self):
        """The number of seconds to wait."""
        return self._seconds

    @property
    def limit(self):
        """The time at which to start triggering."""
        return self._limit

    def __call__(self, driver):
        return self._now() > self._limit

    @staticmethod
    def from_minutes(minutes):
        """Creates an ElapsedTimeTrigger which waits the provided number of minutes.

        Args:
            minutes -- the number of minutes to wait

        Returns an ElapsedTimeTrigger
        """
        return ElapsedTimeTrigger(minutes * 60)

    @staticmethod
    def from_hours(hours):
        """Creates an ElapsedTimeTrigger which waits the provided number of hours.

        Args:
            hours -- the number of hours to wait

        Returns an ElapsedTimeTrigger
        """
        return ElapsedTimeTrigger.from_minutes(hours * 60)

    @staticmethod
    def from_days(days):
        """Creates an ElapsedTimeTrigger which waits the provided number of days.

        Args:
            days -- the number of days to wait

        Returns an ElapsedTimeTrigger
        """
        return ElapsedTimeTrigger.from_hours(days * 24)


def each_step():
    """Creates an interval trigger which fires each step."""
    return IntervalTrigger(1, IntervalUnit.Step)


def each_episode():
    """Creates an interval trigger which fires each episode."""
    return IntervalTrigger(1, IntervalUnit.Episode)


def each_epoch():
    """Creates an interval trigger which fires each epoch."""
    return IntervalTrigger(1, IntervalUnit.Epoch)
