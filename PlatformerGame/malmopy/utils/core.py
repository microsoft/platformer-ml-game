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

"""Module containing utilities functions and classes"""

from multiprocessing.sharedctypes import RawArray
from ctypes import c_float, c_double, c_uint
import numpy as np

NUMPY_TO_C_DTYPE = {np.float32: c_float,
                    np.float64: c_double, np.uint32: c_uint}


def check_methods(class_obj, *methods):
    """Check whether the listed methods are implemented in the provided class.

    Args:
        class_obj - the class object
        methods - a list of method names

    Returns:
        either True (if the methods are implemented) or NotImplemented
    """
    mro = class_obj.__mro__
    for method in methods:
        for base_class_obj in mro:
            if method in base_class_obj.__dict__:
                if base_class_obj.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


def get_rank(shape):
    """Get a shape's rank

    Args:
        shape -- a shape structure (either ndarray or tuple)

    Result:
        the rank of the shape structure
    """
    if isinstance(shape, np.ndarray):
        return shape.ndim

    if isinstance(shape, tuple):
        return len(shape)

    return ValueError('Unable to determine rank of type: %s' % str(type(shape)))


def _assert_range_error_message(left, right, range_min, range_max):
    return "Range condition violated: {} <= {} <= {} <= {}".format(range_min, left,
                                                                   right, range_max)


def assert_range(left_right, min_max):
    """Check that the values fall within the range.

    Description:
        Verifies the property min <= left <= right <= max

    Args:
        left_right -- (left, right). If None, this check passes automatically.
        min_max -- (min, max)

    Returns:
        a valid range, either (left, right) or (min, max)
    """
    if not left_right:
        return min_max

    left, right = left_right
    range_min, range_max = min_max
    if not right:
        right = range_max
    elif right < 0:
        right += range_max

    assert range_min <= left <= right <= range_max, _assert_range_error_message(left, right,
                                                                                range_min,
                                                                                range_max)
    return left, right


def check_rank(shape, required_rank):
    """Check if the shape's rank equals the expected rank

    Args:
        shape -- a shape structure (either ndarray or tuple)
        required_rank -- the required rank for the stucture

    Returns:
        True if at required rank, false otherwise
    """
    return get_rank(shape) == required_rank


def allocate_shared_from_buffer(buffer):
    """Allocate a shared array for multi-process communication.

    Args:
        buffer -- the buffer to use for initializing the shared array

    Return:
        a numpy array which uses shared memory and is initialized to the value of `buffer`
    """
    if buffer.dtype.type not in NUMPY_TO_C_DTYPE:
        raise NotImplementedError(
            'The specified dtype (%s) is not supported' % buffer.dtype)

    shared = RawArray(NUMPY_TO_C_DTYPE[buffer.dtype.type], buffer.ravel())
    return np.frombuffer(shared, buffer.dtype).reshape(buffer.shape)
