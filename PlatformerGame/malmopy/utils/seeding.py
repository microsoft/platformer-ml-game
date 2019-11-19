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

"""Module providing utility functions for seeding random number generators."""

import os
import numpy as np

def np_random(seed=None):
    """Creates a pseudo-random number generator (an instance of numpy.random.RandomState)

    Keyword Args:
        seed -- seed value. If None, it will be generated automatically. [None]

    Returns prng, seed
    """
    seed = create_seed(seed)

    assert isinstance(seed, int), "seed must be an integral type"

    prng = np.random.RandomState()
    prng.seed(seed)
    return prng, seed


def create_seed(seed=None):
    """Create a seed for a random number generator.


    Keyword Args:
        seed -- initial seed value. If None, it will be generated automatically. [None]

    Returns:
        a random number generator seed
    """
    if seed is None:
        random_bytes = os.urandom(4)
        seed = 0
        for byte in random_bytes:
            seed = (seed << 1) + byte
    else:
        seed = seed & 0xFFFFFFFF

    return seed
