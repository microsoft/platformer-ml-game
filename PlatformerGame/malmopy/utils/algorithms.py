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

"""Module containing implementation of various algorithms"""

# NOTE if we need to add more graph algorithms we should instead consider a dependency on networkx

import sys
from heapq import heappush, heappop, heapify
from abc import ABC, abstractmethod

from .core import check_methods


class Node(ABC):
    """Node class for use by graph traversal algorithms"""

    @abstractmethod
    def matches(self, other):
        """Determines whether this node is the same as the other.

        Args:
            other -- another node

        Returns whether the two nodes are the same
        """

    @property
    @abstractmethod
    def key(self):
        """A unique key that describes this Node"""

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Node:
            return check_methods(class_obj, "matches", "key", "__eq__", "__hash__", "__lt__")

        return NotImplemented


class Graph(ABC):
    """Graph class for use by graph traversal algorithms"""

    __slots__ = ()

    @abstractmethod
    def neighbors(self, node):
        """Computes a list of neighbors for the node in the graph.

        Args:
            node -- the node

        Returns [Node, Node, ...] a list of neighboring nodes.
        """

    @abstractmethod
    def distance(self, node0, node1):
        """Computes the distance between two nodes. This should be symmetric.

        Args:
            node0 -- the first node
            node1 -- the second node

        Returns the cost of moving between the two nodes
        """

    @property
    @abstractmethod
    def nodes(self):
        """All of the nodes in the graph"""

    def __iter__(self):
        for node in self.nodes:
            yield node

    @classmethod
    def __subclasshook__(cls, class_obj):
        if cls is Graph:
            return check_methods(class_obj, "neighbors", "distance", "nodes", "__iter__")

        return NotImplemented


def _recover_path(start, end, prev):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = prev[current]

    path.append(start)

    return list(reversed(path))


def _add_or_update(priority_queue, priority, node):
    index = None
    for i, (_, existing_node) in enumerate(priority_queue):
        if existing_node == node:
            index = i
            break

    if index is None:
        heappush(priority_queue, (priority, node))
    else:
        priority_queue[index] = (priority, node)
        heapify(priority_queue)

