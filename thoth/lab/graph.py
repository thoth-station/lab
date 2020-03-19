#!/usr/bin/env python3
# thoth-lab
# Copyright(C) 2018, 2019, 2020 Fridolin Pokorny
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Various helpers and utils for interaction with the graph database."""

import asyncio
import typing

import networkx as nx
import pandas as pd

from collections import OrderedDict

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


class DependencyGraph(nx.OrderedDiGraph):
    """Construct a dependency graph by extending nx.OrderedDiGraph."""

    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict

    @staticmethod
    def get_root(tree):
        """Return root of the current graph, if any.

        By default, tree topology is considered as input,
        so if there are multiple roots, only the first one is returned.
        """
        root = None
        for node, d in tree.in_degree():
            root = node
            break

        return root


get_root = DependencyGraph.get_root
get_root.__doc__ = DependencyGraph.get_root.__doc__


class GraphQueryResult(object):
    """Wrap results of graph database queries."""

    def __init__(self, result):
        """Initialization.

        :param result: the result to be used as a query result, can be directly coroutine that is fired.
        """
        if isinstance(result, typing.Coroutine):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)

        self.result = result

    def _get_items(self):
        """Get items from the result."""
        items = list(self.result.items())
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        return labels, values

    def to_dataframe(self):
        """Construct a panda's dataframe on results."""
        return pd.DataFrame(data=self.result)

    def plot_pie(self):
        """Plot a pie of results into Jupyter notebook."""
        init_notebook_mode(connected=True)

        labels, values = self._get_items()
        trace = go.Pie(labels=labels, values=values)
        return iplot([trace])

    def plot_bar(self):
        """Plot histogram of results obtained."""
        init_notebook_mode(connected=True)

        labels, values = self._get_items()
        trace = go.Bar(x=labels, y=values)
        return iplot([trace])

    def serialize(self):
        """Serialize the output of graph query."""
        # It should be fine to just use one check for nested parts. We can extend this later on.
        def _serialize(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj

        if isinstance(self.result, list):
            return list(map(lambda x: _serialize(x), self.result))

        return _serialize(self.result)
