"""Various helpers and utils for interaction with the graph database."""

import asyncio
import typing

import pandas as pd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go


class GraphQueryResult(object):
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
