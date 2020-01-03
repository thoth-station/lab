# thoth-lab
# Copyright(C) 2018, 2019, 2020 Marek Cermak
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

"""Utilities to work with package dependencies."""

import typing

import networkx as nx
import pandas as pd

from pandas import api

from . import underscore
from .graph import DependencyGraph


@pd.api.extensions.register_dataframe_accessor("root")
class _Root(object):
    """Root accessor."""

    def __init__(self, df):
        self._df = df

    def has_root(self):
        targets = set(self._df.target.unique())
        sources = set(self._df.source.unique())
        root_candidates = sources - targets

        return len(root_candidates) == 1

    def get_root(self, source: str = "source", target: str = "target"):
        """Check whether data are hierarchical (tree-like) with single root node."""
        targets = set(self._df.target.unique())
        sources = set(self._df.source.unique())
        root_candidates = sources - targets

        if len(root_candidates) > 1:
            raise ValueError("Multiple roots found: ", root_candidates)

        root, = root_candidates

        return root


@pd.api.extensions.register_dataframe_accessor("convert")
class _Convert(object):
    """Conversions to DataFrame representation of package dependencies."""

    def __init__(self, df):
        self._validate(df)
        self._df = df

    @staticmethod
    def _validate(df):
        """Validate."""
        if not len(df):
            raise ValueError("Empty DataFrame.")

    def to_dependency_table(
        self, root: typing.Any = None, source: str = "source", target: str = "target", inplace=False
    ):
        """Convert DataFrame to a dependency table following common schema from current dataframe.

        This method requires the dataframe to contain hierarchical data with single
        root node.
        """
        df = self._df if inplace else self._df.copy()

        df["target"] = df._.get(target)
        df["source"] = df._.get(source)

        if not root:
            # try to guess the root by missing target package
            root = df.root.get_root()

        # Create root node
        d = pd.DataFrame({"source": "", "target": root}, columns=df.columns, index=[-1])

        df = df.append(d).sort_index().reindex(sorted(df.columns), axis=1)

        if source != "source":
            df.drop(source, axis=1, inplace=True)
        if target != "target":
            df.drop(target, axis=1, inplace=True)

        return df

    def to_dependency_graph(self, root: typing.Any = None, source: str = "source", target: str = "target"):
        """Convert DataFrame to a dependency graph.

        First a dependency table is build, see `build_dependency_table` for info about required parameters.
        """
        df = self.to_dependency_table(root=root, source=source, target=target).query("source != ''")

        nodes = df.source.append(df.target).unique()
        edges = list(zip(df.source, df.target))

        g = DependencyGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        # root tree at top-level package
        tree = nx.bfs_tree(g, root or df.root.get_root(g))  # collecting is breadth first by default

        return tree
