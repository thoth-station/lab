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

"""Pandas common operations and utilities."""

import logging
import re
import typing

import numpy as np
import pandas as pd

from pandas import api

from thoth.lab.utils import DEFAULT
from thoth.lab.utils import resolve_query
from thoth.lab.utils import rget

logger = logging.getLogger("thoth.lab.underscore")


@pd.api.extensions.register_dataframe_accessor("_")
class _Underscore(object):
    def __init__(self, df: pd.DataFrame):
        self._df: pd.DataFrame = df

    def get(self, key: str, *, default=DEFAULT, **kwargs) -> pd.Series:
        """Get record from column entries by given key."""
        attrs = key.split(".", 1)

        if len(attrs) == 2:
            col, attrs = attrs
            values = self._df[col]._.get(attrs, default=default, **kwargs)
        else:
            col, = attrs
            values = self._df[col]

        return values

    def flatten(
        self,
        col: str,
        record_paths: typing.Union[list, dict] = None,
        *,
        columns: list = None,
        drop: bool = False,
        inplace: bool = False,
        default: typing.Union[str, dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Flatten specific column of dictionaries or lists by extracting records from each entry.

        If the column contains dictionaries, they will be flatten into columns. If the column
        contains lists, they will be flatten into rows.

        A column can not contain combination of lists and dictionaries.
        """
        record_paths = record_paths or set()
        gather_record_paths = not record_paths

        # validate
        dtype = None
        for _, entry in self._df[col].iteritems():
            entry_type = type(entry)

            if pd.isna(dtype):
                dtype = entry_type
            elif entry_type is not dtype:
                raise TypeError(f"Multiple types found in column '{col}': {[dtype, entry_type]}")

            if gather_record_paths and dtype is dict:
                record_paths.update(entry.keys())

        if dtype is None:
            return self._df

        if issubclass(dtype, typing.Mapping):
            df_flat = self._df[col]._.flatten(record_paths, columns=columns, **kwargs)

        elif issubclass(dtype, (list, set, tuple)):
            stacked: pd.Series = self._df[col]._.vstack()
            df_flat = pd.DataFrame(stacked, columns=columns, **kwargs)

        else:
            raise TypeError(f"Unsupported data type: {dtype}")

        df = pd.concat([self._df, df_flat], axis=1)

        if inplace:
            self._df = df

        return df

    def hstack(self, columns: typing.Union[str, list]) -> pd.DataFrame:
        """Stack columns containing list of records horizontally."""
        if isinstance(columns, str):
            columns = [columns]

        df = self._df.copy()
        for col in columns:
            df = pd.concat([df, self._df[col]._.hstack()], axis=1)

        return df

    def vstack(self, columns: typing.Union[str, list], inplace: bool = False, **kwargs) -> pd.DataFrame:
        """Stack column containing list of records vertically.

        Note: The columns are stacked in order, please, have in mind
        that there is combinatorial expansion of those columns.

        :param kwargs: keyword arguments passed to `pd.Series._.vstack` method
        """
        if isinstance(columns, str):
            columns = [columns]

        df = self._df if inplace else self._df.copy()
        for col in columns:
            df[col] = df[col]._.vstack(**kwargs)

        return df

    def str_join(self, cols: typing.List[str] = None, *, sep: str = "") -> pd.DataFrame:
        """Combine two or more columns into one joining them with separator."""
        cols = cols or self._df.columns

        if len(cols) < 2 or not all(isinstance(col, str) for col in cols):
            raise ValueError("Number of columns must be list of strings of length >= 2.")

        def stringify(s):
            return str(s) if not pd.isna(s) else None

        def safe_join(str_row: list):
            return sep.join([col for col in str_row if stringify(col) is not None])

        return self._df[cols].apply(lambda r: safe_join(r), axis=1)

    def groupby(
        self,
        groupby: typing.Union[str, list, set] = None,
        exclude: typing.Union[str, list, set] = None,
        as_group: bool = False,
        as_index: bool = False,
        **kwargs,
    ) -> typing.Any:
        """Group DataFrame columns given column sub-strings and optionally create MultiIndex."""
        groupby = groupby or []
        exclude = exclude or []

        if isinstance(groupby, str):
            groupby = [groupby]

        if isinstance(exclude, str):
            exclude = [exclude]

        groups = []

        for key in groupby:
            columns_idx = self._df.columns.str.contains(key)
            columns = self._df.columns[columns_idx]

            if not len(columns):
                raise KeyError(f"Could NOT find suitable column given the keys: `{groupby}`")

            groups.extend(columns)

        index_groups = []

        for col in self._df[groups].columns:
            # check that the column name is not excluded
            if any(re.search(e, col) for e in exclude):
                continue

            if self._is_valid_group(self._df, col):
                index_groups.append(col)

        index_groups = pd.Series(index_groups).unique().tolist()

        # construct multi-index if grouping is requested
        group = self._df.groupby(index_groups, sort=False, **kwargs)

        if as_group:
            return group

        indices = group.indices

        levels = []
        for level, values in indices.items():
            if isinstance(level, tuple):
                levels.extend([(*level, v) for v in values])
            else:
                levels.extend([(level, v) for v in values])

        index = pd.MultiIndex.from_tuples(levels, names=[*index_groups, None])

        if as_index:
            return index

        return (
            self._df.iloc[[idx[-1] for idx in levels]]
            .set_index(index, drop=True, verify_integrity=True)
            .drop(index_groups, axis=1)
        )

    def query(
        self,
        query: str = None,
        *,
        groupby: typing.Union[str, list, set] = None,
        exclude: typing.Union[str, list, set] = None,
        like: str = None,
        regex: str = None,
        axis: int = None,
        sort_index: typing.Union[bool, int, typing.List[int]] = True,
        engine: str = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Query inspection DataFrame.

        The order of operations is as follows:

            query resolution -> grouping -> filtering

        :param inspection_df: inspection DataFrame to be filtered as returned by `process_inspection_results`
        :param groupby: column or list of columns to group the DataFrame by
        :param exclude: patterns that should be excluded from grouping
        :param query: pandas query to be evaluated on the filtered DataFrame
        :param like, regex, axis: parameters passed to the `pd.DataFrame.filter` function
        :param engine: engine to evaluate the query passed to `where` parameter, see `pd.eval` for more information

            The string provided does NOT need to match the whole column name, the function tries to determine
            the most suitable column name automatically.

        :param kwargs: additional parameters passed to the `_.groupby` function
        """
        # resolve query
        df = resolve_query(query=query, context=self._df)

        if groupby:
            df = df._.groupby(groupby=groupby, exclude=exclude, **kwargs)

        # filter
        if any([like, regex]):
            df = df.filter(like=like, regex=regex, axis=axis)

        if sort_index:
            if isinstance(sort_index, bool):
                levels = np.arange(df.index.nlevels - 1).tolist()
            else:
                levels = sort_index

            return df.sort_index(level=levels)

    @staticmethod
    def _is_valid_group(df: pd.DataFrame, groupby: typing.Union[str, typing.List[str]]) -> bool:
        """Check whether a group is valid for grouping and indexing."""
        is_valid = False
        try:
            # check that grouping is possible
            is_valid = len(df.groupby(groupby).indices) >= 1
            if not is_valid:
                logger.warning(f"Column '{groupby!s}' could NOT be used as index group. Dropped.")

        except TypeError:
            logger.warning(f"Column '{groupby!s}' dtype NOT understood. Dropped")

        return is_valid


@pd.api.extensions.register_series_accessor("_")
class _Underscore(object):
    def __init__(self, series: pd.Series):
        self._s: pd.Series = series

    def get(self, key: str, *, default=DEFAULT, **kwargs) -> pd.Series:
        """Get record from series entries by given key."""
        if default is not DEFAULT:
            kwargs.update(default=default)

        return self._s.apply(rget, args=(key,), **kwargs)

    def flatten(
        self,
        record_paths: typing.Union[str, list, dict] = None,
        *,
        columns: list = None,
        default: typing.Union[str, dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Flatten column of dictionaries by extracting records from each entry."""
        if not record_paths:
            record_paths = set()
            for idx, entry in self._s.iteritems():
                record_paths.update(entry.keys() if not pd.isna(entry) else {})

        elif isinstance(record_paths, str):
            record_paths = {record_paths: record_paths}

        elif isinstance(record_paths, (list, set, tuple)):
            record_paths = {key: key for key in record_paths}

        elif not isinstance(record_paths, typing.Mapping):
            raise TypeError("`record_paths` expected to be of type Union[list, dict], " f"got: {type(record_paths)}")

        if default is None or isinstance(default, str):
            default = {key: default for key in record_paths}

        elif not isinstance(default, dict):
            raise TypeError("`default` expected to be of type Union[str, dict], " f"got: {type(default)}")

        if columns:
            if len(record_paths) != len(columns):
                raise ValueError(
                    "Length of `columns` does not match length of `record_paths`: {} != {}".format(
                        len(columns), len(record_paths)
                    )
                )
            record_paths = {key: col_name for key, col_name in zip(record_paths, columns)}

        records = {col: [None] * len(self._s) for col in record_paths.values()}

        for idx, entry in self._s.iteritems():
            for key, col in record_paths.items():
                records[col][idx] = rget(entry, key, default=default[key])

        return pd.DataFrame(records, **kwargs)

    def hstack(self, **kwargs) -> pd.DataFrame:
        """Stack column containing list of records horizontally.

        Note, the stacking happens in sequence. If the list entries
        are of different length, it might lead to unexpected results.

        :param kwargs: keyword arguments passed to `pd.DataFrame.from_records`
        """
        return pd.DataFrame.from_records(self._s.tolist(), **kwargs)

    def vstack(self, drop_index=True, inplace=False) -> pd.Series:
        """Stack column containing list of records vertically."""
        df = pd.DataFrame.from_records(self._s.tolist())
        stacked: pd.Series = df.stack().reset_index(drop=drop_index)

        if inplace:
            self._s = stacked
            return

        return stacked
