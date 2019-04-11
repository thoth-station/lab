# thoth-lab
# Copyright(C) 2018, 2019 Marek Cermak
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

import typing
import pandas as pd

from pandas import api

from thoth.lab.utils import DEFAULT
from thoth.lab.utils import rget


@pd.api.extensions.register_dataframe_accessor("_")
class _Underscore(object):
    def __init__(self, df: pd.DataFrame):
        self._df: pd.DataFrame = df

    def str_join(self, cols: typing.List[str] = None, *, sep: str = ""):
        """Combine two or more columns into one joining them with separator."""
        cols = cols or self._df.columns

        if len(cols) < 2 or not all(isinstance(col, str) for col in cols):
            raise ValueError("Number of columns must be list of strings of length >= 2.")

        def stringify(s):
            return str(s) if not pd.isna(s) else None

        def safe_join(str_row: list):
            return sep.join([col for col in str_row if stringify(col) is not None])

        return self._df[cols].apply(lambda r: safe_join(r), axis=1)


@pd.api.extensions.register_series_accessor("_")
class _Underscore(object):
    def __init__(self, series: pd.Series):
        self._s: pd.Series = series

    def flatten(
        self, record_paths: typing.Union[list, dict], *, default: typing.Union[str, dict] = None
    ) -> pd.DataFrame:
        """Flatten column of dictionaries by extracting records from each entry."""
        if isinstance(record_paths, list):
            record_paths = {key: key for key in record_paths}
        elif not isinstance(record_paths, dict):
            raise TypeError(
                "`record_paths` expected to be of type Union[list, dict], " f"got: {type(record_paths)}")

        if default is None or isinstance(default, str):
            default = {key: default for key in record_paths}
        elif not isinstance(default, dict):
            raise TypeError(
                "`default` expected to be of type Union[str, dict], " f"got: {type(default)}")

        records = {col: [None] * len(self._s) for col in record_paths.values()}

        for idx, entry in self._s.iteritems():
            for key, col in record_paths.items():
                records[col][idx] = rget(entry, key, default=default[key])

        return pd.DataFrame(records)

    def get(self, key: str, *, default=DEFAULT, **kwargs) -> pd.Series:
        """Get record from series entries by given key."""
        if default is not DEFAULT:
            kwargs.update(default=default)

        return self._s.apply(rget, args=(key,), **kwargs)
