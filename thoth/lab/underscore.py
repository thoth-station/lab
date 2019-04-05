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

import pandas as pd

from pandas import api


@pd.api.extensions.register_dataframe_accessor('_')
class _Underscore(object):

    def __init__(self, df):
        self._df: pd.DataFrame = df

    def str_join(self, cols: list = None, sep: str = ''):
        """Combine two or more columns into one joining them with separator."""
        cols = cols or self._df.columns

        if len(cols) < 2 or not all(isinstance(col, str) for col in cols):
            raise ValueError("Number of columns must be list of strings of length >= 2.")

        def stringify(s):
            return str(s) if not pd.isna(s) else None

        def safe_join(str_row: list):
            return sep.join(
                [col for col in str_row if stringify(col) is not None]
            )

        self._df = self._df[cols].apply(lambda r: safe_join(r), axis=1)
