# thoth-lab
# Copyright(C) 2019, 2020 Francesco Murdaca
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

"""Inspection report generation and visualization."""

import logging
import re

import pandas as pd

from typing import Any, Dict, List, Tuple, Union
from thoth.lab import inspection
from thoth.lab import underscore
from thoth.lab.exception import NotUniqueValues

from IPython.core.display import HTML


logger = logging.getLogger("thoth.lab.inspection_report")

_INSPECTION_REPORT_FEATURES = {
    "hardware": ["platform", "processor", "ncpus"],
    "software_stack": ["requirements_locked"],
    "base_image": ["base_image"],
    "pi": ["script"],
    "exit_codes": ["exit_code"],
}

_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING = {
    "platform": ["hwinfo__platform"],
    "processor": ["cpu_type__is", "cpu_type__has"],
    "ncpus": ["hwinfo__cpu_type__ncpus"],
    "requirements_locked": ["requirements_locked__default", "requirements_locked___meta"],
    "base_image": ["os_release__name", "os_release__version"],
    "script": ["script", "script_sha256", "stdout__component", "@parameters"],
    "exit_code": ["exit_code"],
}


def multi_table(table_dict):
    """Accept a list of IpyTable objects and return a table which contains each IpyTable in a cell."""
    return HTML(
        '<table><br style="background-color:white;">'
        + "".join(["<br>" + table._repr_html_() + "</br>" for table in table_dict.values()])
        + "</br></table>"
    )


def create_df_report(df: pd.DataFrame) -> pd.DataFrame:
    """Show unique values for each column in the dataframe."""
    dataframe_report = {}
    for c_name in df.columns.values:
        try:
            unique_values = df[c_name].unique()
            dataframe_report[c_name] = [unique_values]
        except Exception as exc:
            logger.info(exc)
            dataframe_report[c_name] = [df[c_name].values]
            pass
    df_unique = pd.DataFrame(dataframe_report)
    return df_unique


def create_dfs_inspection_classes(inspection_df: pd.DataFrame) -> dict:
    """Create all inspection dataframes per class with unique values and complete values."""
    class_inspection_dfs = {}
    class_inspection_dfs_unique = {}

    for class_inspection, class_features in _INSPECTION_REPORT_FEATURES.items():

        class_inspection_dfs[class_inspection] = {}
        class_inspection_dfs_unique[class_inspection] = {}

        if len(class_features) > 1:

            for feature in class_features:

                if len(feature) > 1:
                    class_df = inspection_df[
                        [
                            col
                            for col in inspection_df.columns.values
                            if any(c in col for c in _INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[feature])
                        ]
                    ]
                    class_inspection_dfs[class_inspection][feature] = class_df

                    class_df_unique = create_df_report(class_df)
                    class_inspection_dfs_unique[class_inspection][feature] = class_df_unique
                else:
                    class_df = inspection_df[
                        [
                            col
                            for col in inspection_df.columns.values
                            if _INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[feature] in col
                        ]
                    ]
                    class_inspection_dfs[class_inspection][feature] = class_df

                    class_df_unique = create_df_report(class_df)
                    class_inspection_dfs_unique[class_inspection][feature] = class_df_unique

        elif len(_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[class_features[0]]) > 1:

            class_df = inspection_df[
                [
                    col
                    for col in inspection_df.columns.values
                    if any(c in col for c in _INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[class_features[0]])
                ]
            ]
            class_inspection_dfs[class_inspection] = class_df

            class_df_unique = create_df_report(class_df)
            class_inspection_dfs_unique[class_inspection] = class_df_unique

        else:
            class_df = inspection_df[
                [
                    col
                    for col in inspection_df.columns.values
                    if _INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[class_features[0]][0] in col
                ]
            ]
            class_inspection_dfs[class_inspection] = class_df

            class_df_unique = create_df_report(class_df)
            class_inspection_dfs_unique[class_inspection] = class_df_unique

    return class_inspection_dfs, class_inspection_dfs_unique
