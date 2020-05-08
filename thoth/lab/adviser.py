# thoth-lab
# Copyright(C) 2020 Francesco Murdaca
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

"""Adviser results processing and analysis."""

import logging
import os
import json
import sys
import hashlib
import copy

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Any

from numpy import array
from sklearn.preprocessing import LabelEncoder
from thoth.storages import AdvisersResultsStore

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

_LOGGER = logging.getLogger("thoth.lab.adviser")


def aggregate_adviser_results(adviser_version: str, limit_results: bool = False, max_ids: int = 5) -> pd.DataFrame:
    """Aggregate adviser results from jsons stored in Ceph.

    :param adviser_version: minimum adviser version considered for the analysis of adviser runs
    :param limit_results: reduce the number of inspection batch ids considered to `max_ids` to test analysis
    :param max_ids: maximum number of inspection batch ids considered
    """
    adviser_store = AdvisersResultsStore()
    adviser_store.connect()

    adviser_ids = list(adviser_store.get_document_listing())

    _LOGGER.info("Number of Adviser reports identified is: %r" % len(adviser_ids))

    adviser_dict = {}
    number_adviser_results = len(adviser_ids)
    current_a_counter = 1

    if limit_results:
        _LOGGER.debug(f"Limiting results to {max_ids} to test functions!!")

    for n, ids in enumerate(adviser_ids):
        document = adviser_store.retrieve_document(ids)
        datetime_advise_run = document["metadata"].get("datetime")
        analyzer_version = document["metadata"].get("analyzer_version")
        _LOGGER.debug(f"Analysis n.{current_a_counter}/{number_adviser_results}")
        result = document["result"]
        _LOGGER.debug(ids)
        if int("".join(analyzer_version.split("."))) >= int("".join(adviser_version.split("."))):
            report = result.get("report")
            error = result["error"]
            if error:
                error_msg = result["error_msg"]
                adviser_dict[ids] = {
                    "justification": [{"message": error_msg, "type": "ERROR"}],
                    "error": error,
                    "message": error_msg,
                    "type": "ERROR"
                }
            else:
                adviser_dict = extract_adviser_justifications(report=report, adviser_dict=adviser_dict, ids=ids)

        if ids in adviser_dict.keys():
            adviser_dict[ids]["datetime"] = datetime.strptime(datetime_advise_run, "%Y-%m-%dT%H:%M:%S.%f")
            adviser_dict[ids]["analyzer_version"] = analyzer_version

        current_a_counter += 1

        if limit_results:
            if current_a_counter > max_ids:
                return _create_adviser_dataframe(adviser_dict)

    return _create_adviser_dataframe(adviser_dict)


def _create_adviser_dataframe(adviser_data: dict):
    """Create dataframe of adviser results from data collected."""
    adviser_df = pd.DataFrame(
        adviser_data, index=["datetime", "analyzer_version", "error", "justification", "message", "type"]
    )
    adviser_df = adviser_df.transpose()
    adviser_df["date"] = pd.to_datetime(adviser_df["datetime"])

    return adviser_df


def extract_adviser_justifications(report: Dict[str, Any], adviser_dict: Dict[str, Any], ids: str) -> Dict[str, Any]:
    """Retrieve justifications from adviser results."""
    if not report:
        return adviser_dict

    products = report.get("products")
    adviser_dict = extract_justifications_from_products(products=products, adviser_dict=adviser_dict, ids=ids)

    return adviser_dict


def extract_justifications_from_products(
    products: List[Dict[str, Any]], adviser_dict: Dict[str, Any], ids: str
) -> Dict[str, Any]:
    """Extract justifications from products in adviser results."""
    if not products:
        return adviser_dict

    for product in products:
        justifications = product["justification"]
        if justifications:
            for justification in justifications:
                if "advisory" in justification:
                    adviser_dict[ids] = {
                        "justification": justification,
                        "error": True,
                        "message": justification["advisory"],
                        "type": "INFO"
                    }
                else:
                    adviser_dict[ids] = {
                        "justification": justification,
                        "error": False,
                        "message": justification["message"],
                        "type": justification["type"]
                    }

    return adviser_dict


def create_final_dataframe(adviser_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create final dataframe with all information required for plots.

    :param adviser_dataframe: data frame as returned by `aggregate_adviser_results` method.
    """
    jm_encoding = []
    for index, row in adviser_dataframe[["message"]].iterrows():
        hash_object = hashlib.sha256(bytes(row.values[0], "raw_unicode_escape"))
        hex_dig = hash_object.hexdigest()
        jm_encoding.append([index, row.values, hex_dig])

    label_encoder = LabelEncoder()
    justification_result = copy.deepcopy(adviser_dataframe.to_dict())

    jm_hash_id_values = array([pp[2] for pp in jm_encoding])
    integer_jm_hash_id_values_encoded = label_encoder.fit_transform(jm_hash_id_values)

    counter = 0
    for id_jm in integer_jm_hash_id_values_encoded:
        jm_encoding[counter] = jm_encoding[counter] + [id_jm]
        counter += 1

    justification_result["jm_hash_id_encoded"] = {el[0]: el[3] for el in jm_encoding}

    final_dataframe = pd.DataFrame(justification_result)

    return final_dataframe


def create_adviser_results_histogram(plot_df: pd.DataFrame):
    """Create inspection performance parameters plot in 3D.

    :param plot_df dataframe for plot of adviser results
    """
    plotly.offline.init_notebook_mode(connected=True)
    histogram_data = {}

    for index, row in plot_df[["jm_hash_id_encoded", "message", "type"]].iterrows():
        encoded_id = row["jm_hash_id_encoded"]
        if encoded_id not in histogram_data.keys():
            histogram_data[encoded_id] = {
                "jm_hash_id_encoded": f"type-{encoded_id}",
                "message": row["message"],
                "type": row["type"],
                "count": plot_df["jm_hash_id_encoded"].value_counts()[encoded_id]
            }

    justifications_df = pd.DataFrame(histogram_data)
    justifications_df = justifications_df.transpose()
    justifications_df = justifications_df.sort_values(by="count", ascending=False)

    X = justifications_df["jm_hash_id_encoded"]
    Y = justifications_df["count"]

    trace1 = go.Bar(
        x=X,
        y=Y,
        name="Adviser==0.7.3 justifications",
        hovertext=[y[0] for y in justifications_df[["message"]].values],
        hoverinfo="text",
        marker=dict(
            color=justifications_df["count"], colorscale="Viridis", opacity=0.8, showscale=True  # choose a colorscale
        ),
    )

    data = [trace1]

    margin = {
        "l": 0,
        "r": 0,
        "b": 0,
        "t": 0,
    }

    layout = go.Layout(
        title="Adviser justifications",
        margin=margin,
        scene=dict(xaxis=dict(title="Justification encoded ID"), yaxis=dict(title="Counter")),
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename="bar-plot")

    return justifications_df


def _aggregate_data_per_interval(adviser_justification_df: pd.DataFrame, intervals: int = 10):
    """Aggregate advise justifications per time intervals."""
    begin = min(adviser_justification_df["date"].values)
    end = max(adviser_justification_df["date"].values)
    timestamps = []
    delta = (end - begin) / intervals
    value = begin
    for i in range(1, intervals + 1):
        value = value + delta
        timestamps.append(value)
    timestamps[0] = begin
    timestamps[len(timestamps) - 1] = end

    aggregated_data = {}
    for l in range(0, len(timestamps)):
        low = timestamps[l - 1]
        high = timestamps[l]
        aggregated_data[high] = {}
        subset_df = adviser_justification_df[
            (adviser_justification_df["date"] >= low) & (adviser_justification_df["date"] <= high)
        ]

        for index, row in subset_df[["jm_hash_id_encoded", "message", "date"]].iterrows():
            encoded_id = row["jm_hash_id_encoded"]
            if encoded_id not in aggregated_data[high].keys():
                aggregated_data[high][encoded_id] = {
                    "jm_hash_id_encoded": f"type-{encoded_id}",
                    "message": row["message"],
                    "count": subset_df["jm_hash_id_encoded"].value_counts()[encoded_id]
                }

    return aggregated_data


def _create_heatmaps_values(input_data: dict, advise_encoded_type: List[int]):
    """Create values for heatmaps."""
    heatmaps_values = {}

    for advise_type in set(advise_encoded_type):
        _LOGGER.debug(f"Analyzing advise type... {advise_type}")
        type_values = []

        for upper_interval, interval_runs in input_data.items():
            _LOGGER.debug(f"Checking for that advise type in 'interval'... {upper_interval}")

            if advise_type in interval_runs.keys():
                type_values.append(interval_runs[advise_type]["count"])
            else:
                type_values.append(0)

        heatmaps_values[advise_type] = type_values

    return heatmaps_values


def create_adviser_heatmap(
    adviser_justification_df: pd.DataFrame, save_result: bool = False, output_dir: str = ""
):
    """Create adviser justifications heatmap plot.

    :param adviser_justification_df: data frame as returned by `create_final_dataframe' per identifier.
    :param save_result: resulting plots created are stored in `output_dir`.
    :param output_dir: output directory where plots are stored if `save_results` is set to True.
    """
    data = _aggregate_data_per_interval(adviser_justification_df=adviser_justification_df)
    heatmaps_values = _create_heatmaps_values(
        input_data=data, advise_encoded_type=adviser_justification_df["jm_hash_id_encoded"].values
    )
    df_heatmap = pd.DataFrame(heatmaps_values)
    df_heatmap["interval"] = data.keys()
    df_heatmap = df_heatmap.set_index(["interval"])
    df_heatmap = df_heatmap.transpose()

    adviser_justifications_map = {}
    for index, row in adviser_justification_df[["jm_hash_id_encoded", "message"]].iterrows():
        if row["jm_hash_id_encoded"] not in adviser_justifications_map.keys():
            adviser_justifications_map[row["jm_hash_id_encoded"]] = row["message"]

    justifications_ordered = []
    for index, row in df_heatmap.iterrows():

        justifications_ordered.append(adviser_justifications_map[index])

    df_heatmap["advise_type"] = justifications_ordered
    df_heatmap = df_heatmap.set_index(["advise_type"])

    plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(df_heatmap, annot=True, fmt="g")

    plt.show()

    if save_result:
        if output_dir != "":
            current_path = Path.cwd()
            project_dir_path = current_path.joinpath(output_dir)

            os.makedirs(project_dir_path, exist_ok=True)

            fig = ax.get_figure()
            fig.savefig(f"{project_dir_path}/Adviser_justifications_{datetime.now()}.png", bbox_inches="tight")

    plt.close()
    return df_heatmap
