# thoth-lab
# Copyright(C) 2018, 2019 Marek Cermak, Francesco Murdaca
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

"""Inspection results processing and analysis."""

import functools
import logging
import re

import numpy as np
import pandas as pd

import textwrap
import typing

import cufflinks as cf
import plotly
import plotly.offline as py

from pandas_profiling import ProfileReport as profile
from pandas.io.json import json_normalize

from prettyprinter import pformat

from typing import Any, Dict, List, Tuple, Union
from typing import Callable, Iterable

from plotly import graph_objs as go
from plotly import figure_factory as ff
from plotly import tools

from thoth.lab.utils import group_index

logger = logging.getLogger("thoth.lab.inspection")

# cufflinks should be in offline mode
cf.go_offline()


def extract_structure_json(input_json: dict, upper_key: str, level: int, json_structure):
    """Convert a json file structure into a list with rows showing tree depths, keys and values."""
    level += 1
    for key in input_json.keys():
        if type(input_json[key]) is dict:
            json_structure.append([level, upper_key, key, [k for k in input_json[key].keys()]])

            extract_structure_json(input_json[key], f"{upper_key}__{key}", level, json_structure)
        else:
            json_structure.append([level, upper_key, key, input_json[key]])

    return json_structure


def extract_keys_from_dataframe(df: pd.DataFrame, key: str):
    """Filter the specific dataframe created for a certain key, combination of keys or for a tree depth."""
    if type(key) is str:
        available_keys = set(df["Current_key"].values)
        available_combined_keys = set(df["Upper_keys"].values)

        if key in available_keys:
            ndf = df[df["Current_key"].str.contains(f"^{key}$", regex=True)]

        elif key in available_combined_keys:
            ndf = df[df["Upper_keys"].str.contains(f"{key}$", regex=True)]
        else:
            print("The key is not in the json")
            ndf = "".join(
                [
                    f"The available keys are (WARNING: Some of the keys have no leafs):{available_keys} ",
                    f"The available combined keys are: {available_combined_keys}",
                ]
            )
    elif type(key) is int:
        max_depth = df["Tree_depth"].max()
        if key <= max_depth:
            ndf = df[df["Tree_depth"] == key]
        else:
            ndf = f"The maximum tree depth available is: {max_depth}"
    return ndf


def process_inspection_results(
    inspection_results: List[dict],
    exclude: Union[list, set] = None,
    apply: List[Tuple] = None,
    drop: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Process inspection result into pd.DataFrame."""
    if not inspection_results:
        return ValueError("Empty iterable provided.")

    datetime_spec = ("created|started_at|finished_at", pd.to_datetime)
    if apply is None:
        apply = [datetime_spec]
    else:
        apply = [*apply, datetime_spec]

    exclude = exclude or []
    apply = apply or ()

    df = json_normalize(inspection_results, sep="__")  # each row resembles InspectionResult

    if len(df) <= 1:
        return df

    for regex, func in apply:
        for col in df.filter(regex=regex).columns:
            df[col] = df[col].apply(func)

    keys = [k for k in inspection_results[0] if k not in exclude]
    for k in keys:
        if k in exclude:
            continue
        d = df.filter(regex=k)
        p = profile(d)

        rejected = (
            p.description_set["variables"]
            .query("distinct_count <= 1 & type != 'UNSUPPORTED'")
            .filter(regex="^((?!version).)*$", axis=0)
        )  # explicitly include versions

        if verbose:
            print("Rejected columns: ", rejected.index)

        if drop:
            df.drop(rejected.index, axis=1, inplace=True)

    df = df.eval(
        "status__job__duration   = status__job__finished_at   - status__job__started_at", engine="python"
    ).eval("status__build__duration = status__build__finished_at - status__build__started_at", engine="python")

    return df


def create_duration_dataframe(inspection_df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics and duration DataFrame."""
    if len(inspection_df) <= 0:
        raise ValueError("Empty DataFrame provided")

    try:
        inspection_df.drop("build_log", axis=1, inplace=True)
    except KeyError:
        pass

    data = (
        inspection_df.filter(like="duration")
        .rename(columns=lambda s: s.replace("status__", "").replace("__", "_"))
        .apply(lambda ts: pd.to_timedelta(ts).dt.total_seconds())
    )

    def compute_duration_stats(group):
        return (
            group.eval("job_duration_mean      = job_duration.mean()", engine="python")
            .eval("job_duration_upper_bound    = job_duration + job_duration.std()", engine="python")
            .eval("job_duration_lower_bound    = job_duration - job_duration.std()", engine="python")
            .eval("build_duration_mean         = build_duration.mean()", engine="python")
            .eval("build_duration_upper_bound  = build_duration + build_duration.std()", engine="python")
            .eval("build_duration_lower_bound  = build_duration - build_duration.std()", engine="python")
        )

    if isinstance(inspection_df.index, pd.MultiIndex):
        n_levels = len(inspection_df.index.levels)

        # compute duration stats for each group separately
        data = data.groupby(level=list(range(n_levels - 1)), sort=False).apply(compute_duration_stats)
    else:
        data = compute_duration_stats(data)

    return data.round(4)


def create_duration_box(data: pd.DataFrame, columns: Union[str, List[str]] = None, **kwargs):
    """Create duration Box plot."""
    columns = columns if columns is not None else data.filter(regex="duration$").columns

    figure = data[columns].iplot(
        kind="box", title=kwargs.pop("title", "InspectionRun duration"), yTitle="duration [s]", asFigure=True
    )

    return figure


def create_duration_scatter(data: pd.DataFrame, columns: Union[str, List[str]] = None, **kwargs):
    """Create duration Scatter plot."""
    columns = columns if columns is not None else data.filter(regex="duration$").columns

    figure = data[columns].iplot(
        kind="scatter",
        title=kwargs.pop("title", "InspectionRun duration"),
        yTitle="duration [s]",
        xTitle="inspection ID",
        asFigure=True,
    )

    return figure


def create_duration_scatter_with_bounds(
    data: pd.DataFrame, col: str, index: Union[list, pd.Index, pd.RangeIndex] = None, **kwargs
):
    """Create duration Scatter plot with upper and lower bounds."""
    df_duration = (
        data[[col]]
        .eval(f"upper_bound = {col} + {col}.std()", engine="python")
        .eval(f"lower_bound = {col} - {col}.std()", engine="python")
    )

    index = index if index is not None else df_duration.index

    if isinstance(index, pd.MultiIndex):
        index = index.levels[-1] if len(index.levels[-1]) == len(data) else np.arange(len(data))

    upper_bound = go.Scatter(
        name="Upper Bound",
        x=index,
        y=df_duration.upper_bound,
        mode="lines",
        marker=dict(color="lightgray"),
        line=dict(width=0),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    trace = go.Scatter(
        name="Duration",
        x=index,
        y=df_duration[col],
        mode="lines",
        line=dict(color="rgb(31, 119, 180)"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    lower_bound = go.Scatter(
        name="Lower Bound",
        x=index,
        y=df_duration.lower_bound,
        marker=dict(color="lightgray"),
        line=dict(width=0),
        mode="lines",
    )

    data = [lower_bound, trace, upper_bound]
    m = df_duration[col].mean()

    layout = go.Layout(
        yaxis=dict(title="duration [s]"),
        xaxis=dict(title="inspection ID"),
        shapes=[
            {"type": "line", "x0": 0, "x1": len(index), "y0": m, "y1": m, "line": {"color": "red", "dash": "longdash"}}
        ],
        title=kwargs.pop("title", "InspectionRun duration"),
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


def create_duration_histogram(data: pd.DataFrame, columns: Union[str, List[str]] = None, bins: int = None, **kwargs):
    """Create duration Histogram plot."""
    columns = columns if columns is not None else data.filter(regex="duration$").columns

    if not bins:
        bins = np.max([np.lib.histograms._hist_bin_auto(data[col].values, None) for col in columns])

    figure = data[columns].iplot(
        title=kwargs.pop("title", "InspectionRun distribution"),
        yTitle="count",
        xTitle="durations [s]",
        kind="hist",
        bins=int(np.ceil(bins)),
        asFigure=True,
    )

    return figure


def query_inspection_dataframe(inspection_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Wrapper around _.query method which always include `duration` columns in filter expression."""
    like = kwargs.pop("like", None)
    regex = kwargs.pop("regex", None)

    if like is not None:
        df = inspection_df._.query(*args, like=like, regex=regex, **kwargs)

        if not any(df.columns.str.contains("duration")):
            # duration columns must be present
            df = df.join(inspection_df.filter(like="duration"))

        return df

    elif regex is not None:
        regex += "|(.*duration)"

    return inspection_df._.query(*args, like=like, regex=regex, **kwargs)


def make_subplots(data: pd.DataFrame, columns: List[str] = None, *, kind: str = "box", **kwargs):
    """Make subplots and arrange them in an optimized grid layout."""
    if kind not in ("box", "histogram", "scatter", "scatter_with_bounds"):
        raise ValueError(f"Can NOT handle plot of kind: {kind}.")

    index = data.index.droplevel(-1).unique()

    if len(index.names) > 2:
        logger.warning(f"Can only handle hierarchical index of depth <= 2, got {len(index.names)}. Grouping index.")

        return make_subplots(group_index(data, range(index.nlevels - 1)), columns, kind=kind, **kwargs)

    grid = ff.create_facet_grid(
        data.reset_index(),
        facet_row=index.names[1] if index.nlevels > 1 else None,
        facet_col=index.names[0],
        trace_type="box",  # box does not need data specification
        ggplot2=True,
    )

    shape = np.shape(grid._grid_ref)[:-1]

    sub_plots = tools.make_subplots(
        rows=shape[0],
        cols=shape[1],
        shared_yaxes=kwargs.pop("shared_yaxes", True),
        shared_xaxes=kwargs.pop("shared_xaxes", False),
        print_grid=kwargs.pop("print_grid", False),
    )

    if isinstance(index, pd.MultiIndex):
        index_grid = zip(*index.labels)
    else:
        index_grid = iter(
            np.transpose([np.tile(np.arange(shape[1]), shape[0]), np.repeat(np.arange(shape[0]), shape[1])])
        )

    for idx, grp in data.groupby(level=np.arange(index.nlevels).tolist()):
        if not isinstance(columns, str) and kind == "scatter_with_bounds":
            if columns is None:
                raise ValueError("`scatter_with_bounds` requires `col` argument, not provided.")
            try:
                columns, = columns
            except ValueError:
                raise ValueError("`scatter_with_bounds` does not allow for multiple columns.")

        fig = eval(f"create_duration_{kind}(grp, columns, **kwargs)")

        col, row = map(int, next(index_grid))  # col-first plotting
        for trace in fig.data:
            sub_plots.append_trace(trace, row + 1, col + 1)

    layout = sub_plots.layout
    layout.update(
        title=kwargs.get("title", fig.layout.title),
        shapes=grid.layout.shapes,
        annotations=grid.layout.annotations,
        showlegend=False,
    )

    x_dom_vals = [k for k in layout.to_plotly_json().keys() if "xaxis" in k]
    y_dom_vals = [k for k in layout.to_plotly_json().keys() if "yaxis" in k]

    layout_shapes = pd.DataFrame(layout.to_plotly_json()["shapes"]).sort_values(["x0", "y0"])

    h_shapes = layout_shapes[~layout_shapes.x0.duplicated(keep=False)]
    v_shapes = layout_shapes[~layout_shapes.y0.duplicated(keep=False)]

    # handle single-columns
    h_shapes = h_shapes.query("y1 - y0 != 1")
    v_shapes = v_shapes.query("x1 - x0 != 1")

    # update axis domains and layout
    for idx, x_axis in enumerate(x_dom_vals):
        x0, x1 = h_shapes.iloc[idx % shape[1]][["x0", "x1"]]

        layout[x_axis].domain = (x0 + 0.03, x1 - 0.03)
        layout[x_axis].update(showticklabels=False, zeroline=False)

    for idx, y_axis in enumerate(y_dom_vals):
        y0, y1 = v_shapes.iloc[idx % shape[0]][["y0", "y1"]]

        layout[y_axis].domain = (y0 + 0.03, y1 - 0.03)
        layout[y_axis].update(zeroline=False)

    # correct annotation to match the relevant group and width
    annot_df = pd.DataFrame(layout.to_plotly_json()["annotations"]).sort_values(["x", "y"])
    annot_df = annot_df[annot_df.text.str.len() > 0]

    aw = min(  # annotation width magic
        int(max(60 / shape[1] - (2 * shape[1]), 6)), int(max(30 / shape[0] - (2 * shape[0]), 6))
    )

    for i, annot_idx in enumerate(annot_df.index):
        annot = layout.annotations[annot_idx]

        index_label: Union[str, Any] = annot["text"]
        if isinstance(index, pd.MultiIndex):
            index_axis = i >= shape[1]
            if shape[0] == 1:
                pass  # no worries, the order and label are aight
            elif shape[1] == 1:
                index_label = index.levels[index_axis][max(0, i - 1)]
            else:
                index_label = index.levels[index_axis][i % shape[1]]

        text: str = str(index_label)

        annot["text"] = re.sub(r"^(.{%d}).*(.{%d})$" % (aw, aw), "\g<1>...\g<2>", text)  # Ignore PycodestyleBear (W605)
        annot["hovertext"] = "<br>".join(pformat(index_label).split("\n"))

    # add axis titles as plot annotations
    layout.annotations = (
        *layout.annotations,
        {
            "x": 0.5,
            "y": -0.05,
            "xref": "paper",
            "yref": "paper",
            "text": fig.layout.xaxis["title"]["text"],
            "showarrow": False,
        },
        {
            "x": -0.05,
            "y": 0.5,
            "xref": "paper",
            "yref": "paper",
            "text": fig.layout.yaxis["title"]["text"],
            "textangle": -90,
            "showarrow": False,
        },
    )

    # custom user layout updates
    user_layout = kwargs.pop("layout", None)
    if user_layout:
        layout.update(user_layout)

    return sub_plots


def show_categories(inspection_df: pd.DataFrame):
    """List categories in the given DataFrame."""
    index = inspection_df.index.droplevel(-1).unique()

    results_categories = {}
    for n, idx in enumerate(index.values):
        print("\nClass {}/{}".format(n + 1, len(index)))
        
        class_results = {}
        if len(index.names) > 1:
            for name, ind in zip(index.names, idx):
                print(f"{name} :", ind)
                class_results[name] = ind
        else:
            print(f"{index.names[0]} :", idx)
            class_results[index.names[0]] = idx
        results_categories[n + 1] = class_results

        frame = inspection_df.loc[idx]
        print("Number of rows (jobs) is:", frame.shape[0])
        
    return results_categories


def filter_inspection_list(ids_list: List[str], identifier_list: List[str]) -> dict: 
    """Filter inspection ids list according to the identifier selected"""
    filtered_list_ids = {}

    for identifier in identifier_list:
        filtered_list_ids[identifier] = []

    for ids in ids_list:
        inspection_filter = "-".join(ids.split("-")[1:(len(ids.split("-")) - 1)])

        if inspection_filter:
            if inspection_filter in identifier_list:
                filtered_list_ids[inspection_filter].append(ids)

    tot_inspections_selected = sum([len(batch_n) for batch_n in filtered_list_ids.values()])
    inspection_batches = [(batch_name, len(batch_count)) for batch_name, batch_count in filtered_list_ids.items()]
    print(f"There are {tot_inspections_selected} inspection runs!: {inspection_batches} respectively")
    
    return filtered_list_ids


def create_inspection_analysis_plots(df_duration_batches: pd.DataFrame, batch_identifier: str):
    "Create inspection analysis plots per batch"
    # Box plots job duration and build duration
    fig = inspection.create_duration_box(df_duration_batches[batch_identifier], ["build_duration", "job_duration"])

    py.iplot(fig)
    # Scatter job duration
    fig = inspection.create_duration_scatter(
        df_duration_batches[batch_identifier], "job_duration", title="InspectionRun job duration"
    )

    py.iplot(fig)
    # Scatter build duration
    fig = inspection.create_duration_scatter(
        df_duration_batches[batch_identifier], "build_duration", title="InspectionRun build duration"
    )

    py.iplot(fig)
    # Histogram
    fig = inspection.create_duration_histogram(df_duration_batches[batch_identifier], ["job_duration"])

    py.iplot(fig)


def create_scatter_plots_for_multiple_batches(data: pd.DataFrame,
                                  columns: Union[str, List[str]] = None,
                                  list_batches: List[str] = [], 
                                  title_scatter: str = "Scatter plot",
                                  x_label: str = " ", 
                                  y_label: str = " "):
    """Create Scatter plots for multiple batches."""
    columns = columns if columns is not None else data[columns].columns

    figure = {
    'data': [
        {
            'x': data[columns[0] + '_' + batch],
            'y': data[columns[1] + '_' + batch],
            'name': batch, 'mode': 'markers',
        } for batch in list_batches
    ],
    'layout': {
        'title': title_scatter,
        'xaxis': {'title': x_label},
        'yaxis': {'title': y_label}
        }
    }
    
    return figure


def evaluate_statistics(df_batch: pd.DataFrame, parameter_name: str) -> Dict:
    """Evaluate statistical quantities for plots"""
    cv = df_batch[parameter_name].std()/df_batch[parameter_name].mean()*100
    std_error = df_batch[parameter_name].std()/np.sqrt(df_batch[parameter_name].shape[0])
    std = df_batch[parameter_name].std()
    median = df_batch[parameter_name].median()
    q = df_batch[parameter_name].quantile([.25, .75])
    q1 =  q[0.25]
    q3 =  q[0.75]
    IQR = q3 - q1
    maxr = df_batch[parameter_name].max()
    minr = df_batch[parameter_name].min()

    return {"cv": cv,
            "std_error": std_error,
            "std": std,
            "median": median,
            "q1":q1,
            "q3": q3,
            "iqr": IQR,
            "max": maxr,
            "min": minr}


def evaluate_statistics_per_parameter_per_batch(dict_of_df_batches: dict, parameter: str) -> dict:
    """Aggregate results of statistics per parameter from different inspection batches"""
    results_statistics = {}
    for identifier in identifier_inspection:
        results_statistics[identifier] = evaluate_statistics(
            df_batch=dict_of_df_batches[identifier],
            parameter_name=parameter)
        
    results = {}
    
    for quantity in results_statistics[identifier].keys():
        results[quantity] = [values[quantity] for values in results_statistics.values()]
        
    return results


def plot_batches_statistics_interpolated_single_parameter(parameter_selected: str,
                                                          colour_list: List,
                                                          quantities: List,
                                                          title_ylabel: str = " "):
    """Plot interpolated statistics for different batches for a specific parameter"""
    if len(colour_list) != len(quantities):
        logger.exception(f"List of statistical quantities and List of colours need to have the same length!")
    i = 0
    parameter_results = dftotal_statistics[parameter_selected]
    for quantity in quantities:
        plt.plot(identifier_inspection, parameter_results[quantity], f'{colour[i]}o-', label=f'{quantity}')
        i += 1
    plt.xlabel('Batch Identifier')
    plt.ylabel(title_ylabel)
    plt.title(f'Statistics plot for {parameter_selected} of different batch')
    plt.tick_params(axis='x', rotation=45)
    plt.legend()
    plt.show()


def plot_batches_statistics_interpolated_multiple_parameters(parameters_selected: List,
                                                              colour_list: List,
                                                              quantity: str,
                                                              title_ylabel: str = " "):
    """Plot single interpolated statistics for different batches for multiple parameters"""
    if len(colour_list) != len(parameters_selected):
        logger.exception(f"List of parameters and List of colours need to have the same length!")
    i = 0
    for parameter in parameters_selected:
        parameter_results = dftotal_statistics[parameter]
        plt.plot(identifier_inspection, parameter_results[quantity], f'{colour[i]}o-', label=f'{parameter}')
        i += 1
    plt.xlabel('Batch Identifier')
    plt.ylabel(title_ylabel)
    plt.title(f'Statistics plot for {quantity} of different batch for different parameters')
    plt.tick_params(axis='x', rotation=45)
    plt.legend()
    plt.show()

# General functions

def create_scatter_and_correlation(data: pd.DataFrame,
                                  columns: Union[str, List[str]] = None,
                                  title_scatter: str = "Scatter plot"):
    """Create Scatter plot and evaluate correlation coefficients."""
    columns = columns if columns is not None else data[columns].columns

    figure = data[columns].iplot(
        kind="scatter",
        x=columns[0],
        y=columns[1],
        title=title_scatter,
        xTitle=columns[0],
        yTitle=columns[1],
        mode='markers',
        asFigure=True
    )
    
    for correlation_type in ["pearson", "spearman", "kendall"]:
        correlation_matrix = data[columns].corr(correlation_type)
        print(f"\n{correlation_type} correlation results:\n{correlation_matrix}")
    
    return figure


def create_box_plot(data: pd.DataFrame,
               columns: Union[str, List[str]] = None,
               title_box: str = "Box plot",
               y_label: str = "Variable [Measurement Unit]",
               static: str = False):
    """Create duration Box plot."""
    columns = columns if columns is not None else data[columns].columns     
        
    figure = data[columns].iplot(
        kind="box", title=title_box, yTitle=y_label, asFigure=True
    )

    return figure


