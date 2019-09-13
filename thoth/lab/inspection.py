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

"""Inspection results processing ana analysis."""

import functools
import logging
import re

import numpy as np
import pandas as pd

import textwrap
import typing

import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py

from pandas_profiling import ProfileReport as profile
from pandas.io.json import json_normalize
from pandas.plotting import lag_plot

from prettyprinter import pformat
from IPython.display import Markdown, display

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
        xTitle="durations [ms]",
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

        annot["text"] = re.sub(r"^(.{%d}).*(.{%d})$" % (aw, aw), "\g<1>...\g<2>", text)
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

    for n, idx in enumerate(index.values):
        print("\nCategory {}/{}".format(n + 1, len(index)))
        if len(index.names) > 1:
            for name, ind in zip(index.names, idx):
                print(f"{name} :", ind)
        else:
            print(f"{index.names[0]} :", idx)

        frame = inspection_df.loc[idx]
        print("Number of rows (jobs) is:", frame.shape[0])


def columns_to_analyze(df: pd.DataFrame, low=0, high=len(df_original), 
                        display_clusters = False, cluster_by_hue = False)-> pd.DataFrame:
    """Print all columns within dataframe and count of unique column values if any fall within
    limit specified. If limit is not specified, the lower limit is 0 and the upper limit is
    size of data set so that the default function prints all columns and corresponding count of 
    unique values within each column. In addition to printing, function returns dataframe 
    with results that fall within the limits.
    
    :param df: data frame to analyze as returned by `process_inspection_results'
    :param low: the lower limit (0 if not specified) of distinct value counts
    :param high: the upper limit (size of data set if not specified) of distinct value counts
    """
    
    lst_columns_to_analyze = []
    
    #Groups every column by unique values
    printmd("#### Columns to analyze, Unique Value Count")
    for i in df:
        try:
            value_count = len(df.groupby(i).count())
            
            if ((value_count >= low) and (value_count <= high)):
                print(i, value_count)
                lst_columns_to_analyze.append(i)
        except TypeError:
            #Groups every column by unique values if values are in list or dict formats
            try:
                value_count = len(pd.Series(df[i].values).apply(tuple).unique())
                if ((value_count >= low) and (value_count <= high)):
                    print(i, value_count)
                    lst_columns_to_analyze.append(i)
            except TypeError:
                l = (list(df[i].values))
                value_count = len([i for n, i in enumerate(l) if i not in l[n + 1:]])
                if ((value_count >= low) and (value_count <= high)):
                    print(i, value_count)
                    lst_columns_to_analyze.append(i)
                pass
            
    #Filters data frame to columns with distinct value counts within the limit 
    df_analyze = df[lst_columns_to_analyze] 
    
    
    if display_clusters is True: 
        printmd("#### Inspection result count organized by parameter + parameter values")
        try:
            for i in display_jobs_by_subcategories(df_analyze):
                display(i)
        except TypeError:
            pass
    

    if cluster_by_hue is True:
        if ((low > 0) and (high < 100)):
            printmd("#### Distribution of parameters to analyze organized by hues")
            plot_subcategories_by_hues(df_analyze, df, "status__job__duration")
        else:
            printmd("##### Parameter variance too large to plot by hue/color")
    return df_analyze


def display_jobs_by_subcategories(df: pd.DataFrame):
    """Create dataframe with job counts for each subcategory for every column in the data frame.
    
    :param df: dataframe with columns of unique value counts greater than 1 as returned by 
    columns_to_analyze function with all constants filtered out.
    """
    try:
        lst = []
        #Introduce index column for job counts
        df = df.reset_index()
        for i in df.columns:
            created_values = inspection.query_inspection_dataframe(df, groupby=i)
            if not i == 'index':
                df_inspection_id= created_values.groupby([i]).count()
                df_inspection_id = df_inspection_id.filter(['index', i])
                lst.append(df_inspection_id)
        return lst
    except ValueError:
        #Error given if column values in dataframe are constant
        print("Some or all columns passed in are not distinct in values")
        
        
def duration_plots(df: pd.DataFrame):
    """Creates primary scatter plot visuals of job duration, build duration, elapsed time as well as 
    lag plots of job duration and lead time.
    
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds.
    """
    fig = plt.figure(figsize=(10,20))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)

    df = df.reset_index()
    
    #Create value lead_time as: created to status__job__started_at 
    df["lead_time"] = df["status__job__started_at"].subtract(df["created"]).dt.total_seconds()
    df = df.sort_values(by=["created"])

    #Plot job__duration from inspection results
    s_plot = df.plot.scatter(x='status__job__duration', y = 'index', c='DarkBlue', 
                             title = 'Job Duration', ax = ax1)
    s_plot.set_xlabel("status__job__duration [ms]")
    
    #Plot build__duration from inspection results
    s_plot = df.plot.scatter(x='status__build__duration',y = 'index', c='DarkBlue', 
                             title = 'Build Duration', ax = ax2)
    s_plot.set_xlabel("status__job__duration [ms]")
    
    #Plot elapsed time from inspection results
    df["job_log__stdout__@result__elapsed"] = df["job_log__stdout__@result__elapsed"]/1000
    s_plot = df.plot.scatter(x='job_log__stdout__@result__elapsed',y = 'index', c='DarkBlue', 
                             title = 'Elapsed Duration', ax = ax3)
    s_plot.set_xlabel("elapsed [ms]")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    #Plot lag plot of job duration sorted by the start of job
    df = df.sort_values(by=["status__job__started_at"])
    lag_plot(df["status__job__duration"], ax = ax4)
    ax4.set_title("Job Duration Autocorrelation (Sort by job_started)")
    
    #Plot lag plot of lead time sorted by job created
    df = df.sort_values(by=["created"])
    lag_plot(df["lead_time"], ax = ax5)
    ax5.set_title("lead_duration Autocorrelation (Sort by created)")

    #Plot lag plot of lead time sorted by job started
    df = df.sort_values(by=["status__job__started_at"])
    lag_plot(df["lead_time"], ax = ax6)
    ax6.set_title("lead_duration Autocorrelation (Sort by job_started)")

    
def plot_subcategories_by_hues(df_cat: pd.DataFrame, df: pd.DataFrame, column):
    """Create scatter plots with parameter categories separated by hues.
    
    :param df_cat: filtered dataframe with columns to analyze as returned by columns_to_analyze.
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds.
    :param colum: job duration/build duration columns from 'df'
    """
    df = df.reset_index()
    for i in df_cat:
        g = sns.FacetGrid(df, hue=i, margin_titles=True, height=5, aspect=1)
        g.map(sns.regplot, column, "index", fit_reg=False, x_jitter=.1,  scatter_kws=dict(alpha=0.5))      
        g.add_legend()
       
    
def concatenated_df(lst_of_df, column):
    """Reorganizes dataframe to show the distribution of jobs in a category across different
    sets/subsets of data.
    
    :param lst_of_df: inspection result dataframes which can be different datasets or subset of datasets
    :param column: category for grouping to see the distribution of results
    """
    lst_processed = []
    for i in lst_of_df:
        i = i.reset_index()
        df_grouping_category = i.groupby([column]).count()
        col_one = df_grouping_category["index"]
        df_col_one = pd.DataFrame(col_one)
        df_col_one = df_col_one.rename(index=str, columns={"index": "Total jobs:"+"{}".format(len(i))})
        lst_processed.append(df_col_one)
    
    df_final = pd.concat([i for i in lst_processed], axis = 1)
    return df_final


def summary_trace_plot(df: pd.DataFrame, df_categories: pd.DataFrame, lst = []):
    """Create trace plot scaled by percentage of compositions of each parameter analyzed.
    
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds.
    :param df_categories: filtered dataframe with columns to analyze as returned by columns_to_analyze.
    :param lst: list of set or subset dataframes of data with the last value in the list being 
    the entire data set.
    """
    fig = plt.figure(figsize=(15,len(df_categories.columns)*4))
    lst_df = []
    for i in df_categories.columns:
        lst_df.append((concatenated_df(lst, i)))
    lst_to_analyze = df_categories.columns
    count = 0
    for i in range(len(df_categories.columns)):
        ax = fig.add_subplot(len(df_categories.columns), 1, i+1)
        lst_df[count].apply(lambda x: x/x.sum()).transpose().plot(kind='bar', stacked=True, ax=ax, sharex = ax)
        ax.legend(title=lst_to_analyze[count], loc=9, bbox_to_anchor=(1.3,0.7), fontsize='small', fancybox=True)
        count+=1
        
        
def summary_bar_plot(df: pd.DataFrame, df_categories: pd.DataFrame, lst_of_clusters):
    """Create trace stacked plot scaled by total jobs of each parameter within clusters if any as compared 
    to a separate trace of all jobs
    
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds.
    :param df_categories:  filtered dataframe with columns to analyze as returned by columns_to_analyze.
    :param lst_of_clusters: list of set or subset dataframes of data with the last value in the list being 
    the entire data set.
    """
    fig = plt.figure(figsize=(15,len(df_categories.columns)*7))
    lst_df = [] #list of dataframes, dataframe for each cluster
        
    for i in df_categories.columns:
        lst_df.append((concatenated_df(lst_of_clusters, i)))
    
    lst_to_analyze = df_categories.columns
    count = 0

    for i in (lst_df):
        ax = fig.add_subplot(len(df_categories.columns), 1, count+1)
        ax.set_title(lst_to_analyze[count])
        colors = ["#addd8e","#fdd0a2", "#a6bddb","#7fcdbb"]
        
        if (len(lst_of_clusters)>1):
            lst_cluster = []
            for j in range(len(lst_of_clusters)-1):
                lst_cluster.append('Total jobs:{}'.format(len(lst_of_clusters[j])))
            g = lst_df[count].loc[:,lst_cluster].plot(align = 'edge', kind = "barh", stacked=True, color=colors, width = .6, ax = ax)

        g = lst_df[count].loc[:,['Total jobs:{}'.format(len(lst_of_clusters[-1]))]].plot(align = 'edge', kind = "barh", stacked=False, color="#7fcdbb", width = .3, ax = ax)
        g.legend(loc=9, bbox_to_anchor=(1.3,0.7), fontsize='small', fancybox=True)
        x_offset = 1
        y_offset = -.2
        for p in g.patches:
            b = p.get_bbox()
            val = "{0:g}".format(b.x1 - b.x0)
            if (val != '0'):
                g.annotate(val, ((b.x1) + x_offset, b.y1 + y_offset))
        count +=1

        
def plot_distribution_of_jobs_combined_categories(df: pd.DataFrame, df_duration: pd.DataFrame, df_analyze: pd.DataFrame):
    """Create trace stacked plot scaled by total jobs of each parameter within clusters if any as compared 
    to a separate trace of all jobs
    
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds.
    :param df_categories:  filtered dataframe with columns to analyze as returned by columns_to_analyze.
    :param lst_of_clusters: list of set or subset dataframes of data with the last value in the list being 
    the entire data set.
    """
    list_df_combinations = []
    for i in range(len(df)):
        ll = []
        for j in range(len(df.columns)-1):
            ll.append((df_analyze[df.columns[j]] == df.iloc[i,j]))
        df_new = df_duration[np.logical_and.reduce(ll)]
    
        list_df_combinations.append(df_new)

    fig = plt.figure(figsize=(5,4*len(df)))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    for i in range(len(df)):
        plt.subplot(len(df), 1,i+1)
        g = sns.distplot(list_df_combinations[i]["status__job__duration"], kde=True)
        g.set_title(str(df.loc[i]), fontsize = 6)

        
#Function takes in a column and prints out the feature class
def map_column_to_feature_class(column_name):
    """Helper function that maps a column in the original dataframe to a feature class as mentioned in the testing
    document.
    
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds and no columns dropped (drop=False)
    """
    
    #The keys are keywords to help associate each column with the corresponding feature class.
    software_keys = ["specification__files", "specification__packages", "specification__python__requirements"]
    script_keys = ["job_log__script", "job_log__stderr", "job_log__stdout", "specification__script"]
    hardware_keys = ["hwinfo__cpu", "platform__architecture", "platform__machine", "platform__platform",
                    "platform__processor", "platform__release", "platform__version"]
    

    if (any(x in column_name for x in hardware_keys)):
        return ("Hardware (ncpus + Platform + Processor)")
    elif(any(x in column_name for x in software_keys)):
        return ("Software stack (files, python packages, python requirements)")
    elif("specification__base" in column_name):
        return ("Base image")
    elif(any(x in column_name for x in script_keys)):
        return ("Script (script + sha256 + parameters)")
    elif("build__requests" in column_name):
        return ("Build request (hardware + memory)")
    elif("run__requests" in column_name):
        return ("Run request (hardware + memory)")
    elif("build__exit_code" in column_name):
        return ("Failure build (exit_code)")
    elif("job__exit_code" in column_name):
        return ("Failure run/job (exit_code)")
    elif("job_log__exit_code" in column_name):
        return ("Job Log (exit_code)")
    else:
        return ("Inspection Result Info")

    
def process_empty_or_mutable_parameters(df: pd.DataFrame):
    """Prints all columns with values of type dictionary/list (these values will not work with further processing
    using the groupby function). Prints the unique value count of all columns that are unhashable (all such 
    columns are constant). Drops these columns and returns a new dataframe.
    
    :param df: data frame with duration information as returned by process_inspection_results with all 
    duration columns converted into seconds and no columns dropped (drop=False).
    """
    
    #This is a list to populate with columns with no data or columns with unhashable data.
    list_of_unhashable_none_values = [] 
    for i in df:
        try:
            value_count = len(df.groupby(i).count())
            
            #Columns with no data
            if (value_count == 0):
                print(i, value_count)
                list_of_unhashable_none_values.append(i)
        except TypeError:
            #Groups every column by unique values if values are in list or dict formats
            try:
                #If values are type list checks uniqueness
                value_count = len(pd.Series(df[i].values).apply(tuple).unique())
                list_of_unhashable_none_values.append(i)
                print(i, value_count)
            except TypeError:
                #If values are type dict checks uniqueness
                l = (list(df[i].values))
                value_count = len([i for n, i in enumerate(l) if i not in l[n + 1:]])
                list_of_unhashable_none_values.append(i)
                print(i, value_count)
    return df.drop(list_of_unhashable_none_values, axis = 1)


#Function takes in dataframe
def unique_value_count_by_feature_class(df: pd.DataFrame):
    """Prints unique count values per feature/class. Prints results per feature/class that are subdivided in 
    subclasses that map to it.
    
    :param df: processed dataframe as returned by the process_empty_or_mutable_parameters.
    """
    dict_to_feature_class = {}
    list_of_features = []
    
    #Iterate through dataframe and create a dictionary of dataframe column: feature class key value pairs.
    for i in df:
        dict_to_feature_class[i] = map_column_to_feature_class(i)
        list_of_features.append(map_column_to_feature_class(i))
    
    #Get list of distinct feature class labels (this is generalized to accomodate changes made to the label)
    list_of_features = set(list_of_features)
    
    #Iterate through every feature class
    for j in list_of_features:
        #count = 0
        #Get a list of parameters that fall within the class
        list_of_parameters_per_feature = []
        for k in dict_to_feature_class:
            if (j == dict_to_feature_class[k]):
                list_of_parameters_per_feature.append(k)
                #count+=1
        
        #Groupby to get unique value count for feature class
        try:
            group = df.groupby(list_of_parameters_per_feature).size()
            printmd("#### {} {}".format(j, len(group)))
                    
            #Groupby to get unique value count for each dataframe column
            for l in list_of_parameters_per_feature:
                print(l, len(df.groupby(l).size()))
                
        except (TypeError, ValueError) as e:
            print("Parameter does not have any values. Filter these out first")

            
def dataframe_statistics(df: pd.DataFrame, plot_title):
    """Given a dataframe this function outputs a data frame with relevant statistics on job duration, build duration
    and time elapsed.
    
    :param df: data frame to analyze as returned by `process_inspection_results' with duration values in ms
    :param plot_title: title of fit plot

    """
    #Measure of skew and kurtosis for job and build duration    printmd("## Duration Statistics")
    printmd("#### Skew and kurtosis statistics")
    print("Job Duration Skew:", df["status__job__duration"].skew(axis = 0, skipna = True))
    print("Build Duration Skew:", df["status__build__duration"].skew(axis = 0, skipna = True))
    print("Job Duration Kurtosis", (df["status__job__duration"].kurt(axis = 0, skipna = True)))
    print("Build Duration Kurtosis", (df["status__build__duration"].kurt(axis = 0, skipna = True)))
    
    #Statistics for job duration, build duration, and elapsed time 
    printmd("#### Duration statistics")
    df_stats = pd.DataFrame((df["status__job__duration"].describe()))
    df_stats["status__build__duration"] = df["status__build__duration"].describe()
    df_stats["job_log__stdout__@result__elapsed"] = (df["job_log__stdout__@result__elapsed"]/1000).describe()
    display(df_stats)
    
    #Plotting of job duration and build duration with fit
    g= sns.distplot(df["status__job__duration"], kde=True);
    g.set_title("Job Duration: {}".format(plot_title))
    printmd("#### Job and build distribution plots, scatter plots, autocorrelation plots")
    plt.figure()
    g2 = sns.distplot(df["status__build__duration"], kde=True);
    g2.set_title("Build Duration: {}".format(plot_title))
    
    plt.figure()
    
    duration_plots(df_duration)

    
def printmd(string):
    """Alternate print function implementing markdown formatting
    
    :param string: string to print.
    """
    display(Markdown(string))