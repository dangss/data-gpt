# -*- coding: utf-8 -*-
# +
## Basic
import pandas as pd
import pandas.io.formats.style
import numpy as np
import os
import unidecode
import polars as pl
import re
import itertools
from traceback import format_exc

## Location
import shapely
from keplergl import KeplerGl
import geopandas as gpd

import pygwalker as pyg
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html

## Pandas API
from pandas.api.types import (
    is_categorical_dtype,
    is_object_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_datetime64_any_dtype,
)

## Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

## Streamlit
import streamlit as st
import streamlit.components.v1 as components

## Random
import random
from common.app_utils import *

## zeda2 ##
from zeda2.common_report.block import *
from zeda2 import describe
# +
pio.templates.default = "simple_white"

BAR_COLOR = '#1F77B4'
CONSTANT_COLOR = 'rgb(250,128,114)'
LINE_COLOR = 'rgb(13,152,186)'
BAD_COLOR = '#CD0404'

NUM_TYPES = ['uint32', 'uint64', 'int64', 'float64', 'int32', 'float32', 'Int32', 'Float64']
DATETIME_TYPES = ['datetime64[ns]']

WARNING_RED = '#CC0000'
WARNING_ORANGE = '#b36200'
WARNING_GREEN = '#037f51'

WIDTH = 1000
HEIGHT = 600

REPORT_HEIGHT = 2500

LOCATION_FEATURES = ['tinh', 'huyen', 'xa']
COMMON_PATH = f"{os.path.dirname(__file__)}"

# +
def get_dtype_feature(column):
    # check numeric
    if is_integer_dtype(column):
        return "numeric"
    if is_float_dtype(column):
        return "numeric"
    if is_datetime64_any_dtype(column):
        return "datetime"
    if is_categorical_dtype(column) or is_object_dtype(column):
        try:
            pd.to_datetime(column)
            return "datetime"
        except:
            return "categorical"

def detect_timestamp_columns(dataframe):
    timestamp_columns = []
    for column in dataframe.columns:
        try:
            # Attempt to convert the column to datetime
            pd.to_datetime(dataframe[column])
            # If no error is raised, it is a timestamp column
            timestamp_columns.append(column)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # If conversion fails, it's not a timestamp column
            pass
    return timestamp_columns

def convert_data_to_html(df, is_styled=False, class_name:str="dataframe-table"):
    """HTML table with pagination and other goodies"""
    if not is_styled:
        df_html = df.to_html().replace('border="1"','border="0"').replace("<table", f'<table class="{class_name}"')
    else:
        df_html = df.replace("<table", f'<table class="{class_name}"')

    df_html = f'''
    <div class="table-container">
    {df_html}
    </div>
    '''

    base_html = """
    <!doctype html>
    <html><head>
    <meta http-equiv="Content-type" content="text/html; charset=utf-8">
    <script type="text/javascript" src="https://code.jquery.com/jquery-1.11.3.min.js"></script>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.5/css/dataTables.dataTables.css">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/3.0.2/css/buttons.dataTables.css">
    <script type="text/javascript" src="https://cdn.datatables.net/2.0.5/js/dataTables.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/3.0.2/js/dataTables.buttons.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/3.0.2/js/buttons.dataTables.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/3.0.2/js/buttons.html5.min.js"></script>
    <style>
    .table-container {
      width: 95vw;
      max-height: 600px;
      overflow: auto;
    }
    table {
        width: 100% !important;
        font-size: 12px !important;
    }
    </style>
    </head>
    <body>
    """ + df_html + """
    <script type="text/javascript">
    $(document).ready(function(){
    $('.""" + class_name + """').DataTable({
        "pageLength": 5,
        "layout": {
                "topStart": {
                    "buttons": [{
                    extend: 'csv',
                    text: 'Download data',
                    filename: 'data'
                    }]
                }
            }
    });
    });
    </script>
    </body></html>
    """

    return base_html

# +
## Chart Visualize ##
def cate_chart(df, feature, agg_feature=None, unique_keys=[], total_for_percent=None,
               orientation='auto', max_cate=10, order_index=False):
    """
    A function providing table and chart for categorical variable.

    Parameters
    ----------
    df: pandas.Dataframe
        A dataframe contains desired variable.
    feature: str
        The name of variable to present.
    agg_feature: str (default is None)
        The feature that you want to calculate on. If None, it is the "total_count" by default.
    unique_keys: list (default is None)
        Drop duplicates on (unique_keys, feature) before performing value_counts
    unique_feature: str (default is None)
        If unique_feature is provided, instead of performing value_count
    total_for_percent: int (default is None)
        The total value that the "percent" column is based on. If None, it is the total of "total_count" or agg_feature by default.
    orientation: str (default is "auto")
        If auto, the feature is presented on X-axis if the avg length of category's names is less than 15.
        If v, the feature is presented on X-axis;
        Otherwise, if h, the feature is presented on Y-axis.
    max_cate: int (default 5)
        The maximum number of categories will be displayed on chart, the other categories will be grouped into "Others"
    order_index: bool (default False)
        If False, the chart will be descending ordered by values.
        If True, it will be ordered ascending by the feature's categories.

    Returns
    ----------
    A dictionary having format {
        df_categorical: a Pandas DataFrame of variable's categorical statistics
        df_numeric: a Pandas DataFrame of variable value_count statistics if there are more than 50 categories
        fig: A Plotly go.Figure()
    }

    """

    common_config = {
        "xaxis": dict(
            tickmode='auto',       # Set tick mode to auto
            automargin=True        # Automatically adjust margins to prevent overlap
        ),
        "yaxis": dict(
            tickmode='auto',
            automargin=True,
        ),
        "margin": dict(
            l=50,  # Left margin
            r=50,  # Right margin
            b=100,  # Bottom margin
            t=100  # Top margin
        )
    }

    time_col = detect_timestamp_columns(df[[feature]])
    if len(time_col) > 0: # percentage not apply for time category
        return_prop_fig = False
    else:
        return_prop_fig = True

    if agg_feature is None:
        if len(unique_keys) > 0:
            df_series = df.drop_duplicates(subset=unique_keys + [feature])
        else:
            df_series = df.copy()

        df_series = df_series[feature].value_counts(dropna=False).reset_index()
        df_series.columns = [feature, 'total_count']
        df_series['percent'] = (df_series['total_count'] / df_series['total_count'].sum())
        count_var = 'total_count'

    else:
        df_series = df.copy()
        count_var = agg_feature

    if total_for_percent is not None:
        df_series['percent'] = df_series[count_var] / total_for_percent
    else:
        df_series['percent'] = df_series[count_var] / df_series[count_var].sum()

    columns_ord = [count_var, 'percent']
    df_series = df_series.set_index(feature)[columns_ord]

    df_origin = df_series.copy()
    df_series = df_series.sort_values("percent", ascending=False)

    have_other = False
    if len(df_series.index) > max_cate:
        others_series = df_series.iloc[max_cate - 2:, :].sum()
        others_df = pd.DataFrame([others_series.values], index=["Others"],
                                 columns=others_series.index)[columns_ord]
        df_series = pd.concat([df_series.iloc[:max_cate - 2, :].sort_index(ascending=True), others_df], axis=0)
        have_other = True

    #### ORDER INDEX ####
    if order_index or df_origin.index.inferred_type in ['integer', 'floating', 'datetime64', 'mixed-integer']:
        if have_other:
            df_integer = df_series.loc[df_series.index != 'Others', :]
            df_integer = df_integer.sort_index(ascending=True)
            df_series = pd.concat([df_integer, df_series.loc[df_series.index == 'Others', :]])
        else:
            df_series = df_series.sort_index(ascending=True)
    else:
        df_series = df_series.sort_values(count_var, ascending=False)
        if have_other:
            indexes = [idx for idx in df_series.index if idx != "Others"]
            df_series = df_series.loc[indexes + ["Others"], :]

    #### Numeric statistics for multiple categories
    if len(df_origin.index) > 50:
        df_num_cat = describe.describe_1d_numeric_with_plot(df_origin, count_var, return_table=True)
    else:
        df_num_cat = None

    #### Move null index to the last
    null_index = [idx for idx in df_series.index if pd.isna(idx) or idx is None]

    colors = [BAR_COLOR] * len(df_series.index)
    if len(null_index) > 0:
        df_series = df_series.loc[[x for x in df_series.index if pd.notna(x) and x is not None] + null_index, :]
        colors[-1] = BAD_COLOR

    avg_str_len = np.mean([len((str(x))) for x in df_origin.index])

    ## Round the value if it is float
    if is_float_dtype(df_series[count_var]):
        df_series[count_var] = np.round(df_series[count_var], 2)

    bar = go.Figure()
    text_list = list(map(lambda x: f"{x[0]}<br>{x[1]:.2f}%", list(
        zip(np.round(df_series[count_var], 2) if is_float_dtype(df_series[count_var]) else df_series[count_var],
            df_series['percent'] * 100))))

    if (orientation == 'h') or (orientation == 'auto' and avg_str_len >= 10):
        bar.add_trace(go.Bar(y=df_series.index.map(str)[::-1], x=df_series[count_var][::-1], text=text_list[::-1],
                             textposition='outside', marker_color=colors[::-1], name="total records",
                             orientation="h"))
        bar.update_xaxes(title_text=count_var)
        bar.update_layout(
            xaxis_range=[0, df_series[count_var].max() + df_series[count_var].max() * 0.2]
        )

    else:
        bar.add_trace(go.Bar(x=df_series.index.map(str), y=df_series[count_var], text=text_list,
                             textposition='outside', marker_color=colors, name="total records"))
        bar.update_xaxes(title_text=feature)
        bar.update_yaxes(title_text=count_var)

        bar.update_layout(
            yaxis_range=[0, df_series[count_var].max() + df_series[count_var].max() * 0.2],
            **common_config
        )

    unique_str = f',<br>unique by {unique_keys}' if len(unique_keys) > 0 else ""
    bar.update_layout(title_text=f"{count_var} of <b>{feature}</b>" + unique_str,
                      **common_config
                      )

    if total_for_percent is None:
        total_concat = pd.Series(df_series[[count_var, 'percent']].sum())
    else:
        total_concat = pd.Series([total_for_percent, df_series['percent'].sum()])

    final_df = pd.concat([df_series, pd.DataFrame([total_concat.values],
                                                  columns=total_concat.index,
                                                  index=["Total"])])
    final_df['percent'] = (np.round(final_df['percent'], 3) * 100).apply(lambda x: "{:.2f}{}".format(x, ' %'))
    if count_var == 'total_count':
        final_df[count_var] = final_df[count_var].astype(int)

    final_df.index.name = feature

    df_series = df_series.sort_values(count_var, ascending=False)

    prop_fig = None
    if return_prop_fig:
        if len(df_series) < 5:
            prop_fig = px.pie(names=df_series.index.map(str),
                              values=df_series[count_var],
                              color_discrete_sequence=px.colors.qualitative.Vivid)
        else:
            prop_fig = px.treemap(names=df_series.index.map(str),
                                  parents=[''] * len(df_series),
                                  values=df_series[count_var],
                                  color_discrete_sequence=px.colors.qualitative.Vivid)

        prop_fig.update_layout(title_text=f"Proportion of <b>{feature}</b>" + unique_str, **common_config)

    return {
        'df_categorical': df_series,
        'df_numeric': df_num_cat,
        'abs_fig': bar,
        'prop_fig': prop_fig,
        'viz_function': '1d',
        '1d_type': 'cate'
    }

def describe_3d_category_category_numeric_with_plot(df,
                                                    num_col, cate_col_1, cate_col_2, aggfunc='sum', **kwargs):

    plot_size = {
        "width": WIDTH,
        "height": HEIGHT,
    }

    entry_height = 50
    padding = 150

    min_height = 400
    max_height = 600

    plot_legend = {
        "hoverlabel": dict(
            font_size=10,
        ),
        "legend_traceorder" : "reversed",
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        "hovermode" : 'x unified',
    }

    plot_legend_time = {
        "hoverlabel": dict(
            font_size=10, bgcolor="white", font=dict(color="black")
        ),
        "legend_traceorder": "reversed",
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        "hovermode": 'x'
    }

    df[cate_col_1] = df[cate_col_1].astype(str)
    df[cate_col_2] = df[cate_col_2].astype(str)

    timestamp_cols = detect_timestamp_columns(df[[cate_col_1, cate_col_2]])
    time_col = None
    if cate_col_1 in timestamp_cols:
        time_col = cate_col_1
    if cate_col_2 in timestamp_cols:
        time_col = cate_col_2

    all_time_list = []
    if time_col is not None : all_time_list = sorted(df[time_col].unique())

    num_col_name = f"{num_col}_{aggfunc}"
    num_col_percentage = f"{num_col}_percentage"

    df = df.groupby([cate_col_1, cate_col_2]).agg(
        **{num_col_name: (num_col, aggfunc)}
    ).reset_index()

    df = df.sort_values(num_col_name, ascending=False)

    plot_data = df[[num_col_name, cate_col_1, cate_col_2]].dropna(how='any').copy()
    plot_data[[cate_col_1, cate_col_2]] = plot_data[[cate_col_1, cate_col_2]].astype('category')

    color_col = cate_col_1 if plot_data[cate_col_1].nunique() < plot_data[cate_col_2].nunique() else cate_col_2
    x_col = cate_col_2 if color_col == cate_col_1 else cate_col_1

    total_x_value = plot_data.groupby([x_col])[num_col_name].transform('sum')
    plot_data[num_col_percentage] = (plot_data[num_col_name] / total_x_value) * 100

    data_vis = {}
    avg_str_len = np.mean([len((str(x))) for x in df[x_col].unique()])
    plot_legend['margin'] = dict(l=avg_str_len*15)
    chart_types = ['stack'] if avg_str_len > 10 else ['stack', 'area']
    side_charts = ['group'] if plot_data[color_col].nunique() < 5 else []
    x_col_ordered = sorted(plot_data[x_col].unique())

    # absolute value
    for chart_type in chart_types + side_charts:
        nb_entries = df[color_col].nunique() if avg_str_len <= 10 else df[x_col].nunique()
        plot_size["height"] = min(max_height, max(min_height, padding + entry_height*nb_entries))
        plot_data[color_col] = plot_data[color_col].astype(str)
        plot_data = plot_data.sort_values([x_col, color_col])

        fig = None
        if chart_type in ['stack', 'group']:
            traces = []
            if avg_str_len > 10:
                for color_value in sorted(plot_data[color_col].unique()):
                    subset_df = plot_data[plot_data[color_col] == color_value]
                    trace = go.Bar(y=subset_df[x_col], x=subset_df[num_col_name], name=color_value, orientation='h')
                    traces.append(trace)
                fig = go.Figure(data=traces)
                plot_legend["hovermode"] = 'y unified'
                plot_legend['legend_traceorder'] = 'normal'
                fig.update_yaxes(categoryorder='array', categoryarray=x_col_ordered)
            else:
                for color_value in sorted(plot_data[color_col].unique()):
                    subset_df = plot_data[plot_data[color_col] == color_value]
                    trace = go.Bar(x=subset_df[x_col], y=subset_df[num_col_name], name=color_value)
                    traces.append(trace)
                fig = go.Figure(data=traces)
                fig.update_xaxes(categoryorder='array', categoryarray=x_col_ordered)

            fig.update_layout(barmode=chart_type, **plot_legend, **plot_size)

        elif chart_type == 'area':
            fig = px.area(df, x=x_col, y=num_col_name, color=color_col)
            fig.update_xaxes(categoryorder='array', categoryarray=x_col_ordered)
            fig.update_layout(**plot_legend, **plot_size)

        data_vis[f'{chart_type}_abs'] = fig

    # time changes
    if time_col is not None:
        other_col = cate_col_1 if time_col == cate_col_2 else cate_col_2
        plot_size["height"] =  min(max_height, max(min_height, padding + entry_height * len(plot_data[other_col].unique())))
        plot_data = plot_data.sort_values([time_col, other_col], ascending=True)
        shifted = plot_data.groupby([other_col])[num_col_name].shift(1)
        plot_data['time_delta_changes'] = (plot_data[num_col_name] - shifted)
        plot_data['time_percent_changes'] = (plot_data['time_delta_changes'] / shifted) * 100

        try:
            fig = px.line(plot_data, x=time_col, y=num_col_name, color=other_col, markers=True, line_shape='spline')
        except:
            fig = px.line(plot_data, x=time_col, y=num_col_name, color=other_col, markers=True)

        fig.update_xaxes(categoryorder='array', categoryarray=all_time_list)
        fig.update_layout(
            xaxis_type='category',
            **plot_legend_time,
            **plot_size,
        )
        data_vis[f'time_abs'] = fig

        # try:
        #     fig = px.line(plot_data, x=time_col, y='time_delta_changes', color=other_col, markers=True, line_shape='spline')
        # except:
        fig = px.line(plot_data, x=time_col, y='time_delta_changes', color=other_col, markers=True)
        fig.update_xaxes(categoryorder='array', categoryarray=all_time_list)
        fig.update_layout(
            xaxis_type='category',
            **plot_legend_time,
            **plot_size,
        )
        data_vis[f'time_delta_changes'] = fig

        # try:
        #     fig = px.line(plot_data, x=time_col, y='time_percent_changes', color=other_col, markers=True, line_shape='spline')
        # except:
        fig = px.line(plot_data, x=time_col, y='time_percent_changes', color=other_col, markers=True)
        fig.update_xaxes(categoryorder='array', categoryarray=all_time_list)
        fig.update_layout(
            xaxis_type='category',
            **plot_legend,
            **plot_size
        )
        data_vis[f'time_percent_changes'] = fig

    # percent value
    for chart_type in chart_types:
        nb_entries = df[color_col].nunique() if avg_str_len <= 10 else df[x_col].nunique()
        plot_size["height"] = min(max_height, max(min_height, padding + entry_height*nb_entries))
        df = plot_data.sort_values([x_col, color_col])
        if chart_type == 'stack':
            traces = []
            if avg_str_len > 10:
                for color_value in sorted(df[color_col].unique()):
                    subset_df = df[df[color_col] == color_value]
                    trace = go.Bar(y=subset_df[x_col], x=subset_df[num_col_percentage], name=color_value, orientation='h')
                    traces.append(trace)
                fig_percent = go.Figure(data=traces)
                fig_percent.update_yaxes(categoryorder='array', categoryarray=x_col_ordered)
                plot_legend["hovermode"] = 'y unified'
                plot_legend['legend_traceorder'] = 'normal'
            else:
                for color_value in sorted(df[color_col].unique()):
                    subset_df = df[df[color_col] == color_value]
                    trace = go.Bar(x=subset_df[x_col], y=subset_df[num_col_percentage], name=color_value)
                    traces.append(trace)
                fig_percent = go.Figure(data=traces)
                fig_percent.update_xaxes(categoryorder='array', categoryarray=x_col_ordered)
            fig_percent.update_layout(barmode=chart_type, **plot_legend, **plot_size)
        elif chart_type == 'area':
            fig_percent = px.area(plot_data, x=x_col, y=num_col_percentage, color=color_col)
            fig_percent.update_xaxes(categoryorder='array', categoryarray=x_col_ordered)
            fig_percent.update_layout(**plot_legend, **plot_size)

        data_vis[f'{chart_type}_percent'] = fig_percent

    return data_vis

def describe_3d_with_plot(df, col_1, col_2, col_3):
    col_1_type = get_dtype_feature(df[col_1])
    col_2_type = get_dtype_feature(df[col_2])
    col_3_type = get_dtype_feature(df[col_3])

    type_list = [(col_1, col_1_type), (col_2, col_2_type), (col_3, col_3_type)]
    all_types = {
        'categorical': [c[0] for c in type_list if c[1] in ['categorical', 'datetime'] or df[c[0]].nunique() < 10],
        'numeric': [c[0] for c in type_list if c[1] == 'numeric' and df[c[0]].nunique() >= 10],
    }

    if len(all_types['categorical']) == 2 and len(all_types['numeric']) == 1:
        return describe_3d_category_category_numeric_with_plot(df,
                                                               *all_types['numeric'],
                                                               *all_types['categorical']
                                                               )
    return None

def get_feature_distribution(df, col_name, return_html=True):
    dtype_feature = get_dtype_feature(df[col_name])
    is_category = dtype_feature in ['categorical', 'datetime']
    if not is_category:
        df[col_name] = df[col_name].astype(str)
        vis = describe.describe_1d_numeric_with_plot(df, col_name, return_raw=True, plot_trim_kwargs={"title": "", "width": WIDTH-200, "height": HEIGHT-100, "margin": dict(t=200)})
    else:
        vis = describe.describe_1d_category_with_plot(df, col_name, return_raw=True, plot_kwargs={"title": "", "width": WIDTH-200, "height": HEIGHT-100, "margin": dict(t=200)})

    vis_component = ZVStack([ZHStack([comp for comp in vis if comp is not None], styles="font-size: 12px !important;")])
    if return_html:
        return export_html(vis_component)
    return vis_component

def get_report_1d(df:pd.DataFrame, analyzed_columns:list=[], title="1D Visualization", return_html=False):
    full_1d_components = {}

    if len(analyzed_columns) == 0:
        analyzed_columns = df.columns

    for col_name in analyzed_columns:
        try:
            full_1d_components[col_name] = get_feature_distribution(df, col_name, return_html=False)
        except:
            pass

    report_component = ZVStack([ZTitle(title), ZSelectDropdown(full_1d_components)])
    if return_html:
        return report_component.to_html(return_string=True)

    return report_component

def get_report_2d(df:pd.DataFrame, analyzed_columns:list=[], title="2D Visualization", return_html=False):
    ## TODO: make this report with 2 selection
    full_2d_components = dict()

    if len(analyzed_columns) == 0:
        analyzed_columns = df.columns

    all_comb2 = itertools.combinations(analyzed_columns, 2)
    for (col_1, col_2) in all_comb2:
        col_1_type = get_dtype_feature(df[col_1])
        col_2_type = get_dtype_feature(df[col_2])
        if col_1_type == 'numeric' and col_2_type == 'numeric':
            vis = describe.describe_2d_numeric_with_plot(df, col_1, col_2, plot_type='scatter', return_raw=True, plot_kwargs={"title":"", "width": WIDTH, "height": HEIGHT, "margin": dict(t=200)})
        elif col_1_type == 'numeric' and col_2_type in ['categorical', 'datetime']:
            df[col_2] = df[col_2].astype(str)
            if df[col_2].nunique() < df.shape[0]: # Each category has more than 1 value
                vis = describe.describe_2d_numeric_category_with_plot(df,
                                                             num=col_1,
                                                             cate=col_2,
                                                             return_raw=True,
                                                             threshold=20, plot_box_kwargs={"title":"", "width": WIDTH, "height": HEIGHT, "margin": dict(t=200)},
                                                             )
            else:
                vis = [obj for obj_name, obj in cate_chart(df, feature=col_2, agg_feature=col_1).items() if 'df' in obj_name or 'fig' in obj_name]
        elif col_1_type in ['categorical', 'datetime'] and col_2_type == 'numeric':
            df[col_1] = df[col_1].astype(str)
            if df[col_1].nunique() < df.shape[0]: # Each category has more than 1 value
                vis = describe.describe_2d_numeric_category_with_plot(df,
                                                             num=col_2,
                                                             cate=col_1,
                                                             return_raw=True,
                                                             threshold=20, plot_box_kwargs={"title":"", "width": WIDTH, "height": HEIGHT, "margin": dict(t=200)},
                                                             )
            else:
                vis = [obj for obj_name, obj in cate_chart(df, feature=col_1, agg_feature=col_2).items() if 'df' in obj_name or 'fig' in obj_name]

        elif col_1 in ['categorical', 'datetime'] and col_2_type in ['categorical', 'datetime']:
            df[col_1] = df[col_1].astype(str)
            df[col_2] = df[col_2].astype(str)
            vis = describe.describe_2d_category_with_plot(df, cate1=col_1, cate2=col_2, return_raw=True,
                                                 plot_count_kwargs={"title":"", "width": WIDTH, "height": HEIGHT, "margin": dict(t=200)},
                                                 plot_freq_kwargs={"title":"", "width": WIDTH, "height": HEIGHT, "margin": dict(t=200)})
        else:
            continue

        figures = [comp for comp in vis if comp is not None and isinstance(comp, go.Figure)]
        tables = [comp for comp in vis if comp is not None and isinstance(comp, pd.DataFrame)]
        tabs_2d = dict()
        if len(figures) > 0:
            tabs_2d['Figures'] = ZHStack(figures)
        if len(tables) > 0:
            tabs_2d['Table'] = ZHStack(tables)

        if len(tabs_2d.keys()) > 0:
            full_2d_components[f"{col_1} & {col_2}"] = ZVStack([
                "<hr>",
                ZTab(tabs_2d)
            ])

    if len(full_2d_components.keys()) > 0:
        final_report = ZVStack([ZTitle(title), ZSelectDropdown(full_2d_components)])
        if return_html:
            return final_report.to_html(return_string=True)

        return final_report

    return None

@st.cache_data(ttl=60*60)
def get_report_2d_on_target(df:pd.DataFrame, target_feature:str, predictors:list=None, title="2D Visualization on TARGET", return_html=False):
    full_2d_components = {}

    if predictors is None:
        predictors = df.columns.tolist()

    target_type = get_dtype_feature(df[target_feature])
    for predictor in predictors:
        if target_feature == predictor: continue
        predictor_type = get_dtype_feature(df[predictor])
        if target_type == 'numeric' and predictor_type == 'numeric':
            vis = describe.describe_2d_numeric_with_plot(df,
                                                predictor,
                                                target_feature, plot_type='scatter', return_raw=True, plot_kwargs={"width": WIDTH, "height": HEIGHT, "margin": dict(t=200)})
        elif target_type == 'numeric' and predictor_type in ['categorical', 'datetime']:
            df[predictor] = df[predictor].astype(str)
            vis = describe.describe_2d_numeric_category_with_plot(df,
                                                         num=target_feature,
                                                         cate=predictor,
                                                         return_raw=True,
                                                         threshold=20, plot_box_kwargs={"width": WIDTH, "height": HEIGHT, "margin": dict(t=200)},
                                                         )
        elif target_type in ['categorical', 'datetime'] and predictor_type == 'numeric':
            df[target_feature] = df[target_feature].astype(str)
            vis = describe.describe_2d_numeric_category_with_plot(df,
                                                         num=predictor,
                                                         cate=target_feature,
                                                         return_raw=True,
                                                         threshold=20, plot_box_kwargs={"width": WIDTH, "height": HEIGHT, "margin": dict(t=200)},
                                                         )

        elif target_type in ['categorical', 'datetime'] and predictor_type in ['categorical', 'datetime']:
            df[predictor] = df[predictor].astype(str)
            df[target_feature] = df[target_feature].astype(str)
            vis = describe.describe_2d_category_with_plot(df, cate1=predictor, cate2=target_feature, return_raw=True,
                                                 plot_freq_kwargs={"width": WIDTH, "height": HEIGHT, "margin": dict(t=200)},
                                                 plot_count_kwargs={"width": WIDTH, "height": HEIGHT, "margin": dict(t=200)})
        else:
            continue

        figures = [comp for comp in vis if comp is not None and isinstance(comp, go.Figure)]
        tables = [comp for comp in vis if comp is not None and isinstance(comp, pd.DataFrame)]
        tabs_2d = {}
        if len(figures) > 0:
            tabs_2d['Figures'] = ZHStack(figures)
        if len(tables) > 0:
            tabs_2d['Table'] = ZHStack(tables)

        full_2d_components[f"{predictor}"] = ZVStack([
            "<hr>",
            ZTab(tabs_2d)
        ])

    report = ZHStack([ZTitle(title), ZVStack(['<hr>', ZSelectDropdown(full_2d_components)])])
    if return_html:
        return report.to_html(return_string=True)
    return report

def get_report_3d(df, analyzed_columns=[], title="3D Visualization", return_html=False):
    ## TODO: make this report with 3 selection
    if len(analyzed_columns) == 0:
        analyzed_columns = df.columns.tolist()

    if len(analyzed_columns) < 3:
        return None

    all_comb3 = itertools.combinations(analyzed_columns, 3)
    full_3d_components = {}
    for (col_1, col_2, col_3) in all_comb3:
        vis = describe_3d_with_plot(df, col_1, col_2, col_3)
        if vis is not None:
            figure_and_tabs = dict()
            for fig_name in vis.keys():
                if 'time' in fig_name:
                    if 'Changes over time' not in figure_and_tabs.keys():
                        figure_and_tabs['Changes over time'] = dict()
                    figure_and_tabs['Changes over time'][fig_name] = vis[fig_name]
                elif 'abs' in fig_name:
                    if 'Absolute Value' not in figure_and_tabs.keys():
                        figure_and_tabs['Absolute Value'] = dict()
                    figure_and_tabs['Absolute Value'][fig_name] = vis[fig_name]
                else:
                    if 'Percentage' not in figure_and_tabs.keys():
                        figure_and_tabs['Percentage'] = dict()
                    figure_and_tabs['Percentage'][fig_name] = vis[fig_name]

            overall_tab_components = {}
            for tab_name, tab_figures in figure_and_tabs.items():
                overall_tab_components[tab_name] = ZTab(tab_figures)
            full_3d_components[f"{col_1} & {col_2} & {col_3}"] = ZVStack(["<hr>", ZTab(overall_tab_components)])

    if len(full_3d_components.keys()) > 0:
        final_report = ZVStack([ZTitle(title), ZSelectDropdown(full_3d_components)], styles="margin-left:0 !important;")

        if return_html:
            return final_report.to_html(return_string=True)
        return final_report

    return None

@st.cache_resource(ttl=60*60)
def get_pyg_html(df: pd.DataFrame, default_tab:str='vis') -> str:
    # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
    # If you want to use feature of saving chart config, set `debug=True`
    html = get_streamlit_html(df, spec=f"{COMMON_PATH}/tmp_cache/pygwalker_cache.json", use_kernel_calc=True, debug=False, default_tab=default_tab)
    return html

@st.cache_data(ttl=60*60)
def get_report(df, show_overview=True, show_1d=False, show_2d=False, show_3d=False, show_dragndrop=False, analyzed_columns:list=[]):
    """
    This function is used to generate a report on df
    :param df: the dataframe to report
    :return: html string
    """
    if len(analyzed_columns) == 0:
        analyzed_columns = [col for col in df.columns.tolist() if df[col].nunique() > 1]

    if df.shape[0] > 500:
        data_html = convert_data_to_html(df=df.head(500))
        title = "Dataset (first 500 rows)"
    else:
        data_html = convert_data_to_html(df)
        title = "Dataset"

    full_components = [
        ZTitle(title),
        data_html
    ]
    ## overview ##
    if show_overview:
        try:
            df_overview = describe.data_overview_lite(df[analyzed_columns])
            full_components += [
                "<hr>", ZTitle('Data Overview'), convert_data_to_html(df_overview, class_name="overview-table")
            ]
        except:
            print('An error occurred while generating data overview.')
            print(format_exc())

    df_tmp = df[analyzed_columns]
    ## 1d analysis ##
    if show_1d:
        # TODO: add time plot
        try:
            report_1d = get_report_1d(df_tmp.copy(), analyzed_columns=analyzed_columns, return_html=False)
            if report_1d is not None:
                full_components += ["<hr>", report_1d]
        except:
            print('An error occurred while generating 1d visualization.')
            print(format_exc())

    ## 2d analysis ##
    if show_2d and df_tmp.shape[1] <= 20: # 20 charts
        try:
            report_2d = get_report_2d(df_tmp.copy(), analyzed_columns=analyzed_columns, return_html=False)
            if report_2d is not None:
                full_components += ["<hr>", report_2d]
        except:
            print('An error occurred while generating 2d visualization.')
            print(format_exc())

    ## 3d analysis ##
    if show_3d and df_tmp.shape[1] >= 3 and df_tmp.shape[1] <= 10: # charts
        try:
            report_3d = get_report_3d(df_tmp.copy(), analyzed_columns=analyzed_columns, return_html=False)
            if report_3d is not None:
                full_components += ["<hr>", report_3d]
        except:
            print('An error occurred while generating 3d visualization.')
            print(format_exc())
    try:
        blocks = ZVStack(full_components)
        data_html = ZReport(blocks, title='Dataset Report', layout='v').to_html(return_string=True)
    except:
        data_html = None

    dragndrop_html = None
    if show_dragndrop:
        try:
            with st.spinner("Generating Simple BI Tool:"):
                init_streamlit_comm()
                dragndrop_html = get_pyg_html(df)
        except:
            print('An error occurred while generating Simple BI Tool.')

    return {
        "data_html": data_html,
        "dragndrop_html": dragndrop_html,
        "viz_function": "report"
    }

