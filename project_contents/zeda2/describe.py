# custom pybloqs to fix chart not show up when export to html
from zeda2.custom_lib import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import re as _re
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import pybloqs as _p
import scipy.stats as _ss
import math as _math
from plotly.offline import init_notebook_mode as _init_notebook_mode

import plotly.graph_objects as _go
import plotly.express as _px
import plotly.io as _pio
from plotly.subplots import make_subplots as _make_subplots
from pybloqs import Block as _Block, VStack as _VStack
import pybloqs.block.table_formatters as _tf
from tqdm import tqdm as _tqdm
from si_prefix import si_format as _si_format
import pathlib as _pathlib
import geopandas as _gpd
import warnings as _warnings
from pandas.api.types import is_numeric_dtype as _is_numeric_dtype
from datetime import datetime as _datetime
_warnings.simplefilter(action='ignore', category=FutureWarning)

from zeda2 import data_format_helper as _data_format_helper
from zeda2 import describe_utils as _describe_utils
from zeda2.describe_utils import CustomHStack as _CustomHStack
from zeda2.describe_utils import render_html as _render_html, \
    get_mode_table as _get_mode_table, describe_percentiles_v2 as _describe_percentiles_v2, cut_off_dataframe as _cut_off_dataframe, bring_null_to_end as _bring_null_to_end, util_category_rename as _util_category_rename, get_plotly_fig_json as _get_plotly_fig_json, split_plotly_layout_trace_props as _split_plotly_layout_trace_props
import zeda2.describe_utils as _describe_utils
from zeda2.display_helper import _display, _color, display_markdown as _display_markdown
from zeda2.display_helper import display_df_with_title as _display_df_with_title
from zeda2.plot_2d import matplotlib_plot_2d_numeric_hist_compare as _matplotlib_plot_2d_numeric_hist_compare
from zeda2.plot_2d import plotly_plot_2d_category_with_label as _plotly_plot_2d_category_with_label
from zeda2.plot_2d import sns_plot_2d_numeric_with_label as _sns_plot_2d_numeric_with_label


root_dir = str(_pathlib.Path(__file__).parent.resolve())
_AAID_MAP_PATH = root_dir+'/data/input_common/aaid_10_april.csv'
_VIETNAME_PROVINCE_NAME_PATH = root_dir+'/data/input_common/json/province_vn.csv'
# _VIETNAME_GEO_JSON = root_dir+'/assets/json/vietnam2.geojson'
_VIETNAME_GEO_JSON = root_dir+'/data/input_common/json/vietnam_HSTS_lite.geojson'
_FMT_FONT = _tf.FmtFontsize(12, 'px')


_DEFAULT_CSS = [('leaflet_css', 'https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css'), ('bootstrap_css', 'https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css'), ('bootstrap_theme_css', 'https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css'), ('awesome_markers_font_css', 'https://maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css'), ('awesome_markers_css', 'https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css'), ('awesome_rotate_css', 'https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css')]
_DEFAULT_JS = [('leaflet', 'https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js'), ('jquery', 'https://code.jquery.com/jquery-1.12.4.min.js'), ('bootstrap', 'https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js'), ('awesome_markers', 'https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js')]


# Fix bug not show chart
_init_notebook_mode(connected=False)
_px.defaults.template = 'simple_white'
_pio.templates.default = 'simple_white'
_pd.options.mode.chained_assignment = None
_TABLE_WIDTH = '320px'
_NULL_COLOR = '#D62728'
_MAIN_COLOR = '#1F77B4'
_COUNT_COL_NAME = 'count'
_FREQ_COL_NAME = 'frequency (%)'
_FIG_WIDTH = 600
_FIG_HEIGHT = 450
# 1d cate constant
_CATE_THRESHOLD = 9
# 1d num constant 
_NUM_DEFAULT_NBINS = 50
# num cate constatn
_NUM_CATE_THRESHOLD = 4
_NUM_CATE_COUNT_NBINS = 40
_NUM_CATE_FIG_HEIGHT = 300
_POPULATION_PYRAMID_WIDTH = 750
_POPULATION_PYRAMID_HEIGHT = 570
# num num contant
_DEFAULT_2D_HISTOGRAM_NBINS=20
_KDE_DEFAULT_NPOINTS = 300


def init_lib():
    from plotly.offline import init_notebook_mode as _init_notebook_mode
    from IPython.display import display as _display, HTML as _HTML

    # _init_notebook_mode(connected=False)

    # fix jupyterlab not showing chart
    js = '<script defer="defer" src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'
    _display(_HTML(js))

    _init_notebook_mode(connected=False)


def tranform_numeric_column(df, col, dict_format):
    df_temp = _pd.DataFrame(index=df.index, columns=[col])
    for idx in df.index:
        fmt = dict_format.get(idx, '.2f')
        df_temp.loc[idx, col] = f'{{:{fmt}}}'.format(float(df.loc[idx, col]))
    return df_temp[col]


def describe_1d_numeric(df, col, percentile_df=None, percentiles=None, fmt=',.2f', is_simple=False, raw=False):
    if percentile_df is None and df is not None:
        percentile_df = _describe_percentiles_v2(df, col, percentiles)
    dict_format = {
        'count': ',.0f',
        'mean': ',.2f',
        'tr_mean': ',.2f', 
        'std': ',.2f',
        'null': '.2%',
        'non_null': ',.0f',
        'skew': '.2f',
        'kurt': '.2f'
    }
    if not raw:
        for stat in percentile_df.index:
            # if stat not in ['count', 'null', 'mean', 'tr_mean', 'std', 'skew', 'kurt']:
            if stat not in ['count', 'non_null', 'null', 'skew', 'kurt']:
                dict_format[stat] = fmt
        # if raw return number, otherwise return formatted number into string
        percentile_df['value'] = tranform_numeric_column(percentile_df, 'value', dict_format)

    # concat the 2
    if is_simple:
         # quantile stats
        percentiles = ["min", "1%", "5%"]+[f"{i*10}%" for i in range(1, 10)]+["95%", "99%", "max"]
        # descriptive stats
        descriptives = [stat for stat in ["count", "non_null", "null" , "mean", "tr_mean", "std", "skew", "kurt"] if stat in percentile_df.index]
        df_stat = percentile_df.loc[descriptives+percentiles].rename_axis("stat", axis="index").reset_index()
    else:
        # quantile stats
        percentiles = ["1%", "5%"]+[f"{i*10}%" for i in range(1, 10)]+["95%", "99%"]
        quantiles_df = percentile_df.loc[percentiles].rename_axis("percentile", axis="index").reset_index()
        # descriptive stats
        descriptives = ["count", "non_null", "null", "min", "max", "mode", "mean", "tr_mean", "std", "mad", "IQR", "skew", "kurt"]
        descriptive_df = percentile_df.loc[descriptives].rename_axis("stat", axis="index").reset_index()
        # concat
        df_stat = _pd.concat([descriptive_df, quantiles_df], axis=1)
    df_stat = df_stat.set_index("stat").rename_axis(None)
    return df_stat


def _describe_2d_num_cate_table(
        df_temp,
        num, 
        cate, 
        threshold=(_NUM_CATE_THRESHOLD, None),
        list_keep=None,
        fmt=',.2f'
        ):
    df_temp[cate] = df_temp[cate].apply(str)
    # TABLE
    df_count, _ = describe_1d_category_with_percentiles(
        df_temp, 
        cate,
        list_keep=list_keep,
        threshold=threshold
        )
    cate_values = [cate for cate in df_count.index if cate not in ['Others']]
    # table describe num by each cate
    df_describe = _pd.DataFrame({})
    for value in cate_values:
        tmp = describe_1d_numeric(df_temp[df_temp[cate] == value], num, is_simple=True, fmt=fmt)
        tmp.columns=[str(value)]
        if len(df_describe.columns) > 0:
            df_describe = _pd.merge(df_describe, tmp, how='inner', left_index=True, right_index=True)
        else:
            df_describe = tmp
    # describe num for Others
    if df_count.shape[0] > len(cate_values):
        tmp = describe_1d_numeric(df_temp[~df_temp[cate].isin(cate_values)], num, is_simple=True, fmt=fmt)
        tmp.columns = ['Others']
        # when all cate have below threshold frequency occurence, df_describe would be empty
        if len(df_describe.columns) == 0:
            df_describe = tmp
        else:
            df_describe = _pd.merge(df_describe, tmp, how='inner', left_index=True, right_index=True)
    # describe num for All
    tmp = describe_1d_numeric(df_temp, num, is_simple=True, fmt=fmt)
    tmp.columns = ['All']
    df_describe = _pd.merge(df_describe, tmp, how='inner', left_index=True, right_index=True)
    # formatting
    # fmt_seperator = _tf.FmtThousandSeparator(rows='count')
    # fmt_seperator2 = _tf.FmtThousandSeparator(2, rows=[idx for idx in df_describe.index if idx != 'count'])
    # formatters = [fmt_seperator, fmt_seperator2, _FMT_FONT]
    # table = _Block(df_describe, formatters=formatters)

    df_style = (
        df_describe
        .style
        .set_table_styles(_describe_utils._TABLE_PADDING_STYLES)
        )
    table = _Block(_describe_utils.render_html(df_style))
    return table, df_describe, cate_values


def describe_2d_numeric_category(
        df,
        num,
        cate,
        threshold=(_NUM_CATE_THRESHOLD, None),
        list_keep=None):
    _, df_describe, _ = _describe_2d_num_cate_table(
        df, num=num, cate=cate, 
        threshold=threshold,
        list_keep=list_keep
        )
    return df_describe


def _describe_2d_num_cate_count_plot(df, num, cate, cate_values, plot_kwargs):
    # in case all cates do not surpass the threshold cut-off, add Others to fig
    fig_count_layout={
        'width': _FIG_WIDTH,
        'height': _NUM_CATE_FIG_HEIGHT,
        'title': f'Count comparison of {num} by {cate}',
        'title_x': 0.2,
        'title_y': 0.9,
        'margin': dict(b=10, t=60),
        'yaxis_title': 'count',
        'xaxis_title': num
    }
    fig_count_layout.update(plot_kwargs)
    fig_count = _go.Figure()
    common_binsize = None
    for value in cate_values:
        if df[df[cate]==value][num].count() > 1:
            bins, counts, _, binsize = _describe_utils.calculate_bin(
                df[df[cate]==value][num].dropna(), 
                nbins=_NUM_CATE_COUNT_NBINS, 
                binsize=common_binsize, 
                non_empty_bins=common_binsize is None)
            # to ensure same binsize for fair comparison
            if common_binsize is None:
                common_binsize = binsize
            list_x = [bins[0]]
            list_y = [0]
            list_text = [f'{cate}={value}']
            for i in range(len(bins)-1):
                start_x = bins[i]
                end_x = bins[i+1]
                list_x+=[start_x, end_x]
                list_y+=[counts[i]]*2
                list_text+=[f'{cate}={value}<br>{num}={start_x} - {end_x}<br>count={counts[i]}']*2
            fig_count.add_trace(_go.Scatter(x=list_x, y=list_y, name=value, hovertext=list_text))
    fig_count.update_layout(
        **fig_count_layout
        ).update_traces(
        hovertemplate='%{hovertext}'
    )
    return fig_count


def _describe_2d_num_cate_density_plot(df, num, cate, cate_values, plot_kwargs):
    fig_density_layout={
        'width': _FIG_WIDTH,
        'height': _NUM_CATE_FIG_HEIGHT,
        'title': f'Kernel density comparison of {num} by {cate}',
        'title_x': 0.2,
        'title_y': 0.9,
        'margin': dict(b=10, t=60),
        'yaxis_title': 'density',
        'xaxis_title': num
    }
    fig_density_layout.update(plot_kwargs)
    fig_density = _go.Figure()
    for value in cate_values:
        num_of_this_cate = df[df[cate]==value][num].dropna()
        if num_of_this_cate.count() > 1 and num_of_this_cate.nunique() > 1:
            kde = _describe_utils.make_kde(df[df[cate]==value][num].dropna(), value, n_points=150)
            fig_density.add_trace(kde)
    fig_density.update_layout(
        **fig_density_layout
    )
    return fig_density


def _describe_2d_num_cate_box_plot(df, num, cate, cate_values, fmt, plot_kwargs, df_percentile=None):
    fig_box_layout={
            'showlegend': False,
            'title': f'Percentiles comparison of {num} by {cate}',
            'title_x': 0.2,
            'title_y': 0.9,
            'margin': dict(b=10, t=60),
            'width': _FIG_WIDTH,
            'height': _NUM_CATE_FIG_HEIGHT ,
            'xaxis_title': num,
            'yaxis_title': cate
        }
    fig_box_layout.update(plot_kwargs)
    fig_box = _go.Figure()
    for value, color in zip(cate_values, _px.colors.qualitative.D3):
        if df is not None:
            box_trace = _describe_utils.get_box_trace(df[df[cate]==value][num], color=color, orientation='h', fmt=fmt)
        elif df_percentile is not None:
            box_trace = _describe_utils.get_box_trace(
                series=None, 
                series_percentile=df_percentile[value], 
                color=color, orientation='h', fmt=fmt)
        box_trace.update(y=[value], x=None, name=value)
        fig_box.add_trace(box_trace)
    fig_box.update_layout(
        **fig_box_layout
        )
    fig_box.update_xaxes(ticks='')
    return fig_box


def _describe_2d_num_cate_format_table(df, fmt):
    df_temp = df.copy()
    dict_format = {
        'count': ',.0f',
        'mean': ',.2f',
        'tr_mean': ',.2f', 
        'std': ',.2f',
        'null': '.2%',
        'skew': '.2f',
        'kurt': '.2f'
    }
    for stat in df_temp.index:
        if stat not in ['count', 'null', 'mean', 'std', 'skew', 'kurt']:
            dict_format[stat] = fmt
    for col in df_temp.columns:
        df_temp[col] = tranform_numeric_column(df_temp, col, dict_format)
        
    # formatting
    # fmt_seperator = _tf.FmtThousandSeparator(rows='count')
    # fmt_seperator2 = _tf.FmtThousandSeparator(2, rows=[idx for idx in df_temp.index if idx != 'count'])
    # formatters = [fmt_seperator, fmt_seperator2, _FMT_FONT]
    # table = _Block(df_temp, formatters=formatters)

    # df_style = (
    #     df_temp
    #     .style
    #     .format(formatter='{:,.0f}', subset=_pd.IndexSlice['count', :])
    #     .format(formatter='{:,.2f}', subset=_pd.IndexSlice[[idx for idx in df_temp.index if idx != 'count'], :])
    #     .set_table_styles(_describe_utils._TABLE_PADDING_STYLES)
    #     )

    table = _Block(_describe_utils.render_html(df_temp))
    return table, df_temp


def describe_2d_numeric_category_with_plot(
        df,
        num,
        cate,
        threshold=(_NUM_CATE_THRESHOLD, None),
        category_rename=None,
        dropna=False,
        list_keep=None,
        fmt=',.2f',                                 
        plot_count_kwargs={},
        plot_density_kwargs={},
        plot_box_kwargs={},
        precomputed_kwargs=None,
        return_table=False,
        return_raw=False,
        return_json=False):
    """
    Describe interaction between a categorical and a numeric feature.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    num : str
        Name of numeric column to be analysed in the input dataframe.
    
    cate : str
        Name of categorical column to be analysed in the input dataframe

    threshold : int or (int, int), default: (9, None)
        Threshold for `cate`.
        - First value is rank threshold, threshold categories with the largest occurence frequecy are kept and the others are merged together.
        - Second value is count threshold, all categories with occurence frequecy smaller than the threshold will be merged together.
        If an integer is provided, it is assigned to rank threshold and count threshold is set to None.

    category_rename : dict, default None
        Key is column name and value is a dict to map categories in the column into new names. If a category is not given in the value dict, its original name is used.

    dropna : boolean, default: False
        If False, NA values will be treated as a category, dropped otherwise.

    list_keep : list, default: None
        List of categories to be kept regardless of threshold conditions.

    plot_count_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the count plot.
    
    plot_density_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the density plot.
    
    plot_box_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the box plot.

    precomputed_kwargs : dict, default None
        Desgined for internal use.

    return_table : boolean, default False
        Whether to return the formatted table only.

    return_fig : boolean, default False
        Whether to return the plot only.

    return_json : boolean, default False
        Whether to return the analysis in json format. This is specifically designed for internal use.

    Return
    ------
    HStack containing statistics table and plots comparing count, kernel density estimation and percentiles of numeric column by categories."""

    # TABLE
    df_temp = None
    df_describe = None
    json_table = None
    if df is not None and cate and num:
        df_temp = df[[cate, num]].copy()
        if dropna:
            df_temp = df_temp.dropna(subset=[num, cate])
        if category_rename and cate in category_rename:
            df_temp[cate] = df_temp[cate].apply(lambda x: category_rename[cate].get(x, x))
        df_temp[num] = _describe_utils.series_cast_numeric(df_temp[num])

        table, df_describe, cate_values = _describe_2d_num_cate_table(
            df_temp, 
            num = num, 
            cate = cate, 
            threshold = threshold, 
            list_keep=list_keep,
            fmt=fmt
            )
    elif precomputed_kwargs is not None:
        df_describe = precomputed_kwargs['df_describe']
        df_describe_all = precomputed_kwargs['df_describe_all']
        df_describe.columns = df_describe.columns.map(str)
        # truncate df_count to threshold
        df_count, _ = describe_1d_category_with_percentiles(
            df=None,
            col=None, 
            value_count_df=precomputed_kwargs['df_count'].sort_values('count', ascending=False),
            threshold=threshold,
            list_keep=list_keep,
            null_value='nan',
            count_col='count'
            )
        cate_values = [cate for cate in df_count.index if cate not in ['Others']]
        df_describe_join = df_describe[cate_values].merge(df_describe_all, left_index=True, right_index=True)
        sorted_stat = ['count', 'null', 'mean', 'std', 'min', '1%', '5%'] + [f'{x*0.1:.0%}' for x in range(1, 10)] + ['95%', '99%', 'max']
        table, df_describe = _describe_2d_num_cate_format_table(
            df_describe_join.loc[sorted_stat],
            fmt=fmt)

    fig_count, fig_density, fig_box = None, None, None

    if df_temp is not None and num and cate:
        # FIG COUNT
        # fig_count = _describe_2d_num_cate_count_plot(
        #     df_temp,
        #     num, cate,
        #     cate_values=cate_values,
        #         plot_kwargs=plot_count_kwargs)
        # # FIG DENSITY
        # fig_density = _describe_2d_num_cate_density_plot(
        #     df_temp,
        #     num, cate,
        #     cate_values=cate_values,
        #     plot_kwargs=plot_density_kwargs)
        # FIG BOX
        fig_box = _describe_2d_num_cate_box_plot(
            df_temp,
            num, cate,
            cate_values=cate_values,
            fmt=fmt,
            plot_kwargs=plot_box_kwargs)
    elif df_describe is not None:
        fig_box = _describe_2d_num_cate_box_plot(
            None,
            num, cate,
            cate_values=cate_values,
            fmt=fmt,
            plot_kwargs=plot_box_kwargs,
            df_percentile=df_describe_join.loc[['min', 'max', '25%', '50%', '75%', 'mean']]
            )

        json_table = _describe_utils.df_to_json(df_describe.rename_axis("stat"), keep_index=True)

    if return_table:
        return table
    if return_raw:
        if df_describe is not None:
            return [df_describe.rename_axis("stat", axis=1), fig_box]
        return [fig_box]

    if return_json:
        list_json_fig = [
            _get_plotly_fig_json(fig)
            if fig is not None else '' for fig in [fig_box]
            ]
        return (json_table, *list_json_fig)

    return _CustomHStack([table, _VStack([fig for fig in [fig_box] if fig is not None])])


def describe_2d_population_pyramid(
        df,
        col_age,
        col_gender,
        male_value,
        female_value,
        dropna=False,
        category_rename={},
        plot_kwargs={},
        precomputed_kwargs=None,
        return_json=False,
        return_raw=False
    ):
    """Draw population pyramid with plotly.

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    col_age : str
        Name of the age column.

    col_gender : str
        Name of the gender column.

    male_value: str
        Value to be intepreted as male.
    
    female_value: str
        Value to be intepreted as female.
    
    plot_kwargs: dict, default {}
        Define additional kwargs for plotly plot, these params will be passed into update_layout() in plotly.
    
    precomputed_kwargs : dict, default None
        Desgined for internal use.

    return_json : boolean, default False
        Whether to return the analysis in json format. This is specifically designed for internal use.
    """ 
    df_temp = None
    # build age range
    age_range = []
    age_left_edges = list(range(0, 90, 5))
    for i in age_left_edges:
        age_range.append(f'{i}-{i+5}')
    age_range[-1] = '85+'
    bin_edges = age_left_edges.copy()
    bin_edges[0] = -float('inf')
    bin_edges.append(float('inf'))    
    if df is not None and col_age and col_gender:
        df_temp = df[[col_gender, col_age]].copy()
        # preprocess data
        if dropna:
            df_temp = df_temp.dropna(subset=[col_age, col_gender])
        if category_rename and col_gender in category_rename:
            df_temp[col_gender] = df_temp[col_gender].apply(lambda x: category_rename[col_gender].get(x, x))
        df_temp[col_age] = _describe_utils.series_cast_numeric(df_temp[col_age])
        # describe table
        table, df_describe, cate_values = _describe_2d_num_cate_table(
            df_temp, 
            num = col_age, 
            cate = col_gender,
            fmt=',.0f'
            )
        # prepare data for plot
        age_male = df_temp[df_temp[col_gender] == male_value][col_age]
        age_female = df_temp[df_temp[col_gender] == female_value][col_age]
        bins_male = _describe_utils.calculate_bin_values(age_male.sort_values(), bin_edges)
        bins_female = _describe_utils.calculate_bin_values(age_female.sort_values(), bin_edges)
    elif precomputed_kwargs is not None:
        df_describe = precomputed_kwargs['df_describe']
        df_describe_all = precomputed_kwargs['df_describe_all']
        df_count = precomputed_kwargs['df_count']
        bins_male = precomputed_kwargs['bins_male']
        bins_female = precomputed_kwargs['bins_female']
        # cast cate to str
        df_describe.columns = df_describe.columns.map(str)
        df_count.index = df_count.index.map(str)
        # truncate df_count to threshold
        cate_values = [cate for cate in df_count.index if cate not in ['Others']]
        df_describe_join = df_describe[cate_values].merge(df_describe_all, left_index=True, right_index=True)
        sorted_stat = ['count', 'null', 'mean', 'std', 'min', '1%', '5%'] + [f'{x*0.1:.0%}' for x in range(1, 10)] + ['95%', '99%', 'max']
        table, df_describe = _describe_2d_num_cate_format_table(
            df_describe_join.loc[sorted_stat],
            fmt=',.0f')

    # compute value of each bin
    percent_female = bins_female/bins_female.sum()
    percent_male = bins_male/bins_male.sum()
    # compute tick values for xaxis
    max_xvalue = max(max(bins_female), max(bins_male))
    tickvals = []
    xstep = max_xvalue/len(age_left_edges)
    xstep = _describe_utils.round_bin_size(xstep)
    value = xstep
    while value < max_xvalue:
        tickvals.append(value)
        value+=xstep
    tickvals.append(value)
    tickvals=[val*-1 for val in tickvals]+[0]+tickvals
    ticktexts = [_si_format(val, 0) for val in tickvals]

    layout = _go.Layout(
        yaxis=_go.layout.YAxis(
            title='age',
            tickvals=age_left_edges,
            ticktext=age_range,
        ),
        xaxis=_go.layout.XAxis(
            range=[-max_xvalue*1.2, max_xvalue*1.2],
            title='count',
            tickvals=tickvals,
            ticktext=ticktexts,
        ),
        barmode='overlay',
        bargap=0.1,
        width=_POPULATION_PYRAMID_WIDTH,
        height=_POPULATION_PYRAMID_HEIGHT,
        legend=dict(
            title="",
            orientation="v",
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=0.95)
    )

    layout.update(plot_kwargs)

    def _get_pyramid_side(x, y, text, col_gender, gender_value, color):
        return _go.Bar(
            y=y,
            x=x,
            customdata=x.apply(abs),
            orientation='h',
            name=gender_value,
            hoverinfo='x',
            hovertemplate=f'{col_gender}={gender_value}<br>count=%{{customdata}}<br>percent=%{{text:.2%}}',
            text=text,
            textposition='outside',
            texttemplate='%{text:.2%}',
            marker=dict(color=color)
        )


    data = [_get_pyramid_side(
                x=bins_male,
                y=age_left_edges,
                text=percent_male,
                col_gender=col_gender,
                gender_value=male_value,
                color='#1F77B4'
            ),
            _get_pyramid_side(
                x=bins_female*-1,
                y=age_left_edges,
                text=percent_female,
                col_gender=col_gender,
                gender_value=female_value,
                color='#FF7F0E'
            )]

    fig = _go.Figure(data=data, layout=layout)

    if return_raw:
        return [df_describe.rename_axis("stat", axis=1), fig]

    if return_json:
        json_table = _describe_utils.df_to_json(df_describe.rename_axis("stat").reset_index())
        for idx in range(len(json_table)):
            json_table[idx]['stat'] = f'<b>{json_table[idx]["stat"]}</b>'
        json_fig = _get_plotly_fig_json(fig)
        return json_table, json_fig

    return _CustomHStack([table, fig], grid_cell_style='margin-right:0px')


def describe_2d_numeric_with_plot(
        df, 
        num1, 
        num2,
        plot_type='density',
        nbins=_DEFAULT_2D_HISTOGRAM_NBINS,
        threshold=0,
        plot_kwargs={},
        precomputed_kwargs=None,
        return_raw=False,
        return_json=False):
    """Describe 2D numeric and numeric with plotly.

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    num1 : str
        Name of the first numeric column to be analysed in the input dataframe.

    num2 : str
        Name of the second numeric column to be analysed in the input dataframe.

    plot_type : str in ("density", "scatter"), default: "density"
        If "density", a count matrix, or so-called heatmap, constructed by cross-binning the two columns is plotted.
        If "scatter", all points are plotted in a scatter plot. A scatter plot with too many points may leads to notebook crash. 
    
    nbins : int, default 20
        The number of bins split data into for both columns. The resulted number of bins are not necessarily equal to the provided nbins but are rounded for smoother visualization.    

    plot_kwargs : dict, default: {}
        Define additional kwargs for plotly plot, these params will be passed into update_layout() in plotly

    return_json : boolean, default: False
        Specific use for zml_auto_eda
    """
    threshold = threshold or 0
    pearson = None
    if df is not None and num1 and num2:
        df_temp = df[[num1, num2]].copy()
        df_temp[num1] = _describe_utils.series_cast_numeric(df_temp[num1])
        df_temp[num2] = _describe_utils.series_cast_numeric(df_temp[num2])

        pearson = _describe_utils.pearson(df_temp[num1], df_temp[num2])
        if plot_type is None and df_temp.shape[0] > 1e+4:
            plot_type = 'density'
        elif plot_type is None and df_temp.shape[0] <= 1e+4:
            plot_type= 'scatter'
    elif precomputed_kwargs is not None:
        pearson = precomputed_kwargs['pearson']
        bin_centers1 = _describe_utils.get_bin_centers_from_edges(precomputed_kwargs['bin_edges1'])
        bin_centers2 = _describe_utils.get_bin_centers_from_edges(precomputed_kwargs['bin_edges2'])
        count_matrix = precomputed_kwargs['count_matrix']
        plot_type = 'density'
    if pearson is not None:
        pearson_title = f'<br><b style=`font-size:17px`>Pearson r={pearson:.2f}</b>'
    fig_layout={
        'width': 750,
        'height': 500,
        'xaxis_title': num2,
        'yaxis_title': num1
    }
    fig_layout.update(plot_kwargs)
    # start plotting
    if plot_type == 'density':
        if df is not None and num1 and num2:
            bin_edges1, _, bin_centers1, _ = _describe_utils.calculate_bin(df_temp[num1], nbins=nbins)
            bin_edges2, _, bin_centers2, _ = _describe_utils.calculate_bin(df_temp[num2], nbins=nbins)
            count_matrix = _describe_utils.calculate_2d_binning(df_temp[num1], df_temp[num2], bin_edges1, bin_edges2)
            # change 0 to '-'
            count_matrix[_np.isnan(count_matrix)] = 0
        text = _pd.DataFrame(count_matrix)
        text[text<threshold] = 0
        # if less than 1000 precision is 0, otherwise precision is 1
        text = text.applymap(lambda x: _si_format(x, precision=_describe_utils.get_precision(x), format_str='{value}{prefix}'))
        # get hover text
        # if len(bin_edges1)>=2:
        #     hover1 = [f'{left_edge}-{right_edge}' for left_edge, right_edge in zip(bin_edges1[:-1],bin_edges1[1:])]
        # else:
        #     hover1 = []
        
        # if len(bin_edges2)>=2:
        #     hover2 = [f'{left_edge}-{right_edge}' for left_edge, right_edge in zip(bin_edges2[:-1],bin_edges2[1:])]
        # else:
        #     hover2 = []

        text[text=='0'] = '-'
        # symbol_explain_title=
        fig = (
            _go.Figure(_go.Heatmap(
                    x=bin_centers2, y=bin_centers1, z=count_matrix, 
                    text=text, colorscale='Blues',
                    texttemplate="%{text}",
                    hovertemplate=f'{num1}=%{{x}}<br>{num2}=%{{y}}<br>count=%{{text}}'))
                .add_annotation(
                    dict(font=dict(color='grey',size=13),
                    x=0.4,
                    y=1,
                    showarrow=False,
                    text=f"<br><span style=`color:grey;font-size:13px;margin-bottom:40px`>value &le {threshold:,.0f} is annotated by an en dash (-)</span>",
                    textangle=0,
                    xanchor='left',
                    yanchor='bottom',
                    xref="paper",
                    yref="paper"))
            )
        fig_layout['title'] = fig_layout.get('title', None) or (f'Count Density of {num1} by {num2}')
    elif plot_type == 'scatter':
        fig_layout['title'] = fig_layout.get('title', None) or (f'Joint Distribution of {num1} by {num2}')
        fig = _px.scatter(df_temp, x=num1, y=num2, trendline='ols', trendline_color_override="#FF7F0E")
        # add legend to trendline
        fig.data[1].name='Least Square Regression'
        fig.data[1].showlegend=True
        fig.update_layout(legend=dict(x=0.9, y=1.05))

    fig_layout['title'] += pearson_title
    fig.update_layout(**fig_layout)
    fig.update_layout(title_x=0.5)

    if return_raw:
        return [fig]

    if return_json:
        json_fig = _get_plotly_fig_json(fig)
        return json_fig

    return _CustomHStack([fig], styles={'text-align': 'center'})


def data_overview(df, precomputed_df=None, verbose=False, return_raw=False, return_df=False, return_json=False):
    """Generate summary statistics of a Pandas DataFrame.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    verbose : Boolean, default: False
        Whether to include mode value information or not.

    Return
    ------
    Summary statistics for each column of the given dataframe.
    """

    def highlight(row):
        """Heuristics to add visual cues."""
        ret = ['' for _ in row.index]

        # the former handles spark dtype, the latter handles pandas dtype
        if"dtype" in row.index:
            if ((type(row["dtype"]) == str and row['dtype'] == 'string') or 
                (type(row["dtype"]) != str and row["dtype"].name == "object") ):
                ret[row.index.get_loc("dtype")] = "color: red; font-weight: bold"

        if (row['distinct (%)'] <= 0.05) & (row['#distinct'] <= 100):
            ret[row.index.get_loc("#distinct")] = "color: blue; font-weight: bold"
        if row['distinct (%)'] <= 0.1:
            ret[row.index.get_loc("distinct (%)")] = "color: blue"

        if row['null (%)'] >= 0.2:
            ret[row.index.get_loc("#null")] = 'color: red; font-weight: bold'
            ret[row.index.get_loc("null (%)")] = 'color: red; font-weight: bold'
        elif row['null (%)'] >= 0.1:
            ret[row.index.get_loc("#null")] = 'color: red'
            ret[row.index.get_loc("null (%)")] = 'color: red'

        if row['zero (%)'] >= 0.2:
            ret[row.index.get_loc("#zero")] = 'color: red; font-weight: bold'
            ret[row.index.get_loc("zero (%)")] = 'color: red; font-weight: bold'
        elif row['zero (%)'] >= 0.1:
            ret[row.index.get_loc("#zero")] = 'color: red'
            ret[row.index.get_loc("zero (%)")] = 'color: red'

        if row['negative (%)'] >= 0.2:
            ret[row.index.get_loc("#negative")] = 'color: red; font-weight: bold'
            ret[row.index.get_loc("negative (%)")] = 'color: red; font-weight: bold'
        elif row['negative (%)'] >= 0.1:
            ret[row.index.get_loc("#negative")] = 'color: red'
            ret[row.index.get_loc("negative (%)")] = 'color: red'

        if row['empty (%)'] >= 0.2:
            ret[row.index.get_loc("#empty")] = 'color: red; font-weight: bold'
            ret[row.index.get_loc("empty (%)")] = 'color: red; font-weight: bold'
        elif row['empty (%)'] >= 0.1:
            ret[row.index.get_loc("#empty")] = 'color: red'
            ret[row.index.get_loc("empty (%)")] = 'color: red'

        if row.hit_rate <= 0.5:
            ret[row.index.get_loc("hit_rate")] = 'color: red; font-weight: bold'
        elif row.hit_rate <= 0.6:
            ret[row.index.get_loc("hit_rate")] = 'color: red'
        return ret

    if df is not None:
        # total = df.shape[0]
        n_rows = df.shape[0]
        data_type = df.dtypes
        object_cols = data_type[data_type == "object"]

        # Convert object columns (list, set, ndarray, ...) to string for stats computing
        if len(object_cols) > 0:
            df = df.copy()
            df[object_cols.index] = df[object_cols.index].astype("string")

        unique_count = df.nunique()

        null_count = df.isnull().sum()
        zero_count = df.isin([0, "0"]).sum()
        zero_count = df.isin([0, "0"]).sum()
        blank_count = df.isin(["", "{}", '[]']).sum()

        unique_perc = unique_count /  n_rows
        null_perc = null_count / n_rows
        zero_perc = zero_count / n_rows
        blank_perc = blank_count / n_rows

        negative_count = []
        negative_perc = []
        for column_name in df.columns:
            if _is_numeric_dtype(df[column_name]):
                tmp_count = (df[column_name]<0).sum()
                negative_count.append(tmp_count)
                negative_perc.append(tmp_count / n_rows)
            else:
                negative_count.append(0)
                negative_perc.append(0)

        hit_rate = (1 - null_perc)

        missing_df = _pd.DataFrame({
            'total': n_rows,
            'dtype': data_type,
            '#distinct': unique_count,
            'distinct (%)': unique_perc,

            '#null': null_count,
            '#zero': zero_count,
            '#negative': negative_count,
            '#empty': blank_count,

            'null (%)': null_perc,
            'zero (%)': zero_perc,        
            'negative (%)': negative_perc,
            'empty (%)': blank_perc,

            'hit_rate': hit_rate
        })

        if verbose:
            mode_df = _get_mode_table(df)
            missing_df = _pd.concat([missing_df, mode_df], axis=1)
    elif precomputed_df is not None:
        missing_df = precomputed_df

    if return_df:
        return missing_df

    missing_df.insert(0, "#", _np.arange(1, 1 + len(missing_df)))  # add columns index
    # Add Pandas style
    border_props = [
        ("border-left-color", "black"),
        ("border-left-style", "dotted"),
        ("border-left-width", "thin"),
    ]

    inline_props = [
        ('overflow', 'hidden'),
        ('white-space', 'nowrap')]
    
    left_border_class = _pd.DataFrame(
                    index=missing_df.index,
                    columns=['#distinct', '#null', 'null (%)', 'hit_rate']).fillna('left_border_column')
    missing_df_style = (
        missing_df.style
            # .set_caption(f'Data Overview')
            .format({
            'count': lambda x: '{:,.0f}'.format(x),
            'total': lambda x: '{:,.0f}'.format(x),
            '#empty': lambda x: '{:,.0f}'.format(x) if x > 0 else '-',
            '#zero': lambda x: '{:,.0f}'.format(x) if x > 0 else '-',
            '#null': lambda x: '{:,.0f}'.format(x) if x > 0 else '-',
            # '#duplicate': lambda x: '{:,.0f}'.format(x) if x > 0 else '-',
            '#negative': lambda x: '{:,.0f}'.format(x) if x > 0 else '-',
            '#distinct': lambda x: '{:,.0f}'.format(x) if x > 0 else '-',

            'distinct (%)': lambda x: '{:.2%}'.format(x) if x >= 0.1 else '-',
            'empty (%)': lambda x: '{:.2%}'.format(x) if x >= 0.01 else '-',
            'zero (%)': lambda x: '{:.2%}'.format(x) if x >= 0.01 else '-',
            'null (%)': lambda x: '{:.2%}'.format(x) if x >= 0.01 else '-',
            # 'duplicate (%)': lambda x: '{:.2%}'.format(x) if x >= 0.01 else '-',
            'hit_rate': lambda x: '{:.2%}'.format(x) if x >= 0.01 else '-',
            'negative (%)': lambda x: '{:.2%}'.format(x) if x >= 0.01 else '-',
        })
            .apply(highlight, axis=1)
            .set_table_styles([
            dict(selector='caption', props=[
                ('color', 'black'),
                ('font-size', '16px'),
                ('font-weight', 'bold')
            ]),
            dict(selector='th.col_heading.level0.col3', props=border_props),
            dict(selector='th.col_heading.level0.col5', props=border_props),
            dict(selector='th.col_heading.level0.col9', props=border_props),
            dict(selector='th.col_heading.level0.col13', props=border_props),
            dict(selector='th.col_heading.level0.col4', props=inline_props),
            dict(selector='th.col_heading.level0.col9', props=inline_props),
            dict(selector='th.col_heading.level0.col10', props=inline_props),
            dict(selector='th.col_heading.level0.col11', props=inline_props),
            dict(selector='th.col_heading.level0.col12', props=inline_props)
        ])
            .set_properties(**{"text-align": "center"})
            .set_properties(**dict(border_props), subset=['#distinct', '#null', 'null (%)', 'hit_rate'])
            .set_td_classes(left_border_class)
    )
    if verbose:
        missing_df_style = (
            missing_df_style
                .format({"mode_perc": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-"}, subset=["mode_perc"])
                .set_properties(**dict(border_props), subset=["mode"])
                .set_properties(**{"width": "max-content", "text-overflow": "ellipsis"},
                                subset=["mode"])
                .set_table_styles([dict(selector="th.col_heading.level0.col14", props=border_props)], overwrite=False)
        )
    if return_raw:
        return missing_df_style
    
    html_string = _describe_utils.render_html(missing_df_style)
    if return_json:
        return html_string
    return _Block(html_string)


def data_overview_lite(input_dataframe):
    """
        Show full overral missing of dataframe columns
        Args:
            input_dataframe: pandas dataframe
        Returns:
            Table full missing explore by columns (columns: count_total, count_unique, count_duplicate, count_zero, percentile_zero, count_missing, percentile_missing, hit_rate)
    """
    # print("**Overral Stats**")
    total = len(input_dataframe)
    na_count = input_dataframe.isnull().sum()
    zero_count = len(input_dataframe) - input_dataframe.fillna(1).astype(str).astype(bool).sum()
    zero_percent = (zero_count / len(input_dataframe) * 100).round(2).map(lambda n: '{0:.2f} %'.format(n))
    na_percent = (input_dataframe.isnull().sum() / len(input_dataframe) * 100).round(2).map(
        lambda n: '{0:.2f} %'.format(n))
    uniq_count = input_dataframe.nunique()
    dup_count = []
    for column_name in input_dataframe.columns:
        dup_count.append(input_dataframe.duplicated(subset=column_name, keep='first').sum())
    hit_rate = (input_dataframe.notnull().sum() / len(input_dataframe) * 100).round(2).map(
        lambda n: '{0:.2f} %'.format(n))
    return _pd.DataFrame(
        {'count_total': total, 'count_unique': uniq_count, 'count_dup': dup_count, 'count_zero': zero_count,
         'zero_per': zero_percent, 'count_missing': na_count, 'percentile_missing': na_percent,
         'hit_rate': hit_rate})


def plot_vietnam_province_map(
        df=None,
        col_aaid=None, 
        col_province=None, 
        legend_name='', 
        outliers=['Hồ Chí Minh', 'Hà Nội'],
        value_count_df=None,
        return_json=False,
        **kwargs
        ):
    # read df containing all provinces of Vietnam
    df_vietnam_province = _pd.read_csv(_VIETNAME_PROVINCE_NAME_PATH, header=None)
    df_vietnam_province.columns=['tone_name', 'no_tone_name']
    #map from aaid to province
    if value_count_df is None and df is not None:
        if col_province is None and col_aaid:
            df_temp = df[[col_aaid]].copy()
            df_parse = _data_format_helper.parse_aaid(df_temp, col_aaid)
            df_temp[['Country', 'City', 'District', 'Ward']] = df_parse[['Country', 'City', 'District', 'Ward']]
            df_temp = df_temp[df_temp['Country']=='Vietnam']
            # df_temp['City'] = df_temp['City'].apply(lambda x: x.replace('Tỉnh ', '').replace('Thành phố ', '') if x else x)
            col_province = 'City'
        else:
            col_province = col_province or 'Living Province'
            df_temp = df[[col_province]].copy()
            # get rid of invalid province
            df_temp = df_temp[df_temp[col_province].isin(df_vietnam_province.tone_name)]
        # describe
        df_describe, _ = describe_1d_category_with_percentiles(
            df_temp, 
            col_province,
            dropna=True, 
            threshold=None, 
            list_keep=None, 
            category_orders=None)
    elif value_count_df is not None:
        col_province = col_province or 'Living Province'
        df_describe = value_count_df.copy()
    else:
        raise Exception('Either `df` or `value_count_df` must be provided.')
    

    # remove nan index
    df_describe = df_describe.loc[[idx for idx in df_describe.index if idx == idx]]
    # formatted for display
    df_describe['count_str'] = df_describe['count'].apply(lambda x: '{:,.0f}'.format(x))
    df_describe['freq_str'] = df_describe['frequency (%)'].apply(lambda x: '{:.2%}'.format(x))
    # modify outlier count
    outliers = list(set(outliers).intersection(set(df_describe.index)))
    df_describe[f'count_modified'] = df_describe['count']
    temp = df_describe.loc[[c for c in df_describe.index if c not in outliers]]
    max_count_other = temp['count'].max()
    if max_count_other is not None and not _pd.isna(max_count_other):
        df_describe.loc[outliers,f'count_modified']=int(max_count_other*1.5)
    df_describe = df_describe.rename_axis(col_province).reset_index()
    # process absent provinces
    df_describe = (
        df_vietnam_province[['tone_name']]
        .rename(columns={'tone_name': col_province})
        .merge(df_describe, left_on=col_province, right_on=col_province, how='left')
        .fillna(0))
    # read geo json file
    df_geo = (_gpd.read_file(_VIETNAME_GEO_JSON)
        .drop(columns=['cartodb_id', 'id_1', 'slug'])
        .rename(columns={'name': col_province}))
    df_geo = df_geo.merge(df_describe)
    # draw map
    fields = [col_province, 'count_str', 'freq_str']
    aliases = ['Province', 'count', 'percentage']
    legend_name = 'count by province'
    # preprocess for absent provinces
    m = _describe_utils.plot_choropleth(
        df_describe, 
        df_geo, 
        columns=[col_province, 'count_modified'], 
        key_on=col_province, 
        legend_name=legend_name, 
        fields=fields, 
        aliases=aliases,
        scrollWheelZoom=False,
        **kwargs
        )
    
    m.default_css = _DEFAULT_CSS
    m.default_js = _DEFAULT_JS
    if return_json:
        return m._repr_html_().replace('"', "'")
    return m


def hit_rate_explore(df, column):
    """
        Calculate percentage null and not null value of column in data frame
        Args:
            df: input dataframe
            column: name of column need calculate hit rate
        Returns:
            hitrate table of column
    """
    na_count = df[column].isnull().sum()
    not_na_count = df[column].notnull().sum()
    total = len(df)
    na_percentage = '{0:.1f} %'.format(na_count / total * 100)
    not_na_percentage = '{0:.1f} %'.format(not_na_count / total * 100)
    return _pd.DataFrame({'Category': ['Have ' + column, 'Does not have ' + column],
                          'Count': [not_na_count, na_count],
                          'Total': [total, total],
                          'Percent': [not_na_percentage, na_percentage]})


def bad_rate_explore(df, column, label):
    """
        Calculate percentage null and not null value of column in data frame
        Args:
            df: input dataframe
            column: name of column need calculate hit rate
        Returns:
            hitrate table of column
    """
    na_count = df[column].isnull().sum()
    not_na_count = df[column].notnull().sum()
    total = len(df)
    na_percentage = '{0:.1f} %'.format(na_count / total * 100)
    bad_rate_not_na = '{0:.1f} %'.format(len(df[(df[column].notnull()) & (df[label] == 1)]) / not_na_count * 100)
    bad_rate_na = '{0:.1f} %'.format(len(df[(df[column].isnull()) & (df[label] == 1)]) / na_count * 100)
    not_na_percentage = '{0:.1f} %'.format(not_na_count / total * 100)
    return _pd.DataFrame({'Category': ['Have ' + column, 'Does not have ' + column],
                          'Count': [not_na_count, na_count],
                          'Total': [total, total],
                          'Percent': [not_na_percentage, na_percentage],
                          'Bad rate': [bad_rate_not_na, bad_rate_na]})


# Describe category 1D
def describe_category(df, col, dropna=False, sort_by="value", ascending=False, **grouper_kwargs):
    """Get value count table. Support count by specified frequency if input column is datetime-like.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    col: str
        Name of the column to get value count.

    dropna: boolean, default: False
        If False, NA values will be treated as a distinct value, dropped otherwise.

    sort_by: str, {"value", "index"}, default: "value"
        Whether to sort the return dataframe by count value or index.
        By design, datetime-like column will always be sorted by time.

    ascending: boolean, default: False
        Sort ascending vs. descending.

    grouper_kwargs: keyword arguments
        Keyword arguments to feed into pd.Grouper to specify groupby instruction.
        For more details, see: https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html

    Return
    ------
    DataFrame containing counts, freqs and occurence ranks of unique values.
    """
    # Support groupby specified frequency if col is a datetime-like column
    value_count_df = df.groupby([_pd.Grouper(key=col, **grouper_kwargs)], dropna=dropna).size().to_frame("count")

    # Add rank and freq columns
    value_count_df = (
        value_count_df
            .rename_axis(index="index")
            .assign(**{
            "freq": lambda x: x["count"] / x["count"].sum(),
            "rank": lambda x: x["count"].rank(method="min", ascending=False)  # MODIFY: change rank method
        })
            .reset_index().convert_dtypes().set_index("index")  # remove insignificant trailing zeros
            .reindex(columns=["rank", "count", "freq"])
    )

    # Sort dataframe. By design, datetime-like column will always be sorted by time.
    if not _pd.api.types.is_datetime64_dtype(df[col].dtype):
        if sort_by != "index":
            value_count_df.sort_values("count", axis=0, ascending=ascending, inplace=True)
        elif sort_by == "index":
            value_count_df.sort_index(inplace=True, ascending=ascending)
        else:
            raise ValueError(f"Unregconized sort_by value: {sort_by}")

    # Rename index at the end to avoid duplicated columns problem. i.e.. col == "count"
    return value_count_df.rename_axis(index=col)


def describe_numeric_data(column, n_round=2):
    """
        Show describe numeric data
        Args:
            column: pandas serie datatype is numeric
            n_round: round format number (default = 2)
        Returns:
            Table describe data (total, std, percentiles,...)
    """
    return column.describe(percentiles=[i * 0.05 for i in range(20)] + [0.01] + [0.99]).round(n_round)


def describe_category_data(column):
    """
        Show describe category data
        Args:
            column: pandas series datatype is category
        Returns:
            Table describe data (count values, percentiles)
    """
    c = column.value_counts(dropna=False)
    p = column.value_counts(dropna=False, normalize=True).map(lambda n: '{0:.2f} %'.format(n * 100))
    return _pd.concat([c, p], axis=1, keys=['counts', 'percentiles(%)'])


def describe_category_data_without_format(column):
    """
        Show describe category data
        Args:
            column: pandas series datatype is category
        Returns:
            Table describe data (count values, percentiles)
    """
    c = column.value_counts(dropna=False)
    p = column.value_counts(dropna=False, normalize=True)
    return _pd.concat([c, p], axis=1, keys=['counts', 'percentiles'])


def condense_category(df, threshold=(9, None), freq_col="freq", rank_col=None, condense_name="Others"):
    """Condense least ocurrence categories into a single category.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    threshold: double or integer, default: 9
        If threshold <= 1, all categories with occurence frequecy smaller than threshold will be merged together.
        If threshold > 1, only keep threshold categories with the largest occurence frequecy, merge all other categories together.

    freq_col: str, default: "freq"
        Name of occurence frequency column.

    rank_col: str, default: None
        Name of occurence rank columns.

    condense_name: str, default: "Other"
        Name of the condense category.

    Return
    ------
    DataFrame with all least occurence categories condensed into a single category.
    """
    if type(threshold) == int:
        threshold=(threshold, None)
    if threshold[0] and df.shape[0] <= threshold[0]:
        return df

    index_name = df.index.name
    condense_df = df.reset_index()

    if threshold[0] and threshold[1]:
        # rank begin as 1
        mask = ((condense_df[freq_col] <= threshold[1])&
                (condense_df[freq_col].rank(method="first", ascending=False) > threshold[0]))
    elif threshold[0]:
        mask = condense_df[freq_col].rank(method="first", ascending=False) > threshold[0]
    elif threshold[1]:
        mask = condense_df[freq_col] <= threshold[1]

    condense_df[index_name] = _np.where(condense_df.index.isin(condense_df[mask].index), condense_name, condense_df[index_name])
    condense_df = condense_df[~mask].append(condense_df[mask].groupby(index_name).sum().reset_index()).set_index(index_name)

    if rank_col and condense_name in condense_df.index:
        condense_df.loc[condense_name, rank_col] = _np.nan
    return condense_df


# Describe 2D
def descibe_2d_category_data_extend(df_input, cate_col1, cate_col2, dropna=True, threshold_cate_1=9,
                                    threshold_cate_2=9):
    """
        Show describe table 2 categories data
        Args:
            df_input: dataframe contain 2 categories column
            cate_col1: category column 1
            cate_col2: category column 2
            dropna: if true, input data will drop na on columns cate_col1 and cate_col2
            threshold_cate_1: Maximum category show cate_col1 (all other will combine to Others category)
            threshold_cate_2: Maximum category show cate_col2 (all other will combine to Others category)
        Returns:
            Table describe data (count values, percentiles)
    """
    df_input = df_input.copy()
    if dropna:
        df_input = df_input.dropna(subset=[cate_col1, cate_col2])

    df_input[cate_col1] = _data_format_helper._format_limit_data_with_top_number_value(df_input, column_name=cate_col1,
                                                                                       ntop=threshold_cate_1)
    df_input[cate_col2] = _data_format_helper._format_limit_data_with_top_number_value(df_input, column_name=cate_col2,
                                                                                       ntop=threshold_cate_2)
    df_count = df_input.groupby([cate_col1, cate_col2])[cate_col2].size().unstack(fill_value=0)
    list_columns = list(df_count.columns)
    df_count.loc['Total'] = df_count.sum()
    df_percentile = _pd.DataFrame()
    for value in list_columns:
        df_percentile['percentile_hor_' + str(value)] = df_count[value] / df_count.sum(axis=1)
        df_percentile['percentile_hor_' + str(value)] = (df_percentile['percentile_hor_' + str(value)] * 100).round(2)
    for value in list_columns:
        df_percentile['percentile_ver_' + str(value)] = df_count[value] / df_count[value][:-1].sum(axis=0)
        df_percentile['percentile_ver_' + str(value)] = (df_percentile['percentile_ver_' + str(value)] * 100).round(2)

    df_count['Total'] = df_count.sum(axis=1)
    df_count['%Total'] = (df_count['Total'] / df_count['Total'][:-1].sum(axis=0) * 100).round(2)
    df_output = df_count.join(df_percentile)
    df_output = df_output.reset_index()
    df_output[cate_col1] = df_output[cate_col1].astype(str)
    df_output = df_output.rename(columns={cate_col1: cate_col1 + ' / ' + cate_col2})
    return df_output


def describe_2d_numeric_with_label(df, column_name, label='label'):
    """
        Show describe numeric data
        Args:
            df: dataframe
            column_name: name of cont features
            label: column label name
        Returns:
            Table describe data (total, std, pecentiles,...)
    """
    values = df[label].dropna().unique()
    df_describe = _pd.DataFrame({})
    for value in values:
        tmp = df.loc[df[label] == value, column_name].describe(
            percentiles=[i * 0.1 for i in range(10)] + [0.01] + [0.99]).reset_index()
        tmp.columns = ['Describe', 'Label_' + str(value)]
        if len(df_describe.columns) > 0:
            df_describe = _pd.merge(df_describe, tmp, on='Describe', how='inner')
        else:
            df_describe = tmp

    return df_describe


def describe_1d_category_with_percentiles(
        df, col,
        dropna=False, 
        threshold=(_CATE_THRESHOLD, None), 
        list_keep=None, 
        category_orders=None,
        null_value='nan',
        value_count_df=None,
        count_col=None,
        sort_by=None,
        **kwargs):
    """
    Describe a categorical variable with its value counts and the distribution of the value counts.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    col : str
        Name of a column to be analysed in the input dataframe.

    threshold : int or (int, int), default: (9, None)
        First value is rank threshold, second value is count threshold.
        As for rank threshold, threshold categories with the largest occurence frequecy are kept and the others are merged together.
        As for count threshold, all categories with occurence frequecy smaller than the threshold will be merged together.
        If an integer is provided, it is assigned to rank threshold and count threshold is set to None.

    list_keep : list of str, default: None
        List of categories to be kept regardless of threshold conditions.

    category_orders : dict, default None
        Key is column name and value is a list of ordered categories.

    null_value : str, default "nan"
        Category name representing null value which is brought to the end of the resulting value count dataframe.

    value_count_df : Pandas Dataframe
        Precomputed dataframe containing categories name as indices and a count column whose name is provided in `count_col` param.

    count_col : str, default: None
        Ignored if `value_count_df` is not provided. Name of the count column in the precomputed `value_count_df`.

    Return
    ------
    A tuple of value count dataframe and percentile dataframe."""
    if df is not None:
        df_temp = df[[col]].copy()
        df_temp = df_temp.fillna('nan')
    count_col = count_col or 'count'
    sort_by = sort_by or count_col
    if value_count_df is None:
        value_count_df = describe_category(df_temp, col, dropna=dropna, sort_by=sort_by, **kwargs)
    percentile_df = None
    if value_count_df.shape[0] > 9:
        # hardcode assume that value count df place the count col at the beginning
        percentile_df = describe_1d_numeric(value_count_df, count_col, fmt=',.0f', is_simple=True)

    # This is an adhoc way to make Plotly & Pybloqs work with nan value
    value_count_df.index = value_count_df.index.fillna("nan").map(str)

    # Remove unnecessary 00 trailing in time format
    if df is not None and _pd.api.types.is_datetime64_dtype(df_temp[col].dtype):
        value_count_df.index = value_count_df.index.map(lambda x: _re.sub("([: ]00)*$", "", x))

    # process value count
    if df is not None:
        value_count_df = (value_count_df.drop(columns=["rank"], errors='ignore')
                            .rename_axis(None)
                            .rename(columns={"count": _COUNT_COL_NAME,
                                            "freq": _FREQ_COL_NAME
                                            }))
    # cut
    if threshold is not None or list_keep:
        value_count_df = _cut_off_dataframe(
            value_count_df,
            list_keep=list_keep,
            threshold=threshold,
            null_value=str(null_value),
            # if sort_by is not provided, count_col is used to sort instead
            by=sort_by,
            ascending=kwargs.get('ascending', False)
        )
    # order
    if category_orders is not None:
        first = [x for x in category_orders if x in value_count_df.index]
        last = [x for x in value_count_df.index if x not in category_orders]
        value_count_df = value_count_df.loc[first+last]
    value_count_df = _bring_null_to_end(value_count_df, str(null_value))
    # handle invalid literal value for Int64 when converting value into string, Float64 solved.
    value_count_df = value_count_df.assign(**{count_col: lambda x: x[count_col].astype(float)})
    return value_count_df, percentile_df


def plot_1d_category(
        df, 
        count_col,
        text_col=None,
        feature_name='',
        threshold=None, 
        list_keep=None,
        fmt={},
        category_orders=None,
        category_rename=None,
        color_map = {},
        orientation = 'v',
        null_value='nan',
        show_percentile=True,
        exclude_in_plot=[],
        exclude_in_table=[],
        table_width=_TABLE_WIDTH,
        return_table=False,
        return_plot=False,
        plot_kwargs={}):
    df_temp = df.sort_values(count_col, ascending=False)
    if category_rename:
        df_temp.index = [category_rename.get(idx, idx) for idx in df_temp.index]
    df_describe, df_percentile = describe_1d_category_with_percentiles(
        df=None,
        col=None, 
        value_count_df=df_temp,
        threshold=threshold,
        list_keep=list_keep,
        category_orders=category_orders,
        null_value=null_value,
        count_col=count_col
        )
    df_describe = df_describe.loc[[idx for idx in df_describe.index if idx not in exclude_in_table]]
    if 'Total' in exclude_in_table:
        table_total_row = False
    else:
        table_total_row = True
    formatters = {}
    if fmt:
        formatters.update(fmt)
    table = _describe_utils.format_table_cate_full(df_describe, 
                                                null_value=str(null_value),
                                                fmt=formatters,
                                                table_index_name=feature_name,
                                                table_width=table_width,
                                                append_total_row=table_total_row
                                                )
    # fig
    # construct color map
    custom_color_map = {str(null_value): _NULL_COLOR}
    casted_color_map = dict()
    for key, value in color_map.items():
        casted_color_map[str(key)] = value
    custom_color_map.update(casted_color_map)
    generated_color_map = _describe_utils.gen_color_map(df_describe.index, _MAIN_COLOR, custom_color_map)

    if exclude_in_plot:
        # exclude_in_plot = [idx for idx in exclude_in_plot if idx in df_describe.index]
        df_describe = df_describe.drop(index=[exclude_in_plot], errors='ignore')
    
    fig_layout = {
        'title': f"Distribution of {feature_name}",
        'width': _FIG_WIDTH,
        'height': _FIG_HEIGHT,
        'showlegend':False
    }

    if orientation == 'v':
        fig_layout['xaxis_title'] = feature_name
        fig_layout['yaxis_title'] = 'count'
        fig_layout.update(plot_kwargs)
        fig = _px.bar(df_describe, 
                    orientation=orientation,
                    x=df_describe.index, 
                    y=count_col,
                    text=text_col,
                    color=df_describe.index,
                    color_discrete_map=generated_color_map)
        fig.update_xaxes(ticks='')
        fig.update_traces(
            hovertemplate=f"{feature_name}=%{{x}}<br>count=%{{y}}"
        )
    elif orientation == 'h':
        fig_layout['xaxis_title'] = 'count'
        fig_layout['yaxis_title'] = feature_name
        fig_layout.update(plot_kwargs)
        fig = _px.bar(df_describe, 
                    orientation=orientation,
                    x=count_col, 
                    y=df_describe.index,
                    text=text_col,
                    color=df_describe.index,
                    color_discrete_map=generated_color_map)
        fig.update_traces(
            hovertemplate=f"{feature_name}=%{{y}}<br>count=%{{x}}")
    else:
        raise Exception(f'Invalid value for orientation. Expected `h` or `v`, received {orientation}')

    fmt_text_col = fmt.get(text_col, None)
    if fmt_text_col is not None or fmt_text_col=='percent_html':
        fmt_text_col = fmt_text_col or '.2f'
        fig.update_traces(texttemplate=f'%{{text:{fmt_text_col}}}')
    
    # update axes title
    fig.update_layout(**fig_layout)
    if orientation == 'h' and show_percentile:
        fig.update_yaxes(
            ticks=''
        ).update_layout(
            margin=dict(l=100)
        )
    
    if return_table:
        return table
    elif return_plot:
        return fig

    # format percentile df
    if df_percentile is not None and show_percentile:
        fmt_seperator = _tf.FmtThousandSeparator(0, rows=[idx for idx in df_percentile.index if idx not in ['mean', 'std']])
        table_percentile = _Block(df_percentile, formatters=[_FMT_FONT, fmt_seperator])
        return _CustomHStack([table, table_percentile, fig], grid_cell_style='margin-right:20px')
    else:
        return _CustomHStack([_Block(table, width=_TABLE_WIDTH), fig])
        

# Describe with plot 1D
def describe_1d_category_with_plot(
        df, 
        col,
        threshold=(9, None),
        list_keep=None,
        category_rename=None,
        category_orders=None,
        dropna=False,
        fillna=None,
        null_value=float('nan'),
        show_percentile=True,
        sort_by=None,
        ascending=False,
        orientation = None,
        plot_kwargs={},
        color_map={},
        table_width=_TABLE_WIDTH,
        table_index_name=None,
        exclude_in_table=[],
        exclude_in_plot=[],
        precomputed_kwargs=None,
        return_table=False,
        return_raw=False,
        return_json=False,
        **kwargs):
    """
    Describe a categorical column.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    col : str
        Name of a column to be analysed in the input dataframe.

    threshold : int or (int, int), default: (9, None)
        - First value is rank threshold, threshold categories with the largest occurence frequecy are kept and the others are merged together.
        - Second value is count threshold, all categories with occurence frequecy smaller than the threshold will be merged together.
        If an integer is provided, it is assigned to rank threshold and count threshold is set to None.

    list_keep : list, default: None
        List of categories to be kept regardless of threshold conditions.

    category_rename : dict, default None
        Key is column name and value is a dict to map categories in the column into new names. If a category is not given in the value dict, its original name is used.
    
    category_orders : dict, default None
        Key is column name and value is a list of values in the desired order. This is used to override the default category ordering behaviour in which categories are sorted in descending order according to value counts.

    dropna : boolean, default: False
        If False, NA values will be treated as a category, dropped otherwise.

    fillna : scalar, default: None
        Fill NA/NaN values using the specified value. 

    null_value : str, default "nan"
        Category name representing null value which is brought to the end of the resulting value count dataframe.
    
    show_percentile : boolean, default True
        Whether to show percentile dataframe which describes the distribution of value counts.

    sort_by : str in ("count", "index"), default None
        If None or "count" the value count dataframe is sorted based on value counts.
        If "index" the value count dataframe is sorted based on category names.

    ascending : boolean, default False
        If True, the value count dataframe is sorted in ascending order based on `sort_by`
        If False, the value count dataframe is sorted in descending order based on `sort_by`

    orientation : str in ("h", "v"), default "h"
        Orientation of the bar plot. "h" mean horizontal and "v" mean vertical

    plot_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` or `update_traces()` function of the plotly figure.

    color_map : dict, default {}
        Key is category name and value is color. String values should define valid CSS-colors used to override
        coloring behaviour to assign a specific colors to marks corresponding with specific categories.
    
    table_width : str, default "320px"
        Size of table in string. This is passed to Block class of pybloqs. In `col` or category names are too long that the table is partly overlapped by other table or plot, consider increasing table_width or set another `table_index_name`. 
    
    table_index_name : str, default None
        Used as index name of the table. If None, `col` is used.

    exclude_in_table : list, default []
        Categories to exclude from the resulting table. Specific catefories not presented in the column can also be passed such as "nan", "Others", "Total".

    exclude_in_plot : list, default []
        Categories to exclude from the resulting plot. Specific catefories not presented in the column can also be passed such as "nan", "Others".

    precomputed_kwargs : dict, default None
        value_count_df : Pandas Dataframe
            Precomputed dataframe containing categories name as indices and a count column whose name is provided in `count_col` param.
        count_col : str, default: None
            Name of the count column in the precomputed `value_count_df`.
        text_col : str, default: None
            Name of the text column in the precomputed `value_count_df`.
        fmt : dict, default {}
            Key is column name and value is formatter for values in the corresponding column e.g {"age": ",.0f"}. See https://github.com/d3/d3-format/blob/main/README.md#locale_format for more format options. A special formatter supported by zeda2 is "html_percent".

    return_table : boolean, default False
        Whether to return the formatted table only.

    return_fig : boolean, default False
        Whether to return the plot only.

    return_json : boolean, default False
        Whether to return the analysis in json format. This is specifically designed for internal use.

    Return
    ------
    HStack containing value count table, plot and possibly percentile table."""
    if category_orders and col in category_orders:
        orders = category_orders[col]
    else:
        orders = None
    df_temp = None
    value_count_df=None
    fmt = _describe_utils._CATE_DEFAULT_FORMATTER
    list_keep_col=[_COUNT_COL_NAME, _FREQ_COL_NAME]
    text_col = _FREQ_COL_NAME
    count_col = _COUNT_COL_NAME
    # get cleaned value_count_df
    if df is not None and col:
        df_temp = _util_category_rename(df[[col]].copy(), category_rename, [col])
        table_index_name = table_index_name or col
        fillna = fillna or 'nan'
        if fillna:
            df_temp[col] = df_temp[col].fillna(fillna)
    elif precomputed_kwargs is not None:
        text_col = precomputed_kwargs.get('text_col', _FREQ_COL_NAME)
        count_col = precomputed_kwargs.get('count_col', _COUNT_COL_NAME)
        value_count_df = precomputed_kwargs['value_count_df'].sort_values(count_col, ascending=False)
        fmt = precomputed_kwargs.get('fmt', {})
        # if add some cols to plot hover but dont keep them in table
        list_keep_col = precomputed_kwargs.get('list_keep_col', value_count_df.columns)
    df_describe, df_percentile = describe_1d_category_with_percentiles(
        df_temp, col,
        dropna=dropna,
        threshold=threshold, 
        list_keep=list_keep, 
        category_orders=orders, 
        null_value=str(null_value), 
        value_count_df=value_count_df,
        count_col=count_col,
        sort_by=sort_by,
        ascending=ascending,
        **kwargs)
    # initialize orientation if not provided
    if orientation is None:
        max_length = max(df_describe.index.map(str).map(len))
        if max_length > 10:
            orientation = 'h'
            # if axis and yaxis are provided together with None orientation, default the axis titles are set in vertical mode, now switch them
            if 'xaxis_title' in plot_kwargs or 'yaxis_title' in plot_kwargs:
                old_xtitle = plot_kwargs.get('xaxis_title', None)
                old_ytitle = plot_kwargs.get('xaxis_title', None)
                if old_xtitle is not None:
                    plot_kwargs['yaxis_title'] = old_xtitle
                if old_ytitle is not None:
                    plot_kwargs['xaxis_title'] = old_ytitle
        else:
            orientation = 'v'
    # table
    if 'Total' in exclude_in_table:
        table_total_row = False
    else:
        table_total_row = True

    df_describe.rename_axis(table_index_name, axis=1, inplace=True)
    # query list_keep_cols for table
    table = _describe_utils.format_table_cate_full(
        df_describe.loc[[idx for idx in df_describe.index if idx not in exclude_in_table], list_keep_col],
        null_value=str(null_value), 
        table_width=table_width,
        table_index_name=table_index_name,
        append_total_row=table_total_row,
        fmt=fmt
        )
    # fig
    # construct color map
    custom_color_map = {str(null_value): _NULL_COLOR}
    casted_color_map = dict()
    for key, value in color_map.items():
        casted_color_map[str(key)] = value
    custom_color_map.update(casted_color_map)
    generated_color_map = _describe_utils.gen_color_map(df_describe.index, _MAIN_COLOR, custom_color_map)

    fig_layout = {
        'title': f"Distribution of {col}",
        'width': _FIG_WIDTH,
        'height': _FIG_HEIGHT,
        'showlegend':False
    }

    # we can reassign this as no further operation on df_describe will be made
    df_describe_plot = df_describe
    if exclude_in_plot:
        df_describe_plot = df_describe.loc[[idx for idx in df_describe.index if idx not in exclude_in_plot]]

    fig_layout.update(plot_kwargs)
    custom_bar_props, _, _ = _split_plotly_layout_trace_props(fig_layout)
    text_fmt = fmt.get(text_col, ',.2f')
    count_fmt = fmt.get(count_col, ',.2f')
    if text_fmt == 'html_percent':
        text_fmt = '.2%'
    if count_fmt == 'html_percent':
        count_fmt = '.2%'
    if orientation == 'v':
        fig_layout['xaxis_title'] = col
        fig_layout['yaxis_title'] = count_col
        fig_layout.update(plot_kwargs)
        fig = _px.bar(
            df_describe_plot, 
            orientation=orientation,
            x=df_describe_plot.index, 
            y=count_col,
            text=text_col, 
            color=df_describe_plot.index,
            color_discrete_map=generated_color_map,
            **custom_bar_props
            )
        fig.update_xaxes(ticks='')
        fig.update_traces(
            hovertemplate=f"{col}=%{{x}}<br>{count_col}=%{{y:{count_fmt}}}<br>{text_col}=%{{text:{text_fmt}}}",
            texttemplate=f'%{{text:{text_fmt}}}'
        )
    elif orientation == 'h':
        fig_layout['xaxis_title'] = count_col
        fig_layout['yaxis_title'] = col
        fig_layout.update(plot_kwargs)
        fig = _px.bar(
            df_describe_plot, 
            orientation=orientation,
            x=count_col, 
            y=df_describe_plot.index,
            text=text_col,
            color=df_describe_plot.index,
            color_discrete_map=generated_color_map,
            **custom_bar_props
            )
        fig.update_traces(
            hovertemplate=f"{col}=%{{y}}<br>{count_col}=%{{x:{count_fmt}}}<br>{text_col}=%{{text:{text_fmt}}}",
            texttemplate=f'%{{text:{text_fmt}}}')
    else:
        raise Exception(f'Invalid value for orientation. Expected `h` or `v`, received {orientation}')
    # update layout, traces, axes title
    _, custom_layout_props, custom_traces_props = _split_plotly_layout_trace_props(fig_layout)

    fig.update_layout(**custom_layout_props).update_traces(**custom_traces_props)
    if orientation == 'h' and show_percentile:
        fig.update_yaxes(
            ticks=''
        ).update_layout(
            margin=dict(l=100)
        )

    if return_table:
        return table
    if return_raw:
        return [df_describe, df_percentile, fig]

    if return_json:
        # get index name of json table
        if table_index_name is None:
            table_index_name = col
        # query list_keep_cols for table
        html_table = _describe_utils.format_table_cate_html(
            df_describe[list_keep_col], 
            table_index_name, 
            null_value=str(null_value), 
            fmt=fmt)
        json_table = _describe_utils.df_to_json(html_table)
        json_fig = _get_plotly_fig_json(fig)
        return json_table, None, json_fig
        
    # format percentile df
    if df_percentile is not None and show_percentile:
        fmt_seperator = _tf.FmtThousandSeparator(0, rows=[idx for idx in df_percentile.index if idx not in ['mean', 'std']])
        table_percentile = _Block(df_percentile, formatters=[_FMT_FONT, fmt_seperator])
        return _CustomHStack([table, table_percentile, fig], grid_cell_style='margin-right:20px')
    else:
        return _CustomHStack([_Block(table, width=_TABLE_WIDTH), fig])


def _describe_1d_numeric_get_plot(
        df, col,
        show_density=True, 
        show_box=True, 
        nbins=_NUM_DEFAULT_NBINS,
        plot_kwargs={}, 
        binsize=None, 
        is_trim=False,
        kde_npoints=_KDE_DEFAULT_NPOINTS,
        fmt=None,
        # precomputed params
        bin_edges=None,
        bin_values=None,
        df_percentile=None,
        **kwargs
        ):
    '''
        df_percentile: Dataframe containing precomputed percentile data for box plot with the following mandatory indices: "min", "max", "mean", "25%", "50%", "75%"
    '''
    # histogram plot
    if df is not None and col:
        bin_edges, bin_values, bin_centers, binsize = _describe_utils.calculate_bin(df[col], nbins=nbins, binsize=binsize, **kwargs)
    elif bin_edges and bin_values and len(bin_edges) >= 2:
        binsize = bin_edges[1] - bin_edges[0]
        bin_centers = []
        for idx in range(len(bin_edges)-1):
            bin_centers.append((bin_edges[idx] + bin_edges[idx+1])/2)
    else:
        # raise Exception('describe._describe_1d_numeric_get_plot: not enough data to describe numeric.')
        bin_centers = []
        bin_values = []
        bin_edges = [-float('nan'), float('nan')]
        binsize = 0
    
    if len(bin_edges)>=2:
        text = [f'{col}={left_edge}-{right_edge}' for left_edge, right_edge in zip(bin_edges[:-1],bin_edges[1:])]
    else:
        text = []
    # start plotting
    fig = _make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(_go.Bar(x=bin_centers, 
                        y=bin_values, 
                        name='count',
                        hovertext=text, 
                        hovertemplate='%{hovertext}<br>count=%{y:.0f}'))
    fig.update_xaxes(
        tick0=bin_edges[0],
        dtick=binsize*_math.ceil(len(bin_edges)/10)
    )
    fig.update_layout(
        bargap=0.01,
        width=620,
        height=480
    )
    # kde plot
    if show_density and df is not None and col:
        if df[col].dropna().shape[0] > 1:
            kde_plot = _describe_utils.make_kde(df[col].dropna(), n_points=kde_npoints)
            fig.add_trace(kde_plot, secondary_y=True)
    # box plot
    if show_box:
        if df is not None and col:
            box_plot = _describe_utils.get_box_trace(df[col], color=_MAIN_COLOR, fmt=fmt)
        elif df_percentile is not None:
            box_plot = _describe_utils.get_box_trace(
                series=None, 
                series_percentile=df_percentile['value'], 
                color=_MAIN_COLOR, fmt=fmt)
        else:
            box_plot = _go.Figure()
        box_plot.update(xaxis="x3", yaxis="y3", name='', showlegend=False)
        fig.add_trace(box_plot)
    # set default title
    # update layout
    fig.layout.xaxis = {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': col}}
    fig.layout.xaxis3 = {'anchor': 'y3', 'domain': [0.0, 1.0], 'matches': 'x', 'showticklabels': False}
    fig.layout.yaxis = {'anchor': 'x', 'domain': [0.0, 0.8316], 'title': {'text': 'count'}}
    fig.layout.yaxis3 = {'anchor': 'x3',
                'domain': [0.8416, 1.0],
                'matches': 'y3',
                'showline': False,
                'showticklabels': False,
                'ticks': ''}
    if is_trim:
        title = f"Trimmed distribution of {col}"
    else:
        title = f"Distribution of {col}"
    fig_layout = {
        'title': title,
        'xaxis_title': col,
        'yaxis_title': 'count',
        'width': 570,
        'height': 480,
        'legend': dict(
            title="",
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="right",
            x=0.95)
        }
    fig_layout.update(plot_kwargs)
    fig.update_layout(
        **fig_layout
    )

    if show_density:
        fig.data[1].hovertemplate=f"{col}=%{{x:.2f}}<br>kde=%{{y:.4f}}"
        fig.layout.yaxis2.update(title_text='density')
    return fig


def describe_1d_numeric_with_plot(
        df, col,
        percentiles=None,
        dropna=False,
        nbins=_NUM_DEFAULT_NBINS,
        fmt=',.2f',
        show_box=True,
        show_density=True,
        kde_npoints=_KDE_DEFAULT_NPOINTS,
        plot_mode='trim',
        trim_percentiles=(0.01, 0.99),
        trim_range=(None, None),
        trim_inclusive="both",
        table_simple=False,
        plot_base_kwargs={},
        plot_trim_kwargs={},
        precomputed_kwargs=None,
        return_json=False,
        return_raw=False,
        return_table=False,
        **kwargs
        ):
    """
    Describe a numeric column.
    Contributor: ThanhNM3, QuanHM

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    col : str
        Name of a column to be analysed in the input dataframe.

    percentiles : list, default [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 99]
        The percentiles to include in the percentile table. All should fall between 0 and 1.

    dropna : boolean, default: False
        If False, NA values will be treated as a category, dropped otherwise.

    nbins : int, default 50
        Number of bins to split data into, used in histogram plot. The resulted number of bins are not necessarily equal to the provided nbins but are rounded for smoother visualization.
    
    fmt : str, default ",.2f"
        Formatter for values in the column. See https://github.com/d3/d3-format/blob/main/README.md#locale_format for more format options.
    
    show_box : boolean, default True
        Whether to show box plot which is a part of the resulting figure.
        
    show_density : boolean, default True
        Whether to show kernel density estimation plot which is a part of the resulting figure.
    
    kde_npoints : int, default 300,
        The number points to plot kde plot. Too many points may lead to slow loading of the figure.
    
    plot_mode : str in ("base", "trim", "both"), optional, default: "default"
        "base": plot default numeric distribution.
        "trim": plot trim (remove value outside specified quantiles) numeric distribution.
        "both": plot both the default and trim numeric distribution.
    
    trim_percentiles : (float, float), optional, default: (0.01, 0.99)
        Quantile boundaries. The value outside the boundaries will be excluded when draw Trim plot.
    
    trim_range : (scalar, scalar), default (None, None)
        Value boundaries. The value outside the boundaries will be excluded when draw Trim plot.
    
    trim_inclusive : str in (“both”, “neither”, “left”, “right”), default: "both"
        Include boundaries. Whether to set each bound as closed or open.
    
    table_simple : boolean, default False
        Whether to show statistic table in simple mode. If True, statistics table is shown with only 1 column and less statistics are shown. Otherwise, all computed statistics will be shown in 2 columns.
    
    plot_base_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the base plot.
    
    plot_trim_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the trim plot.
        
    precomputed_kwargs : dict, default {}
        Designed for internal use.

    return_table : boolean, default False
        Whether to return the formatted table only.

    return_fig : boolean, default False
        Whether to return the plot only.

    return_json : boolean, default False
        Whether to return the analysis in json format. This is specifically designed for internal use.

    Return
    ------
    HStack containing statistics table and plot."""
    _px.defaults.template = 'simple_white'
    # validate
    if ((trim_percentiles[0] and (trim_percentiles[0] > 1 or trim_percentiles[0] < 0)) or 
        (trim_percentiles[1] and (trim_percentiles[1] > 1 or trim_percentiles[1] < 0))):
        raise Exception("Invalid value for `trim_percentiles`. Valid value is a tuple with each of its element in the range [0, 1].")
    if trim_percentiles[0] and trim_percentiles[1] and trim_percentiles[0] >= trim_percentiles[1]:
        raise Exception("Invalid value for `trim_percentiles`. First percentile must be smaller than the second one.")
    if trim_range[0] and trim_range[0] and trim_range[0] >= trim_range[1]:
        raise Exception("Invalid value for `trim_range`. First value must be smaller than the second one.")        
    if plot_mode not in ['base', 'trim', 'both']:
        raise Exception("Invalid value for `plot_mode`. Valid values are in ['base', 'trim', 'both']")
    if plot_mode != 'base' and trim_inclusive not in ['both', 'left', 'right', 'none', None]:
        raise Exception("Invalid value for `trim_inclusive`. Valid values are in ['both', 'left', 'right', 'none', None]")

    df_temp = None
    if precomputed_kwargs:
        if 'df_percentile' in precomputed_kwargs:
            df_percentile = precomputed_kwargs['df_percentile']
        if 'hist_bin_edges' in precomputed_kwargs:
            bin_edges = precomputed_kwargs['hist_bin_edges']
        if 'hist_bin_values' in precomputed_kwargs:
            bin_values = precomputed_kwargs['hist_bin_values']
    # dropna
    if df is not None and col:
        df_temp = df[[col]].copy()
        if dropna:
            df_temp[col] = df_temp[col].dropna()
        df_temp[col] = _describe_utils.series_cast_numeric(df_temp[col])
        # table
        df_stat = describe_1d_numeric(df_temp, col, percentiles=percentiles, fmt=fmt, is_simple=table_simple)
    elif df_percentile is not None:
        df_stat = describe_1d_numeric(
            None, col, 
            percentile_df=df_percentile, 
            percentiles=percentiles, 
            fmt=fmt, 
            is_simple=table_simple)
    # format the table
    df_stat.rename_axis('stat', axis=1, inplace=True)
    table = _describe_utils.format_table_num_full(df_stat)
    # fig
    fig_base = None
    fig_trim = None
    if plot_mode == 'both' or plot_mode == 'base':
        if df_temp is not None:
            fig_base = _describe_1d_numeric_get_plot(
                df=df_temp, 
                col=col, 
                nbins=nbins, 
                show_density=show_density, 
                show_box=show_box, 
                plot_kwargs=plot_base_kwargs,
                kde_npoints=kde_npoints,
                fmt=fmt,
                **kwargs
                )
        elif bin_edges is not None and bin_values is not None:
            fig_base = _describe_1d_numeric_get_plot(
                df=None, 
                col=col, 
                nbins=nbins, 
                show_density=False,
                show_box=show_box, 
                plot_kwargs=plot_base_kwargs,
                fmt=fmt,
                bin_edges=bin_edges, 
                bin_values=bin_values, 
                df_percentile=df_percentile,
                **kwargs
                )
    if plot_mode == 'both' or plot_mode == 'trim':
        s = df_temp[col]
        additive_title = ''
        # get start and end value of trimmed column
        if trim_range[0] or trim_range[1]:
            start=trim_range[0]
            end=trim_range[1]
            if start is None:
                start = s.min()
            if end is None:
                end = s.max()
            additive_title = f'<br><span style="font-size:14px">Range: @{start}, {end}#</span>'
        else:
            if trim_percentiles[0] is None:
                trim_percentiles[0] = 0
            if trim_percentiles[1] is None:
                trim_percentiles[1] = 1
            start, end = _describe_utils.series_get_percentiles(df_temp[col], trim_percentiles)
            additive_title = f'<br><span style="font-size:14px">Percentiles: @{trim_percentiles[0]}, {trim_percentiles[1]}#</span>'
        
        # trim the column
        if trim_inclusive == 'both':
            df_temp = df_temp.loc[(s>=start)&(s<=end), :]
            additive_title = additive_title.replace("@", '[').replace('#', ']')
        elif trim_inclusive == 'left':
            df_temp = df_temp.loc[(s>=start)&(s<end), :]
            additive_title = additive_title.replace("@", '[').replace('#', ')')
        elif trim_inclusive == 'right':
            df_temp = df_temp.loc[(s>start)&(s<=end), :]
            additive_title = additive_title.replace("@", '(').replace('#', ']')
        else:
            df_temp = df_temp.loc[(s>start)&(s<end), :]
            additive_title = additive_title.replace("@", '(').replace('#', ')')
        # if df_temp.shape[0] == 0:
        #     raise Exception("Invalid trimmed_range. Trimmed dataframe is empty.")
        fig_trim = _describe_1d_numeric_get_plot(
            df=df_temp, 
            col=col, 
            nbins=nbins,
            show_density=show_density, 
            show_box=show_box, 
            plot_kwargs=plot_trim_kwargs,
            kde_npoints=kde_npoints,
            is_trim=True,
            fmt=fmt
            )
        trimmed_title = f'{fig_trim.layout.title.text}{additive_title}'
        fig_trim.update_layout(
            title=trimmed_title
        )
    # return fig
    if return_raw:
        def bold_column(s, column_name):
            if s.name == column_name:
                return ['font-weight: bold;border-left: 1px dashed black'] * len(s)
            else:
                return [''] * len(s)
                
        if not table_simple:
            df_stat.columns = ["value", "percentile", "value "]

            border_props = [
                ("border-left-color", "black"),
                ("border-left-style", "dashed"),
                ("border-left-width", "thin"),
            ]
            df_stat = (df_stat
                .rename_axis("stat", axis=1)
                .style
                .apply(bold_column, column_name='percentile', axis=0)
                .set_table_styles([dict(selector='th.col_heading.level0.col1', props=border_props)])
                )
        if plot_mode=='both' or plot_mode=='trim':
            return [df_stat, fig_trim]
        else:
            return [df_stat, fig_base]
    
    if return_table:
        return table
    
    if return_json:
        # different format functions for simple and not simple table
        if len(df_stat.columns)==3:
            html_table = _describe_utils.format_table_num_html(df_stat)
        elif len(df_stat.columns)==1:
            html_table = _describe_utils.format_table_cc_html(df_stat)
        json_table = _describe_utils.df_to_json(html_table)
        if fig_trim:
            json_fig = _get_plotly_fig_json(fig_trim)
        else:
            json_fig = _get_plotly_fig_json(fig_base)
        return json_table, json_fig
    
    return _CustomHStack([table, 
                        _CustomHStack([f for f in [fig_trim, fig_base] if f is not None], 
                                        grid_cell_style='magin-right:0px')], grid_cell_style='magin-right:20px')

# 1D DATETIME
def describe_1d_datetime_with_plot(
        df, col,
        datetime_format='%Y-%m-%d',
        plot_kwargs=None,
        precomputed_kwargs=None,
        return_json=False,
        **kwargs):
    """
    Draw heatmap by year and month of a datetime column in string type.

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    col : str
        Name of a column to be analysed in the input dataframe. Value of this column must be in string type.

    datetime_format : str, default "%Y-%m-%d"
        Python datetime format. See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for more options.

    plot_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the plotly figure.
        
    precomputed_kwargs : dict, default {}
        Designed for internal use.

    return_json : boolean, default False
        Whether to return the analysis in json format. This is specifically designed for internal use.

    Return
    ------
    HStack containing heatmap by year and month."""
    # initialize params if not provided
    plot_kwargs = plot_kwargs or {}
    # clone df
    if df is not None and col:
        df_temp = df[[col]].copy()
        df_temp['year_month'] = df[col].apply(lambda x: _datetime.strptime(x, datetime_format).strftime('%Y/%m'))
        df_temp[['year', 'month']] = df_temp['year_month'].str.split('/', expand=True)
        df_pivot = (df_temp
            .groupby(['year', 'month']).size()
            .reset_index().rename(columns={0: 'count'})
            .pivot(index='year', columns='month', values='count'))
    elif precomputed_kwargs and 'df_count' in precomputed_kwargs:
        df_temp = precomputed_kwargs['df_count']
        df_temp[['year', 'month']] = df_temp['year_month'].str.split('/', expand=True)
        df_pivot = (df_temp
            .pivot(index='year', columns='month', values='count'))
    df_perc = df_pivot / df_pivot.values.sum()
    default_kwargs = {
        'title': f'Heatmap of <b>{col}</b> by year and month',
        'xaxis_title': 'month',
        'yaxis_title': 'year',
        'width': max(400, 55*df_pivot.shape[1]+270),
        'height': max(300, 50*df_pivot.shape[0]+200)
    }
    default_kwargs.update(plot_kwargs)        
    fig = _describe_utils.plot_heatmap(
        df_pivot, 
        plot_kwargs=default_kwargs,
        customdata=df_perc,
        hovertemplate='year month: %{y}/%{x}<br>count: %{z:,.0f}<br>frequency: %{customdata:.2%}',
        **kwargs
    )
    if return_json:
        return _get_plotly_fig_json(fig)
    else:
        return fig


# Describe with plot 2D
def describe_2d_category_with_label_with_plot(df, cate_column_name, label_column_name="label", **kwargs):
    """
        Describe 2D data and plot histogram chart of category data
        Arguments:
            df: dataframe has column
            cate_column_name: name of category column 1
            label_column_name: name of category column 2
            kwargs: additional layout config for plotly, any param in function .update_layout()
    """
    df_groupby = descibe_2d_category_data_extend(df, cate_column_name, label_column_name)
    fig = _plotly_plot_2d_category_with_label(df, df_groupby, cate_column_name, label_column_name, **kwargs)
    _display_df_with_title(df_groupby, title="Describe " + cate_column_name + " with " + label_column_name)
    fig.show()


def describe_2d_numeric_with_label_with_plot(df, column_name, label='label'):
    """
        Describe 2D data and plot histogram chart of category data
        Arguments:
            df: dataframe has column
            column_name: name of continuous column
            label: name of label column
    """
    df_groupby = describe_2d_numeric_with_label(df, column_name, label)
    fig, ax = _sns_plot_2d_numeric_with_label(df, column_name, label)

    _display_df_with_title(df_groupby, title="Describe " + column_name + " with " + label)
    _plt.show()


def _get_condense_cate_map(df, column_name, threshold, dropna, value_count_df=None, **kwargs):
    if value_count_df is None:
        value_count_df = describe_category(df, column_name, dropna=dropna, **kwargs)
    else:
        value_count_df = value_count_df.sort_values('count', ascending=False)
    cate_values = set(value_count_df.index)

    # Not support condense category if column-type is datetime-like
    # if threshold is not None and not _pd.api.types.is_datetime64_dtype(df[column_name].dtype):
    if threshold is not None:
        value_count_df = _describe_utils.cut_off_dataframe(value_count_df, list_keep=None, threshold=threshold, null_value=None)

    # value_count_df = (
    #     value_count_df
    #         .reset_index()
            # .loc[:, [column_name, "freq"]]
            # .rename(columns={"freq": "%"})
    # )

    keep_cate_values = cate_values.intersection(set(value_count_df.index))
    merged_cate_values = cate_values.difference(set(value_count_df.index))
    cate_map = {k: "Others" for k in merged_cate_values}
    cate_map.update({k: k for k in keep_cate_values})
    return cate_map, value_count_df.reset_index()
    

def describe_2d_category(
        df, 
        cate1, cate2,
        threshold1=(9, None),
        threshold2=(9, None),
        category_rename=None,
        category_orders=None,
        dropna=False, 
        fillna1=None,
        fillna2=None,
        encode_target=False,
        sort_by="value", 
        ascending=False,
        precomputed_kwargs=None,
        return_json=False,
        **kwargs
        ):
    """
    Return value count dataframe by two categorical columns.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    cate1 : str
        Name of the first column to be analysed in the input dataframe.
    
    cate2 : str
        Name of the second column to be analysed in the input dataframe

    threshold : int, default: 9
        Rank threshold threshold categories with the largest occurence frequecy are kept and the others are merged together.

    dropna : boolean, default: False
        If False, NA values will be treated as a category, dropped otherwise.

    sort_by : str in ("value", "index"), default None
        If None or "count" the value count dataframe is sorted based on value counts.
        If "index" the value count dataframe is sorted based on category names.

    ascending : boolean, default False
        If True, the value count dataframe is sorted in ascending order based on `sort_by`
        If False, the value count dataframe is sorted in descending order based on `sort_by`

    encode_target : boolean, default: False
        Whether to map target variable to increasing number, for cleaner visualization.

    Return
    ------
    Value count dataframe."""
    if df is not None and cate1 and cate2:
        df_temp = df[[cate1, cate2]].copy()
        if dropna:
            df_temp = df_temp.dropna(subset=[cate1, cate2])
    elif precomputed_kwargs is not None:
        value_count_df1 = precomputed_kwargs['value_count_df1'].sort_values('count', ascending=False)
        value_count_df2 = precomputed_kwargs['value_count_df2'].sort_values('count', ascending=False)
        value_count_df1.index = value_count_df1.index.map(str)
        value_count_df2.index = value_count_df2.index.map(str)
        df_temp = precomputed_kwargs['df_count'].copy()
    # rename category
    if category_rename and cate1 in category_rename:
        df_temp[cate1] = df_temp[cate1].apply(lambda x: category_rename[cate1].get(x, x))
    if category_rename and cate2 in category_rename:
        df_temp[cate2] = df_temp[cate2].apply(lambda x: category_rename[cate2].get(x, x))
    # fillna
    if fillna1:
        df_temp[cate1] = df_temp[cate1].fillna(fillna1)
    if fillna2:
        df_temp[cate2] = df_temp[cate2].fillna(fillna1)
    # process raw data
    df_temp = df_temp.assign(**{cate1: lambda x: x[cate1].astype(str), # WARNING
                                cate2: lambda x: x[cate2].astype(str)})
    # target_values = sorted(df_temp[col2].unique())
    if precomputed_kwargs is None:
        cate_map1, value_count_df1 = _get_condense_cate_map(df_temp, cate1, threshold1, dropna, None, **kwargs)
        cate_map2, value_count_df2 = _get_condense_cate_map(df_temp, cate2, threshold2, dropna, None, **kwargs)
    else:
        cate_map1, value_count_df1 = _get_condense_cate_map(None, cate1, threshold1, dropna, value_count_df1, **kwargs)
        cate_map2, value_count_df2 = _get_condense_cate_map(None, cate2, threshold2, dropna, value_count_df2, **kwargs)
    # get target value after being cut off in descending order by sum
    target_values = value_count_df2[cate2]
    # keep of col2
    list_keep_cate2 = list(value_count_df2[cate2])
    if 'Others' in list_keep_cate2:
        list_keep_cate2.remove('Others')
    # describe total
    df_describe_total = value_count_df2.drop(columns=['rank'], errors='ignore').copy()
    df_describe_total[cate1] = 'Total'
    df_describe_total['%'] = 1
    # unstack table
    table_total = (df_describe_total
            .groupby([cate1, cate2, "%"], sort=False)
            .first()
            .unstack([cate2], fill_value=0)
            .reset_index().set_index(cate1))

    target_map = None
    if encode_target:
        target_map = {v: str(k) for k, v in enumerate(target_values)}
        target_values = [str(x) for x in range(len(target_values))]  # avoid continous legend color map in Plotly
        df_temp = df_temp.assign(**{cate2: lambda x: x[cate2].map(target_map)})
        target_map = _pd.DataFrame(target_map.items(), columns=[f"original_{cate2}", "encode"])
    
    if precomputed_kwargs is None:
        df_count = (
            df_temp.loc[:, [cate1, cate2]]
                .assign(**{cate1 : lambda x: x[cate1].map(cate_map1)})
                .assign(**{cate2 : lambda x: x[cate2].map(cate_map2)})
                .groupby([_pd.Grouper(key=cate1, **kwargs), cate2], dropna=dropna)
                .size().to_frame("count")
                .reset_index()
                .groupby(cate1, as_index=False)
                .apply(lambda df_temp:
                    df_temp.assign(**{"freq": lambda x: (x["count"]/sum(x["count"]))})
                    )
                .merge((value_count_df1
                        .drop(columns=['count', 'rank'])
                        .rename(columns={'freq': '%'})
                        ), on=cate1, how="left")
        )
    else:
        df_count = (
            df_temp
            .assign(**{cate1 : lambda x: x[cate1].map(cate_map1)})
            .assign(**{cate2 : lambda x: x[cate2].map(cate_map2)})
            .groupby([_pd.Grouper(key=cate1, **kwargs), cate2], dropna=dropna)
            .agg({'count': 'sum'})
            .reset_index()
            .groupby(cate1, as_index=False)
                .apply(lambda df_temp:
                    df_temp.assign(**{"freq": lambda x: (x["count"]/sum(x["count"]))})
                    )
            .merge(
            (value_count_df1
             .drop(columns=['count', 'rank'], errors='ignore')
             .rename(columns={'freq': '%'})),
             left_on=cate1, right_on=cate1))

    # Sort dataframe. By design, datetime-like column will always be sorted by time.
    if not _pd.api.types.is_datetime64_dtype(df_count[cate1].dtype):
        if sort_by == "value":
            df_count = _pd.concat([
                df_count.query(f"{cate1} != 'Others'").sort_values(by=["%", cate2], axis=0, ascending=[ascending, True]),
                df_count.query(f"{cate1} == 'Others'")
            ])
        elif sort_by == "index":
            df_count = _pd.concat([
                df_count.query(f"{cate1} != 'Others'").sort_values(by=[cate1, cate2], axis=0, ascending=[ascending, True]),
                df_count.query(f"{cate1} == 'Others'").query(f"{cate1} == 'Others'")
            ])
        else:
            raise ValueError(f"Unregconized sort_by value: {sort_by}")
    df_count = df_count.reset_index(drop=True)

    # Pivot table won't preserve sorting order, so we use groupby unstack instead
    table = (
        df_count
            .groupby([cate1, cate2, "%"], sort=False)
            .first()
            .unstack([cate2], fill_value=0)
            .reset_index().set_index(cate1)
    )
    
    table = _pd.concat([table_total, table])
    #order categories of column2 
    current_columns = table['count'].columns
    if not encode_target and category_orders and cate2 in category_orders:
        target_values = category_orders[cate2]
        # add values that in current but not in category_order
        tmp = []
        for col in current_columns:
            if col not in target_values:
                tmp.append(col)
        target_values += tmp
        # remove values that in category_order but not in current
        tmp = []
        for col in target_values:
            if col not in current_columns:
                tmp.append(col)
        target_values = [val for val in target_values if val not in tmp]

    table[[('count', x) for x in current_columns]] = table[[('count', x) for x in target_values]]
    table[[('freq', x) for x in current_columns]] = table[[('freq', x) for x in target_values]]
    table.columns = _pd.MultiIndex.from_tuples([('%', '')] + [('count', x) for x in target_values] + [('freq', x) for x in target_values])

    #order categories of col1
    if category_orders and cate1 in category_orders:
        orders1 = ['Total']+list(category_orders[cate1])
        orders1 = [cate for cate in orders1 if cate in table.index]
        table = table.loc[orders1]

    # Remove index names
    table.index.name = None
    table.columns.names = (None, None)

    if return_json:
        return table, df_count, value_count_df2, target_values, target_map
    
    if encode_target:
        return table, target_map
    
    return table


def describe_2d_category_with_plot(
        df, 
        cate1, cate2,
        threshold1=(9, None),
        threshold2=(9, None),
        category_rename=None,
        category_orders=None,
        dropna=False, 
        fillna1=None,
        fillna2=None,
        sort_by="value", 
        ascending=False,
        encode_target=False,
        orientation='v',
        figsize=(545, _FIG_HEIGHT),
        plot_count_kwargs={},
        plot_freq_kwargs={},
        color_map=None,
        gradient_background=True,
        precomputed_kwargs=None,
        return_table=False,
        return_raw=False,
        return_json=False,
        **kwargs):
    """
    Describe interaction between two categorical columns.
    Contributor: ThanhNM3

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame.

    cate1 : str
        Name of the first column to be analysed in the input dataframe.
    
    cate2 : str
        Name of the second column to be analysed in the input dataframe

    threshold1 : int or (int, int), default: (9, None)
        Threshold for `cate1`.
        - First value is rank threshold, threshold categories with the largest occurence frequecy are kept and the others are merged together.
        - Second value is count threshold, all categories with occurence frequecy smaller than the threshold will be merged together.
        If an integer is provided, it is assigned to rank threshold and count threshold is set to None.

    threshold2 : int or (int, int), default: (9, None)
        Threshold for `cate2`.

    category_rename : dict, default None
        Key is column name and value is a dict to map categories in the column into new names. If a category is not given in the value dict, its original name is used.
    
    category_orders : dict, default None
        Key is column name and value is a list of values in the desired order. This is used to override the default category ordering behaviour in which categories are sorted in descending order according to value counts.

    dropna : boolean, default: False
        If False, NA values will be treated as a category, dropped otherwise.

    fillna1 : scalar, default: None
        Fill NA/NaN values in `cate1` using the specified value. 
    
    fillna2 : scalar, default: None
        Fill NA/NaN values in `cate2` using the specified value. 

    sort_by : str in ("value", "index"), default None
        If None or "count" the value count dataframe is sorted based on value counts.
        If "index" the value count dataframe is sorted based on category names.

    ascending : boolean, default False
        If True, the value count dataframe is sorted in ascending order based on `sort_by`
        If False, the value count dataframe is sorted in descending order based on `sort_by`

    encode_target : boolean, default: False
        Whether to map target variable to increasing number, for cleaner visualization.

    orientation : str in ("h", "v"), default "h"
        Orientation of the bar plot. "h" mean horizontal and "v" mean vertical

    figsize : (float, float), default: None
        Specify the size of sub bar chart. Use heuristic to determine figsize if None.

    plot_count_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the count plot.
    
    plot_freq_kwargs : dict, default {}
        Keyword arguments which will be sent to `update_layout()` function of the frequency plot.

    color_map : dict, default {}
        Key is category name and value is color. String values should define valid CSS-colors used to override
        coloring behaviour to assign a specific colors to marks corresponding with specific categories.
    
    gradient_background: bool, default True
        Whether to apply gradient background to table.

    precomputed_kwargs : dict, default None
        value_count_df : Pandas Dataframe
            Precomputed dataframe containing categories name as indices and a count column whose name is provided in `count_col` param.
        count_col : str, default: None
            Name of the count column in the precomputed `value_count_df`.
        text_col : str, default: None
            Name of the text column in the precomputed `value_count_df`.

    return_table : boolean, default False
        Whether to return the formatted table only.

    return_fig : boolean, default False
        Whether to return the plot only.

    return_json : boolean, default False
        Whether to return the analysis in json format. This is specifically designed for internal use.

    Return
    ------
    HStack containing value count table, count plot and frequency plot."""

    table, df_count, value_count_df2, target_values, target_map = describe_2d_category(
        df=df,
        cate1=cate1, cate2=cate2,
        threshold1=threshold1,
        threshold2=threshold2,
        category_rename=category_rename,
        category_orders=category_orders,
        dropna=dropna,
        fillna1=fillna1,
        fillna2=fillna2,
        encode_target=encode_target,
        sort_by=sort_by,
        ascending=ascending,
        precomputed_kwargs=precomputed_kwargs,
        return_json=True,
        **kwargs
    )

    # Table style
    border_props = [
        ('border-left-color', 'black'),
        ('border-left-style', 'dotted'),
        ('border-left-width', 'thin'),
    ]

    row_border_props = [
        ('border-bottom-color', 'black'),
        ('border-bottom-style', 'dotted'),
        ('border-bottom-width', '1.1px'),
    ]

    header_border_fmt = [
        dict(selector=f"th.col_heading.level0.col1", props=border_props),
        dict(selector=f"th.col_heading.level0.col{len(target_values)+1}", props=border_props),

        dict(selector=f"th.col_heading.level1.col1", props=border_props),
        dict(selector=f"th.col_heading.level1.col{len(target_values)+1}", props=border_props),\

        dict(selector=f"th.row_heading.level0.row0", props=row_border_props)
    ]
    
    max_length_index = max([len(str(x)) for x in table.index])
    max_length_column = max([len(str(x)) for x in value_count_df2[cate2]])
    min_width = min(max_length_index*8.5, 150)
    max_count = table.loc[_pd.IndexSlice[[idx for idx in table.index if idx != 'Total'], 'count']].max().max()
    min_count = table.loc[_pd.IndexSlice[[idx for idx in table.index if idx != 'Total'], 'count']].min().min()
    max_freq = table.loc[_pd.IndexSlice[:, 'freq']].max().max()
    min_freq = table.loc[_pd.IndexSlice[:, 'freq']].min().min()

    table_style = (
        table
            .style
            .format("{:,.0f}", na_rep="-", subset="count")
            .format("{:.1%}", na_rep="-", subset=["%", "freq"])
            .set_properties(**dict(border_props), subset=[("count", str(target_values[0])), ("freq", str(target_values[0]))])
            .set_properties(**dict(row_border_props), subset= _pd.IndexSlice[['Total'], :])
            .set_table_styles([
            dict(selector='th', props=[('text-align', 'center')]),
            dict(selector='td', props=[('text-align', 'center')]),
            dict(selector='th.row_heading.level0', props=[('text-align', 'right'), ('min-width', f'{min_width}px')]),
            *header_border_fmt
        ])

    )

    if gradient_background:
        table_style = (
            table_style
            .background_gradient(cmap="Blues", subset=_pd.IndexSlice[[idx for idx in table.index if idx != 'Total'], 'count'], axis=1, vmax=max_count, vmin=min_count)
            .background_gradient(cmap="Blues", subset=_pd.IndexSlice[[idx for idx in table.index if idx != 'Total'], 'freq'], axis=1, vmax=max_freq, vmin=min_freq)
            .background_gradient(cmap="Reds", subset=_pd.IndexSlice[[idx for idx in table.index if idx != 'Total'], '%'])
            .background_gradient(cmap="Reds", subset=_pd.IndexSlice['Total', table.columns[table.columns.get_level_values(0) == 'count']], axis=1)
            .background_gradient(cmap="Reds", subset=_pd.IndexSlice['Total', table.columns[table.columns.get_level_values(0) == 'freq']], axis=1)
            .set_caption('<b>freq</b> is divided by horizontal sum')
            )

    def config_plotly_bar_chart(fig, figsize=None):
        """Modify layout of a Plotly Bar chart plot."""
        if figsize:
            width, height = figsize
            width = width + max(0, max_length_column - 11)
        else:
            # Get number of categories
            width = 520
            height = 400
            width = width + max(0, max_length_column - 11)

        return fig.update_traces(
            width=0.7
        ).update_layout(
            width=width,
            height=height,
            autosize=True,
            title=dict(
                x=0.5,  # modify title's position
                font_size=16
            ),
            xaxis=dict(
                type="category"
            )
        ).update_xaxes(
            ticks="",
            showline=True
        ).update_yaxes(
            showline=True,
            title_standoff=5
        )

    # color_discrete_sequence = [_MAIN_COLOR, "#ff9900", "#0cb577", "#ff474c", "#751973", '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    # "#fcb228"
    color_discrete_sequence = ["#1F77B4", "#00a2bc", "#ade9da", "#ff7c00", "#ffe65e", 
                               "#009855", "#7dc75f", "#d5d75c", "#5c737c"]
    color_discreate_sequence_two = ['#1F77B4', '#69b7fa']
    cate2_to_map = [cate for cate in value_count_df2[cate2] if cate not in ['nan', 'Others']]
    default_color_map = _describe_utils.gen_color_map_from_sequence(
        cate2_to_map,
        color_discreate_sequence_two if len(cate2_to_map) == 2 else color_discrete_sequence)
    default_color_map['Others'] = "#a2b4b4"
    default_color_map['nan'] = _NULL_COLOR
    if color_map is not None:
        default_color_map.update(color_map)

    category_orders_temp = dict()
    if category_orders is None:
        category_orders_temp = {
            cate2: list(value_count_df2.sort_values('freq', ascending=False)[cate2]),
            cate1: list(table.index)
        }
    else:
        if cate1 not in category_orders_temp:
            category_orders_temp[cate1] = list(table.index)
        if cate2 not in category_orders_temp:
            category_orders_temp[cate2] = list(value_count_df2.sort_values('freq', ascending=False)[cate2])
    if 'Others' in category_orders_temp[cate2]:
        category_orders_temp[cate2].remove('Others')
        category_orders_temp[cate2].append('Others')

    fig_count_layout = {
        'width': figsize[0],
        'height': figsize[1],
        'legend': dict(title=''),
        'title': f'Count of {cate1} by {cate2}'
    }
    fig_count_layout.update(plot_count_kwargs)
    if orientation == 'v':
        chart = config_plotly_bar_chart(
            _px.bar(
                df_count, x=cate1, y="count", color=cate2,
                color_discrete_map=default_color_map,
                category_orders=category_orders_temp
            ), 
        ).update_layout(
            **fig_count_layout
        )

    fig_freq_layout = {
        'width': figsize[0],
        'height': figsize[1],
        'legend': dict(title=''),
        'title': f'Percentage of {cate1} by {cate2}',
        'showlegend': True
    }
    fig_freq_layout.update(plot_freq_kwargs)

    freq_chart = config_plotly_bar_chart(
        _px.bar(
            df_count, x=cate1, y="freq", color=cate2,
            color_discrete_map=default_color_map,
            category_orders=category_orders_temp
        ),
    ).update_layout(
        **fig_freq_layout
    ).update_traces({
        'yhoverformat': '.2%'
    })

    if encode_target:
        left = _p.VStack([
            _render_html(target_map.style.hide_index()),
            _render_html(table_style)
        ])
    else:
        left = _render_html(table_style)

    if return_table:
        return left
    if return_raw:
        return [table_style, chart, freq_chart]

    if return_json:
        json_fig_count = _get_plotly_fig_json(chart)
        json_fig_freq = _get_plotly_fig_json(freq_chart)
        # for visual reason, column1 name is added to only html version, otherwise in jupyter version this looks ugly
        table.index.name = cate1
        json_table_count, json_table_freq, json_table_total = _describe_utils.format_table_cate_cate_json(table)
        return json_table_count, json_table_freq, json_table_total, json_fig_count, json_fig_freq

    return _VStack([
        _Block(left, styles={'overflow': 'scroll'}), 
        _CustomHStack([chart, freq_chart], grid_cell_style="margin-right:0px")
        ], styles={'text-align':'center'})


# Correlation
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = _ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return _np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def cramer_correlation_with_cate_target(df, target="label", cate_cols=[], unique_value_thres=100, display=True,
                                        min_count=None):
    """
        cramer correlation used for cate-cate variables
        this function calculate cramer correlation between target and other columns in dataframe
        df : pandas dataframe
        target : column name to calculate cramer correlation
        unique_value_thres : threshold to remove column that has too many unique values (cause slowing down the function) in case cate_cols is not set
        cate_cols : list of cate column name, if empty , then used all columns in df but remove column has num unique value >  unique_value_thres
        display : display df if True, else return df
        min_count : minimum count of a cate group. If it is set, remove all group have smaller count than min_count out of the data

        value = 1 : means we can completely know what target is given value of column feature
        value = 0 : means we know nothing about target value givin value of column feature
    """
    if len(cate_cols) == 0:
        cate_cols = df.columns
        cate_cols = [col_name for col_name in cate_cols if col_name != target]
        unique_count = df[cate_cols].nunique()
        cate_cols = unique_count[unique_count <= unique_value_thres].index.values.tolist()

    cate_cols = [col_name for col_name in cate_cols if col_name != target]

    correlation_list = []
    target_series = df[target].fillna("null")
    for col_name in _tqdm(cate_cols):
        tmp_df = df.copy(deep=True)
        if (min_count):
            group_count = df.groupby(col_name).count()[target].reset_index()
            group_count = group_count[group_count[target] >= min_count]
            tmp_df = _pd.merge(tmp_df, group_count[[col_name]], on=col_name, how="inner")
            target_series = tmp_df[target].fillna("null")
        column_series = tmp_df[col_name].fillna("null")
        #         print(len(tmp_df))
        confusion_matrix = _pd.crosstab(target_series, column_series)
        res = cramers_corrected_stat(confusion_matrix)
        correlation_list.append([col_name, res])

    correlation_df = _pd.DataFrame(correlation_list, columns=["column_name", "Cramer"])
    correlation_df = correlation_df.sort_values(["Cramer"], ascending=[False])

    if display == True:
        _display_df_with_title(correlation_df, "Cramer Correlation (with target = {})".format(target))
    else:
        return correlation_df


def correlation_ratio(categories, measurements):
    categories = categories.reset_index(drop=True)
    measurements = measurements.reset_index(drop=True)

    fcat, _ = _pd.factorize(categories)
    cat_num = _np.max(fcat) + 1
    y_avg_array = _np.zeros(cat_num)
    n_array = _np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[_np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = _np.average(cat_measures)
    y_total_avg = _np.sum(_np.multiply(y_avg_array, n_array)) / _np.sum(n_array)
    numerator = _np.sum(_np.multiply(n_array, _np.power(_np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = _np.sum(_np.power(_np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator / denominator
    return eta


def correlation_cont_with_cate_target(df, target="label", cont_cols=[], display=True):
    """
        correlation ratio used for cate-cont variables
        this function calculate correlation ratio between target and other columns in dataframe
        df : pandas dataframe
        target : column name to calculate correlation ratio
        cont_cols : list of cont column name, if empty , then used all columns with type int and float in df
        display : display df if True, else return df

        value = 1 : means cont feature seperate with label
        value = 0 : means the opposite
        value >= 0.001 : good enough to use in model (personal experience, not based on any researches)
    """

    num_rows = df.shape[0]

    if len(cont_cols) == 0:
        tmp = df.dtypes.to_frame()
        tmp = tmp.reset_index()
        tmp.columns = ["column_name", "dtypes"]
        tmp["dtypes"] = tmp["dtypes"].astype(str)
        tmp["flag"] = tmp["dtypes"].apply(lambda x: "int" in x or "float" in x)
        tmp = tmp[tmp["flag"] == True]
        cont_cols = tmp["column_name"].values.tolist()

    cont_cols = [col_name for col_name in cont_cols if col_name != target]

    correlation_list = []
    for col_name in _tqdm(cont_cols):
        #         print(col_name)
        df_tmp = df[[target, col_name]]
        df_tmp = df_tmp.dropna()
        df_tmp[col_name] = df_tmp[col_name].astype("float")
        res = correlation_ratio(df_tmp[target], df_tmp[col_name])
        percentage = str(round((100 * df_tmp.shape[0] / num_rows), 2)) + "%"
        correlation_list.append([col_name, res, df_tmp.shape[0], percentage])

    correlation_df = _pd.DataFrame(correlation_list,
                                   columns=["column_name", "correlation_ratio", "#affected_rows", "%affected_rows"])
    correlation_df = correlation_df.sort_values(["correlation_ratio"], ascending=[False])

    if display == True:
        _display_df_with_title(correlation_df, "Correlation Ratio (with target = {})".format(target))
    else:
        return correlation_df


def corr_check_with_label(df, label="label", ignore_na=False):
    corr = {}
    for c in df.columns:
        if c == label:
            continue
        if ignore_na:
            tmp = df[[c, label]].corr()
            if tmp.shape == (2, 2):
                corr[c] = tmp.iloc[0, -1]
            else:
                continue
        else:
            try:
                corr[c] = _np.corrcoef(df[c].values, df[label].values)[0][-1]
            except:
                pass

    print("num cols calculated corr = ", len(corr))
    print("min corr:", _np.min(x for x in corr.values() if ~_np.isnan(x)))
    print("max corr:", _np.max(x for x in corr.values() if ~_np.isnan(x)))

    corr = dict(sorted(corr.items(), key=lambda x: _np.abs(x[1]), reverse=True))
    return corr


def describe_compare_numeric_data_with_chart_hist(series_1, series_2, round=2):
    """
        Describe data and plot histogram of numeric data of 2 serie
        Arguments:
            series_1: series 1 need describe
            series_2: series 2 need describe
            round: number of round data
    """
    print(_color.BOLD + "Describe compare data of " + series_1.name + ' and ' + series_2.name + ":" + _color.END)
    table1 = describe_numeric_data(series_1, round).reset_index()
    table2 = describe_numeric_data(series_2, round).reset_index()
    table1[series_2.name] = table2[series_2.name]
    _display(table1)
    print(_color.BOLD + "Histogram chart compare of " + series_1.name + ' and ' + series_2.name + ":" + _color.END)
    _matplotlib_plot_2d_numeric_hist_compare(series_1, series_2)


def describe_psi_between_two_series(expected, actual, buckettype='quantiles', buckets=10):
    expected = expected.dropna().copy()
    actual = actual.dropna().copy()
    if expected.name == actual.name:
        expected.name = expected.name + "_x"
        actual.name = actual.name + "_y"
    _display("PSI Monthly between " + expected.name + " and " + actual.name)
    raw_breakpoints = _np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = _np.stack([_np.percentile(expected, b) for b in raw_breakpoints])
    if buckettype == 'bins':
        breakpoints = _describe_utils._psi_scale_range(raw_breakpoints, _np.min(expected), _np.max(expected))
    dev_counts = _np.histogram(expected, breakpoints)[0]
    valid_counts = _np.histogram(actual, breakpoints)[0]
    df_psi = _pd.DataFrame({'Bucket': _np.arange(1, buckets + 1), 'Breakpoint Value': breakpoints[1:],
                            expected.name + ' Count': dev_counts, actual.name + ' Count': valid_counts})
    df_psi[expected.name + ' Percent'] = df_psi[expected.name + ' Count'] / len(expected)
    df_psi[actual.name + ' Percent'] = df_psi[actual.name + ' Count'] / len(actual)
    df_psi[expected.name + ' Percent'].loc[df_psi[expected.name + ' Percent'] == 0] = 0.0001
    df_psi[actual.name + ' Percent'].loc[df_psi[actual.name + ' Percent'] == 0] = 0.0001
    df_psi = df_psi[
        ['Bucket', 'Breakpoint Value', expected.name + ' Count', expected.name + ' Percent', actual.name + ' Count',
         actual.name + ' Percent']]
    df_percents = df_psi[[expected.name + ' Percent', actual.name + ' Percent', 'Bucket']] \
        .melt(id_vars=['Bucket']) \
        .rename(columns={'variable': 'Population', 'value': 'Percent'})
    fig = _px.bar(df_percents, x='Bucket', y='Percent', color='Population', barmode="group")
    fig.show()
    df_psi['PSI'] = (df_psi[actual.name + ' Percent'] - df_psi[expected.name + ' Percent']) * _np.log(
        df_psi[actual.name + ' Percent'] / df_psi[expected.name + ' Percent'])
    df_psi.loc['Total'] = df_psi.sum()
    df_psi['Bucket'] = df_psi['Bucket'].astype(str)
    df_psi.loc['Total', 'Bucket'] = ''
    df_psi.loc['Total', 'Breakpoint Value'] = ''

    df_psi[expected.name + ' Percent'] = df_psi[expected.name + ' Percent'] * 100
    df_psi[actual.name + ' Percent'] = df_psi[actual.name + ' Percent'] * 100
    df_psi_table = df_psi.style.format(
        {expected.name + ' Count': '{:.0f}', expected.name + ' Percent': '{:.2f} %', actual.name + ' Count': '{:.0f}',
         actual.name + ' Percent': '{:.2f}%'}).apply(_style_specific_cell_psi, axis=None, n_bucket=buckets)
    _display(df_psi_table)

    psi_final = _np.sum(df_psi['PSI'].iloc[:-1])
    _display_markdown("=> $PSI = " + str('{0:.10f}'.format(psi_final)) + "$")


def _style_specific_cell_psi(x, n_bucket):
    color_green = 'background-color: green'
    color_yellow = 'background-color: yellow'
    color_red = 'background-color: red'
    df1 = _pd.DataFrame('', index=x.index, columns=x.columns)

    # Format PSI
    if x.iloc[n_bucket, 6] > 0.25:
        df1.iloc[n_bucket, 6] = color_red
    elif x.iloc[n_bucket, 6] > 0.1:
        df1.iloc[n_bucket, 6] = color_yellow
    else:
        df1.iloc[n_bucket, 6] = color_green
    return df1

# app.layout = html.Div([
#     html.Div(style={'display': 'flex'}, children=[
#         #minor
#         html.Div([
#             html.H2(f"Top 15 Major Occupations", id="title-major", style={'position':'relative', 'marginTop': 20 ,'marginBottom': 40}),
#             html.Div(id='table-major-occupation', children=table_major, style={"position": "relative"}),
#         ], style={'width': left_width, 'marginRight': 30}),
#         html.Div([
#             html.H2(f"Distribution of Major Occupations", style={'position':'relative', 'marginTop': 20}),
#             dcc.Graph(
#                 id='fig-major-occupation',
#                 figure=fig_major,
#                 config={
#                     'displaylogo':False}
#             )
#         ], style={'width': right_width, 'padding': '0 0'})
#     ]),
#     html.Div(style={'display': 'flex'}, children=[
#         #minor
#         html.Div([
#             html.H2(f"Top 10 Minor Occupations", id="title-minor", style={'position':'relative', 'marginTop': 20 ,'marginBottom': 40}),
#             html.Div(id='table-minor-occupation', children=table_minor, style={"position": "relative"}),
#         ], style={'width': left_width, 'marginRight': 30}),
#         html.Div([
#             html.H2(f"Distribution of Minor Occupations", style={'position':'relative', 'marginTop': 20}),
#             dcc.Graph(
#                 id='fig-minor-occupation',
#                 figure=fig_minor,
#                 config={
#                     'displaylogo':False}
#             )
#         ], style={'width': right_width, 'padding': '0 0'})
#     ]),
#     html.Div(style={'display': 'flex'}, children=[
#         #broad
#         html.Div([
#             html.H2(f"Top 10 Broad Occupations", id="title-broad", style={'position':'relative', 'marginTop': 20 ,'marginBottom': 40}),
#             html.Div(id='table-broad-occupation', children=table_broad, style={"position": "relative"}),
#         ], style={'width': left_width, 'marginRight': 30}),
#         html.Div([
#             html.H2(f"Distribution of Broad Occupations", style={'position':'relative', 'marginTop': 20}),
#             dcc.Graph(
#                 id='fig-broad-occupation',
#                 figure=fig_broad,
#                 config={
#                     'displaylogo':False}
#             )
#         ], style={'width': right_width, 'padding': '0 0'})
#     ])
# ], id='container')