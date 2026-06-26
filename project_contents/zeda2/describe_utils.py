import math as _math
import pandas as _pd
import numpy as _np
import math as _math
import calendar as _calendar
from datetime import datetime as _datetime
import plotly.graph_objects as go
from plotly import optional_imports
import folium
from si_prefix import si_format
import re
import math
scipy_stats = optional_imports.get_module("scipy.stats")
# pybloqs
from pybloqs.html import append_to as _append_to
from pybloqs.block.layout import CompositeBlockMixin as _CompositeBlockMixin
from pybloqs.block.layout import BaseBlock as _BaseBlock
from pybloqs.block.layout import Cfg as _Cfg
from pybloqs import Block
import pybloqs.block.table_formatters as tf
# dash
from pandas.api.types import is_numeric_dtype

from zeda2 import table_utils as tu
import plotly
import inspect
import plotly.express as px
from functools import reduce
html_replace_dict = [(' ', '&nbsp;'), ('-', '&#8209;')]

# from plotly.offline import init_notebook_mode as _init_notebook_mode

# # Fix bug not show chart
# _init_notebook_mode(connected=True)

_TABLE_WIDTH = "320px"
_DEFAULT_NBINS_HISTOGRAM=50
_MAIN_COLOR = '#1F77B4'
_LIST_VALID_LAYOUT_PROPS = plotly.graph_objs.Layout.__dict__.get('_valid_props')
_FREQ_COL_NAME='frequency (%)'
_COUNT_COL_NAME='count'
_CATE_DEFAULT_FORMATTER = {_FREQ_COL_NAME: 'html_percent', _COUNT_COL_NAME: ',.0f'}
_TABLE_PADDING_STYLES = [
            dict(selector="td", props=[("padding", "5px")]),
            dict(selector="th", props=[("padding", "5px")]),
        ]


def merge_nan_rows(df, col, merge_metric='sum'):
    df_nan = df.loc[df[col].isna() | df[col].isin([None, 'nan'])]
    df_new = df.loc[~df[col].isna() & ~df[col].isin([None, 'nan'])]
    df_merge = _pd.DataFrame([[None]*len(df.columns)], columns=df.columns)
    df_merge.loc[0, col] = 'nan'
    for tmp_col in df.columns.drop(col):
        if is_numeric_dtype(df[tmp_col]):
            if merge_metric == 'mean':
                df_merge.loc[0, tmp_col] = df_nan[tmp_col].mean()
            else:
                df_merge.loc[0, tmp_col] = df_nan[tmp_col].sum()
        else:
            df_merge.loc[0, tmp_col] = '-'
    return _pd.concat([df_new, df_merge])


def get_valid_argument_names(func):
    return list(inspect.signature(func).parameters.keys())


def split_plotly_layout_trace_props(plot_kwargs):
    general_props = dict()
    layout_props = dict()
    trace_props = dict()
    _LIST_VALID_BAR_PROPS = get_valid_argument_names(px.bar)
    for key in plot_kwargs.keys():
        if key in _LIST_VALID_BAR_PROPS:
            general_props[key] = plot_kwargs[key]
        elif key.split('_')[0] in _LIST_VALID_LAYOUT_PROPS:
            layout_props[key] = plot_kwargs[key]
        else:
            trace_props[key] = plot_kwargs[key]
    return general_props, layout_props, trace_props


def get_plotly_fig_json(fig):
        return plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)


def fill_index_df(df, indices, fill_value=None, sort_index=False):
    df_tmp = df.copy()
    for idx in indices:
        if idx not in df.index:
            df_tmp.loc[idx] = fill_value
    if sort_index:
        df_tmp = df_tmp.sort_index()
    return df_tmp


def format_series(series, fmt):
    return series.apply(lambda x: f'{x:{fmt}}')


def get_bin_centers_from_edges(bin_edges):
    bin_centers= []
    for idx in range(len(bin_edges)-2):
        bin_centers.append((bin_edges[idx] + bin_edges[idx+1])/2)
    return bin_centers


def util_category_rename(df, category_rename, cols=None):
    '''
    Changes apply to the input df
    '''
    cols = cols or df.columns
    for col in cols:
        if category_rename and col in category_rename:
            df[col] = df[col].apply(lambda x: category_rename[col].get(x, x))
    return df


def spark_to_python_datetime_format(datetime_format):
    '''
    Transform spark datetime format into compatible python datetime format. 
    Currently, this function does not fully transform the format.
    Supported components: year, month, day of month, hour, minute, second.
    '''
    dict_spark2python_dateformat = {
        'yyyy': '%Y', 'yy': '%y', 'y': '%Y',
        # December
        'MMMM': '%B', 'LLLL': '%B', 
        # Dec
        'MMM': '%b', 'LLL': '%b',
        # 12, special case for M as M can be mistaken in AM PM
        'MM': '%m', 'LL': '%m', '([^A^P])(M)': r'\1%m', 'L': '%m',
        'dd': '%d', 'd': '%d',
        'HH': '%H', 'H': '%H', 'KK': '%I', 'K': '%I', 'K': '%I', 'mm': '%M', 'm': '%M',
        'ss': '%S', 's': '%S'
    }
    python_date_format = datetime_format
    for spark_pattern in re.findall('\w+', datetime_format):
        python_date_format = re.sub(spark_pattern, 
                             dict_spark2python_dateformat.get(spark_pattern, spark_pattern), 
                             python_date_format)
    return python_date_format


def plot_heatmap(df, text=None, fmt='si', plot_kwargs=None, **kwargs):
    plot_kwargs = plot_kwargs or {}
    kwargs = kwargs or {}
    if text is None and fmt=='si':
        text = df.fillna(0).applymap(lambda x: si_format(x, precision=0, format_str='{value}{prefix}'))
        text[text=='0'] = '-'
    elif text is None and fmt:
        text = df.fillna(0).applymap(lambda x: f'{{:{fmt}}}'.format(x))
    fig = go.Figure(
        go.Heatmap(
            x=df.columns,
            y=df.index,
            z=df.fillna(0),
            text=text,
            texttemplate='%{text}',
            colorscale='Teal',
            **kwargs
            )
        ).update_layout(plot_kwargs)
    return fig


class ZedaBaseStack(object):
    objs = []
    layout = ''
    default_grid_cell_styles = {'margin-right': '50px', 'margin-bottom': '30px'}
    default_grid_row_styles = {'display': 'flex', 'align-items': 'center'}
    default_grid_column_styles = {'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center'}
    
    
    def __init__(self, objs, grid_cell_style=None, grid_row_style=None, grid_column_style=None):
        if type(grid_cell_style) == dict:
            self.default_grid_cell_styles.update(grid_cell_style)
        if type(grid_row_style) == dict:
            self.default_grid_row_styles.update(grid_row_style)
        if type(grid_column_style) == dict:
            self.default_grid_column_styles.update(grid_column_style)
        grid_cell_style_str = ';'.join([f'{key}:{value}' for key, value in self.default_grid_cell_styles.items()])
        grid_row_style_str = ';'.join([f'{key}:{value}' for key, value in self.default_grid_row_styles.items()])
        grid_column_style_str = ';'.join([f'{key}:{value}' for key, value in self.default_grid_column_styles.items()])
        self.objs = objs
        self.layout = self._generate_html_string(self.objs, 
                                                 grid_cell_style_str=grid_cell_style_str,
                                                 grid_row_style_str=grid_row_style_str,
                                                 grid_column_style_str=grid_column_style_str
                                                )
    
    def show(self):
        from IPython.display import display_html
        
        display_html(self.layout, raw=True)
    
    
    def get_html(self):
        return self.layout
    
    
    def _generate_html_string(self, objs=None, 
                              grid_cell_style_str=None, 
                              grid_row_style_str=None, 
                              grid_column_style_str=None):
        pass
    
    
    def _get_plotly_html_string(self, fig):
        from plotly.io import renderers
        from plotly.io._utils import validate_coerce_fig_to_dict
    
        fig_dict = validate_coerce_fig_to_dict(fig, validate=True)
        fig_html = renderers._build_mime_bundle(fig_dict)['text/html']
        return fig_html
    
    
    def _get_obj_html_string(self, obj):
        from plotly.graph_objects import Figure
        from pandas import DataFrame
        from pandas.io.formats.style import Styler
        
        if type(obj) == str:
            obj_html = obj
        elif isinstance(obj, ZedaBaseStack):
            obj_html = obj.get_html()
        elif isinstance(obj, Figure):
            obj_html = self._get_plotly_html_string(obj)
        elif isinstance(obj, DataFrame) or isinstance(obj, Styler):
            obj_html = obj.to_html()
        else:
            obj_html = ''
            print(f'''Type {type(obj)} is not supported. 
                    Only object of the following classes is supported: str, plotly.graph_objects.Figure, pandas.DataFrame, pandas.io.formats.style.Styler, ZedaHStack, ZedaVStack''')
        return obj_html


class ZedaHStack(ZedaBaseStack):
    
    def __init__(self, objs, grid_cell_style=None, grid_row_style=None):
        super(ZedaHStack, self).__init__(objs, grid_cell_style, grid_row_style, None)
        return
    
    
    def _generate_html_string(self, objs=None, 
                              grid_cell_style_str=None, 
                              grid_row_style_str=None,
                              grid_column_style_str=None):
        objs = objs or []
        grid_cell_style_str = grid_cell_style_str or ''
        grid_row_style_str = grid_row_style_str or ''
        
        serialized_html = ''
        for obj in objs:
            obj_html = self._get_obj_html_string(obj)
            serialized_html += f'<div class="zeda_cell" style="{grid_cell_style_str}">{obj_html}</div>'
        layout = f'''
        <div class="zeda_container">
            <div class="zeda_row" style="{grid_row_style_str}">
            {serialized_html}
            </div>
        </div>'''
        return layout
    
    
class ZedaVStack(ZedaBaseStack):
    
    def __init__(self, objs, grid_cell_style=None, grid_column_style=None):
        super(ZedaVStack, self).__init__(objs, grid_cell_style, None, grid_column_style)
        return
    
    
    def _generate_html_string(self, objs=None, 
                              grid_cell_style_str=None, 
                              grid_row_style_str=None,
                              grid_column_style_str=None):
        objs = objs or []
        grid_cell_style_str = grid_cell_style_str or ''
        grid_column_style_str = grid_column_style_str or ''
        
        serialized_html = ''
        for obj in objs:
            obj_html = self._get_obj_html_string(obj)
            serialized_html += f'<div class="zeda_cell" style="{grid_cell_style_str}">{obj_html}</div>'
            
        layout = f'''
        <div class="zeda_container">
            <div class="zeda_column" style="{grid_column_style_str}">
            {serialized_html}
            </div>
        </div>'''
        return layout


class CustomGrid(_CompositeBlockMixin, _BaseBlock):
    def __init__(self, contents, cols=1, cascade_cfg=True, grid_row_style=None, grid_cell_style=None, grid_style=None, **kwargs):
        super().__init__(**kwargs)
        self._contents = self._blockify_contents(contents, kwargs, self._settings.title_level)
        self._cols = cols
        self._cascade_cfg = cascade_cfg
        self._grid_style = grid_style
        self._grid_row_style = grid_row_style
        self._grid_cell_style = grid_cell_style

    def _write_contents(self, container, actual_cfg, *args, **kwargs):
        content_count = len(self._contents)

        # Skip layout if there is no content
        if content_count > 0:
            row_count = int(_math.ceil(content_count / float(self._cols)))

            for row_i in range(row_count):
                row_el = _append_to(container, "div")
                row_el["class"] = ["pybloqs-grid-row"]
                if self._grid_row_style:
                    row_el["style"] = self._grid_row_style

                written_row_item_count = row_i * self._cols
                for col_i in range(self._cols):
                    item_count = written_row_item_count + col_i
                    if item_count >= content_count:
                        break

                    cell_el = _append_to(row_el, "div")
                    cell_el = _append_to(row_el, "div", style=self._grid_cell_style)
                    cell_el["class"] = ["pybloqs-grid-cell"]
                    self._contents[item_count]._write_block(cell_el,
                                                            actual_cfg if self._cascade_cfg else _Cfg(),
                                                            *args, **kwargs)

            _append_to(container, "div", style="clear:both")


def series_cast_numeric(s):
    if not is_numeric_dtype(s):
        try:
            return s.astype('double')
        except:
            raise Exception(f'Cannot cast numeric-defined column {s.name} of type {s.dtype} into numeric dtype.')
    else:
        return s


# Add default grid_row_style and grid_cell_style
class CustomHStack(CustomGrid):
    def __init__(
        self, contents, cascade_cfg=True,
        grid_row_style="display:flex;align-items:center;",
        grid_cell_style="margin-right:50px;",
        **kwargs
    ):
        super().__init__(
            contents,
            cascade_cfg=cascade_cfg,
            grid_row_style=grid_row_style,
            grid_cell_style=grid_cell_style,
            **kwargs
        )
        self._cols = len(self._contents)


def format_table_cate_full(
        df, null_value='nan',
        fmt={},
        table_width=_TABLE_WIDTH,
        table_index_name='',
        append_total_row=True
    ):
    fmt = fmt or _CATE_DEFAULT_FORMATTER
    fmt_total = tu.FmtAppendTotalRow(row_name='Total')
    formatters = []
    if fmt:
        for key, value in fmt.items():
            if key not in df.columns:
                break
            if value == 'html_percent':
                temp_fmt = tu.FmtPercentHtml(n_decimals=2, columns=[key], red_null=True, null_value=null_value)
                formatters.append(temp_fmt)
            else:
                temp_fmt = tu.FmtFormat(fmt=value, columns=[key])
                formatters = [temp_fmt]+formatters
    if null_value in df.index:
        fmt_red = tu.FmtRed(rows=[null_value], apply_to_index=True)
        formatters.append(fmt_red)
    if append_total_row:        
        formatters = [fmt_total] + formatters
    else:
        formatters = formatters
    df_temp = tu.EdaTable(df, formatters).get_dataframe()
    df_temp.index.name=table_index_name
    #format output table
    fmt_fontsize = tf.FmtFontsize(12, 'px')
    fmt_total_format = tu.FmtFormatTotalRow(rows=['<div class="total_row"><b>Total</b></div>'])
    return Block(df_temp, formatters=[fmt_fontsize, fmt_total_format], width=table_width)


def format_table_cate_html(df, col_name='stat', red_null=True, null_value='nan', main_color=_MAIN_COLOR, fmt={}):
    formatters = []
    fmt = fmt or _CATE_DEFAULT_FORMATTER
    if fmt:
        for key, value in fmt.items():
            if key not in df.columns:
                break
            if value == 'html_percent':
                temp_fmt = tu.FmtPercentHtml(
                    n_decimals=2, 
                    columns=[key],
                    null_value=null_value, 
                    red_null=red_null,
                    main_color=main_color)
                formatters.append(temp_fmt)
            else:
                temp_fmt = tu.FmtFormat(fmt=value, columns=[key])
                formatters = [temp_fmt]+formatters
    fmt_total_row = tu.FmtAppendTotalsRowCustom(fmt=fmt)
    if red_null and null_value in df.index:
        fmt_red = tu.FmtRed(rows=[null_value], apply_to_index=True)
        formatters += [fmt_red, fmt_total_row]
    else:
        formatters += [fmt_total_row]
    df_formatted = tu.EdaTable(df, formatters).get_dataframe()
    df_formatted = df_formatted.rename_axis(col_name).reset_index()
    df_formatted[col_name] = df_formatted[col_name].apply(lambda x: 
                                                reduce(lambda s, rep_tup: s.replace(rep_tup[0], rep_tup[1]), [x]+html_replace_dict))
    return df_formatted


def format_table_num_full(df):
    # simple table
    if len(df.columns) == 1:
        df_style = df.rename_axis(None).style.set_table_styles(_TABLE_PADDING_STYLES)
        return Block(render_html(df_style), width=_TABLE_WIDTH, styles={'text-align':'center'})
    else:
        fmt_line = tu.FmtVerticalLine(columns=['percentile'])
        fmt_highlight = tf.FmtHighlightText(bold=True, italic=False, columns="percentile", font_color="#000000")
        fmt_fontsize = tf.FmtFontsize(12, 'px')
        formatters = [fmt_highlight, fmt_fontsize, fmt_line]
        table = Block(df, formatters=formatters, width=_TABLE_WIDTH, styles={'text-align':'center'})
    return table
    

def format_table_num_html(df):
    df_temp = df.copy()
    df_temp = df_temp.rename_axis('stat').reset_index()
    df_temp.columns = ['stat', 'value ', 'percentile', 'value  ']
    df_temp['stat'] = df_temp['stat'].apply(lambda x: f'<b>{x}</b>')
    df_temp['percentile'] = df_temp['percentile'].apply(lambda x: f'<b>{x}</b>')
    return df_temp


def format_table_cc_html(df):
    df_temp = df.copy().rename_axis('statistic').reset_index()
    df_temp['statistic'] = df_temp['statistic'].apply(lambda x: f'<b>{x}</b>')
    return df_temp


def format_table_count_cate_html(df):
    df_temp = df.copy()
    integer_stat = [idx for idx in df_temp.index if idx not in ['mean', 'std']]
    df_temp.loc[['mean', 'std'], 'value'] = df_temp.loc[['mean', 'std'], 'value'].apply(lambda x: f'{x:,.2f}')
    df_temp.loc[integer_stat, 'value'] = df_temp.loc[integer_stat, 'value'].apply(lambda x: f'{x:,.0f}')
    df_temp = df_temp.rename_axis('count stat').reset_index()
    df_temp['count stat'] = df_temp['count stat'].apply(lambda x: f'<b>{x}</b>')
    return df_temp


def format_table_cate_cate_json(df):
    new_columns = [col[1] for col in df[['count']].columns]
    # rename df_count's columns
    df_count = df[['count']]
    df_count.columns = new_columns
    df_count = df_count.applymap(lambda x: f"{x:,.0f}")
    json_count = df_to_json(df_count)
    # rename df_freq's columns
    df_freq = df[['freq']]
    df_freq.columns = new_columns
    df_freq = df_freq.applymap(lambda x: f"{x:.1%}")
    json_freq = df_to_json(df_freq)
    # rename df_total's columns
    df_total = df[['%']]
    df_total.columns = ['%']
    df_total = df_total.applymap(lambda x: f"{x:.1%}")
    # keep index for total as index column will be included in this table
    json_total = df_to_json(df_total, keep_index=True)
    return json_count, json_freq, json_total


def df_to_json(df, keep_index=False):
    rows = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        row_dict = dict()
        if keep_index:
            index_name = df.index.name
            row_dict[index_name] = f'<b>{row.name}</b>'
        for column in df.columns:
            row_dict[column] = row[column]
        rows.append(row_dict)
    return rows


def gen_color_map(list_name, base_color, specific_map=None):
    color_map = dict()
    for name in list_name:
        color_map[name] = base_color
    if specific_map is not None:
        color_map.update(specific_map)
    return color_map


def gen_color_map_from_sequence(list_name, color_sequence):
    color_map = dict()
    l = len(list_name)
    for i in range(l):
        name = list_name[i]
        color = color_sequence[i % len(color_sequence)]
        color_map[name] = color
    return color_map


#histogram calculate bins
def increment_numeric(x, delta):
    if not delta:
        return x
    scale = 1/abs(delta)
    if scale > 1:
        newx=(scale*x+scale*delta)/scale
    else:
        newx=x+delta
    len_newx=len(str(newx))
    if len_newx>16: #likely a rounding error    
        len_x=len(str(x))
        len_delta=len(str(delta))
        if (len_newx >= len_x+len_delta):
            newx = float("{:.12f}".format(newx))
    return newx

histfunc_map = {
    'count': _np.ma.count,
    'sum': _np.sum,
    'min': _np.min,
    'max': _np.max,
    'avg': _np.mean
}

def calculate_bin_edges(min_val, max_val, nbins=100):
    if min_val is None or max_val is None or min_val!=min_val or max_val!=max_val:
        return []
    # calculate binsize
    value_range = max_val - min_val
    raw_binsize = value_range / nbins
    binsize = round_bin_size(raw_binsize)
    # calculate bin edges
    bin_edges = []
    start_value = math.floor(min_val/binsize)*binsize
    bin_edges.append(start_value)
    while bin_edges[-1] < max_val:
        x0 = bin_edges[-1]
        x1 = increment_numeric(x0, binsize)
        bin_edges.append(x1)
    return bin_edges


def aggregate_value_by_bin(series, binsize, histfunc='count'):
    if binsize == 0:
        return [0, 1], [0], [0.5], 0
    min_value = series.min()
    max_value = series.max()
    bin_edges = []
    bin_values = []
    bin_centers = []
    if isinstance(histfunc, str):
        if histfunc in histfunc_map:
            histfunc = histfunc_map[histfunc]
        else:
            raise Exception(f'Invalid value for histfunc. Supported histfunc values are {",".join(histfunc_map.keys())}')
    # zero is a tick no matter if zero is in the range visilbe o
    start_value = _math.floor(min_value/binsize)*binsize
    bin_edges.append(start_value)
    while bin_edges[-1] < max_value:
        x0 = bin_edges[-1]
        x1 = increment_numeric(x0, binsize)
        center = increment_numeric(x0, binsize/2)
        value = histfunc(series[(series>=x0) & (series<x1)])
        bin_values.append(value)
        bin_edges.append(x1)
        bin_centers.append(center)
    return bin_edges, bin_values, bin_centers, binsize


def round_bin_size(raw_binsize):
    if raw_binsize == 0:
        return 0
    multiplier = 1
    b = raw_binsize
    if b <= 9:
        while b < 1:
            b *= 10
            multiplier/=10
    elif b > 9:
        while b > 9:
            b /= 10
            multiplier*=10
    breakpoints = [1, 2, 5, 10]
    rounded_idx = 0
    rounded_up = breakpoints[rounded_idx]
    while rounded_up < b:
        rounded_idx += 1
        rounded_up = breakpoints[rounded_idx]    
    rounded_up=rounded_up*multiplier
    return rounded_up


def calculate_bin(series, nbins=None, binsize=None, non_empty_bins=True):
    '''
    non_empty_bins: bool, default True
        Whether to limit nbins to the number of unique values in order for every bin to contain at least 1 value.
    '''
    value_range = series.max() - series.min()
    nbins = nbins or _DEFAULT_NBINS_HISTOGRAM

    if binsize and binsize <= 0:
        nbins = 0
    elif binsize:
        nbins = value_range / binsize
    if non_empty_bins:
        if nbins > series.nunique():
            nbins = series.nunique()
    if nbins < 1 or _pd.isna(nbins):
        raw_binsize = 0
    else:
        raw_binsize = value_range / nbins
    binsize = round_bin_size(raw_binsize)
    return aggregate_value_by_bin(series, binsize)


# precompute box plot
def get_box_trace(series, series_percentile=None, color=_MAIN_COLOR, orientation='v', fmt=None):
    if series is not None:
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        min_val=series.min()
        mean=series.mean()
        max_val=series.max()
    elif series_percentile is not None:
        min_val, q1, median, q3, max_val, mean = series_percentile.loc[['min', '25%', '50%', '75%', 'max', 'mean']]
    else:
        raise Exception(f'Not enough data for a box plot.')
    if orientation == 'v':
        box_plot = go.Box(
            y=[''],
            q1=[q1], 
            median=[median], 
            q3=[q3], 
            lowerfence=[min_val], 
            upperfence=[max_val],
            mean=[mean],
            marker=dict(color=color),
            boxpoints=False,
            xhoverformat=fmt,
            yhoverformat=fmt
            )
    elif orientation=='h':
        box_plot = go.Box(
            x=[''],
            q1=[q1], 
            median=[median], 
            q3=[q3], 
            lowerfence=[min_val], 
            upperfence=[max_val],
            mean=[mean],
            marker=dict(color=color),
            boxpoints=False,
            yhoverformat=fmt,
            xhoverformat=fmt
            )
    return box_plot

    
def get_html_percent_content(percent):
    percent = round(percent*100, 2)
    if percent > 44:
        return f"<div style='width:{percent}%;color:#FFFFFF;background:#1F77B4;border-radius:3px;text-align:center;'>{percent}%</div>"
    else:
        return f"""<div style='display:flex;flex-direction:row'>
                                            <div style='width:{percent}%;height:100%;color:#1F77B4;background:#1F77B4;border-radius:3px;text-align:center;'><pre> </pre></div>
                                            <div style='margin-left:3px'>{percent}%</div>
                                          </div>"""


def make_kde(hist_data, name='kernel density', n_points=100, weights=None, curve_x=None):
    if len(hist_data) == 0:
        return go.Scatter()
    
    start = min(hist_data) * 1.0
    end = max(hist_data) * 1.0
    if curve_x is None:
        curve_x = [
            start + x * (end - start) / n_points
            for x in range(n_points)
        ]
    curve_y = scipy_stats.gaussian_kde(hist_data, weights=weights)(curve_x)
    curve = go.Scatter(
                x=curve_x,
                y=curve_y,
                mode="lines",
                name=name
            )
    return curve


def plot_choropleth(df_data, df_geo, columns, key_on, legend_name, fields, aliases, color="YlGn", **kwargs):
    m = folium.Map(location=[16, 106.660172], zoom_start=5.4, tiles="cartodbpositron", **kwargs)
    folium.Choropleth(
        geo_data=df_geo,
        name="choropleth",
        data=df_data,
        columns=columns,
        key_on=f"feature.properties.{key_on}",
        fill_color=color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
    ).add_to(m)
    
    style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}
    NIL = folium.features.GeoJson(
        df_geo,
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=folium.features.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    )
    m.add_child(NIL)
    m.keep_in_front(NIL)
    folium.LayerControl().add_to(m)
    return m


def cut_off_dataframe(
        df, 
        list_keep, threshold=(9, None), 
        by='count', null_value='nan', merge_metric='sum', 
        ascending=False):
    sort_by = by
    by = 'count' if by == 'index' else by
    # temporaily remove null from the df
    df_null = None
    df_merge = None
    if null_value in df.index:
        df_null = df.loc[null_value, :]
        df = df.drop(index=[null_value])
    # preprocess thresholde
    threshold = threshold or (9, None)
    if type(threshold) == int:
        threshold = (threshold, None)
    threshold = list(threshold)
    if threshold[0] is not None and threshold[0] > df.shape[0]:
            threshold[0] = df.shape[0]
    # start cut and keep
    if list_keep is not None:
        if len(list_keep) > 0 and type(list_keep[0]) != str:
            keep_type = float
        else:
            keep_type = str
        if null_value in list_keep:
            list_keep.remove(null_value)
        # !WARNING get rid on invalid cates without raising warning
        try:
            feat_keep = [feat for feat in df.index if keep_type(feat) in list_keep]
        except:
            feat_keep = []
        feat_except = list(set(df.index).difference(set(feat_keep)))
        df_merge = df.loc[feat_except, :]
        df = df.loc[feat_keep, :]
    elif threshold[0]:
        df_merge = df.iloc[threshold[0]:, :]
        df = df[:threshold[0]]
    # if neither threshols nor list_keep is provided, return the df
    elif threshold[1] is None:
        if df_null is not None:
            df.loc[null_value] = df_null
        return df
    # frequency threshold is prioritized
    if threshold[1] is not None:
        if df_merge is None:
            df_merge = df.loc[[]]
        df_merge = _pd.concat([df_merge, df[df[by] < threshold[1]]])
        df = df[df[by] >= threshold[1]]
    if sort_by=='index':
        df = df.sort_index(ascending=ascending)
    else:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # merge cut rows
    def sum_df(df, merge_metric):
        df_sum = _pd.DataFrame([[None]*len(df.columns)], columns=df.columns)
        for col in df.columns:
            try:
                df[col].astype('double')
                if merge_metric == 'mean':
                    df_sum.loc[0, col] = df[col].mean()
                else:
                    df_sum.loc[0, col] = df[col].sum()
            except:
                df_sum.loc[0, col] = '-'
        return df_sum
    
    df_merge = sum_df(df_merge, merge_metric=merge_metric).loc[0]
    # merge out-of-bag rows into Others row
    if df_merge[by] != 0:
        df.loc["Others"] = df_merge
    # put null back to the df
    if df_null is not None:
        df.loc[null_value] = df_null
    return df


def bring_null_to_end(df, null_value):
    if null_value in df.index:
        df_null = df.loc[null_value, :]
        df = df.drop(index=[null_value])
        df.loc[null_value] = df_null
    return df


def pearson(num1, num2):
    mean1 = num1.mean()
    mean2 = num2.mean()
    if _pd.isna(mean1) or _pd.isna(mean2):
        return 0
    # fix scipy 1.7.3, in 1.9.3 access by .statistic instead of [0]
    return scipy_stats.pearsonr(num1.fillna(mean1), num2.fillna(mean2))[0]


def get_mode_table(df):
    """
    Contributor: ThanhNM3
    Get mode value information of each column in a DataFrame.
    Only return a random mode value in case many modes are detected.
    """
    n_rows = df.shape[0]
    mode_values = []
    for col in df.columns:
        values = df[col].convert_dtypes().value_counts(dropna=False)
        mode_values.append([col, values.idxmax(), values.max()])

    mode_df = (
        _pd.DataFrame(mode_values, columns=["column", "mode", "mode_count"])
            .assign(mode_perc=lambda x: x["mode_count"] / n_rows)
            .set_index("column")
    )
    return mode_df


#num num
def calculate_bin_number(data_sorted, bin_edges):
    if len(bin_edges) <= 0:
        return _np.full(len(data_sorted), -1)
    i = 0
    e = 1
    bin_number = _np.zeros(len(data_sorted))
    for data in data_sorted:
        if e >= len(bin_edges) or data!=data or data is None:
            bin_number[i] = _np.NaN
        elif data <= bin_edges[e]:
            bin_number[i] = e-1
        else:
            bin_number[i] = e
            e+=1
        i+=1
    return bin_number


def calculate_bin_values(data_sorted, bin_edges):
    bin_index = calculate_bin_number(data_sorted, bin_edges)
    tmp = _pd.DataFrame({'bin': bin_index})
    df_count = tmp.groupby('bin').size()
    for i in range(len(bin_edges)-1):
        if i not in df_count.index:
            df_count.loc[i] = 0
    return df_count.sort_index()


def calculate_2d_binning(col1, col2, bin_edges1, bin_edges2):
    if len(bin_edges1) < 1 or len(bin_edges2) < 1:
        return None
    count_matrix = _np.full((len(bin_edges1)-1, len(bin_edges2)-1), _np.NaN)
    df_raw = _pd.DataFrame({'col1': col1, 'col2': col2})
    df_raw = df_raw.sort_values('col1')
    df_raw['bin1'] = calculate_bin_number(df_raw['col1'], bin_edges1)
    df_raw = df_raw.sort_values('col2')
    df_raw['bin2'] = calculate_bin_number(df_raw['col2'], bin_edges2)
    df_groupby = df_raw.groupby(['bin1', 'bin2']).size().reset_index().rename(columns={0: 'count'})
    for idx in df_groupby.index:
        row = df_groupby.iloc[idx]
        count_matrix[int(row['bin1']), int(row['bin2'])] = row['count']
    return count_matrix


def get_precision(x):
    return x>=1000


def describe_percentiles(df, col, percentiles=None, fmt=".2f"):
    """
    Contibuted by thanhnm3
    Generate descriptive statistics of a numeric column."""
    if percentiles is None:
        percentiles = [i * 0.1 for i in range(1, 10)] + [0.01, 0.05, 0.95, 0.99]

    percentile_df = _pd.to_numeric(df[col]).describe(percentiles=percentiles).to_frame().rename(columns={col: "value"})

    # Add Null count, Skewness & Kurtosis
    percentile_df.loc["nan"] = df[col].isnull().sum() / df.shape[0]
    percentile_df.loc["skew"] = df[col].skew()
    percentile_df.loc["kurt"] = df[col].kurtosis()
    percentile_df = percentile_df.loc[
                    ["count", "mean", "std", "min"] +
                    [f"{int(x * 100)}%" for x in sorted(percentiles)] + ["max"], :
                    ]
    return percentile_df


def series_get_percentiles(series, percentiles):
    result_series = _pd.to_numeric(series).describe(percentiles).loc[[f'{x:.0%}' for x in percentiles]]
    return result_series


def describe_percentiles_v2(df, col, percentiles=None, fmt=".2f"):
    """
    Contibuted by thanhnm3
    Generate descriptive statistics of a numeric column."""
    if percentiles is None:
        percentiles = [i * 0.1 for i in range(1, 10)] + [0.01, 0.05, 0.95, 0.99]
    quartiles = [0.25, 0.75]

    percentile_df = _pd.to_numeric(df[col]).describe(percentiles=percentiles+quartiles).to_frame().rename(columns={col: "value"})

    # Add Null count, Skewness & Kurtosis
    percentile_df.loc["count"] = df.shape[0]
    percentile_df.loc["non_null"] = df.shape[0] - df[col].isnull().sum()
    percentile_df.loc["null"] = (percentile_df.loc["count"] - percentile_df.loc["non_null"]) / df.shape[0]
    percentile_df.loc["skew"] = df[col].skew()
    percentile_df.loc["kurt"] = df[col].kurtosis()
    # percentile_df.loc["distinct"] = df[col].nunique()
    mode_series = df[col].mode()
    if len(mode_series) > 0:
        percentile_df.loc["mode"] = df[col].mode()[0]
    else:
        percentile_df.loc["mode"] =  None
    percentile_df.loc["mad"] = (df[col] - df[col].mean()).abs().mean()
    percentile_df.loc["range"] = percentile_df.loc["max", "value"] - percentile_df.loc["min", "value"]
    percentile_df.loc["IQR"] = percentile_df.loc["75%", "value"] - percentile_df.loc["25%", "value"]
    value_1 = percentile_df.loc["1%", "value"]
    value_99 = percentile_df.loc["99%", "value"]
    percentile_df.loc["tr_mean"] = df[(_pd.to_numeric(df[col]) >= value_1) & (_pd.to_numeric(df[col]) <= value_99)][col].astype('double').mean()
    percentile_df = percentile_df.loc[
                    ["count", "non_null", "null", "mean", "tr_mean", "std", "skew", "kurt", "min", "mode", "mad", "IQR"] +
                    [f"{int(x * 100)}%" for x in sorted(percentiles)] + ["max"], :
                    ]
    return percentile_df


def render_html(obj):
    if isinstance(obj, _pd.core.frame.DataFrame):
        return obj.to_html()
    elif isinstance(obj, _pd.io.formats.style.Styler):
        return obj.set_table_attributes("style='display:inline-block'").render()
    else:
        return obj


def display_hstack(arr: list, margin=75):
    render_arr = [render_html(x) for x in arr]
    return CustomHStack(
        render_arr,
        grid_row_style="display:flex;align-items:center;",
        grid_cell_style=f"margin-right:{margin}px;"
    )


def add_months(sourcedate, n_months):
    """
        Add n_months month to sourcedate
        Args:
            sourcedate (datetime): input date need add months
            n_months: number of months need add
        Returns:
            datetime result after add n_months
    """
    month = sourcedate.month - 1 + n_months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, _calendar.monthrange(year, month)[1])
    return _datetime(year, month, day)


def _psi_scale_range(input, min, max):
    input += -(_np.min(input))
    input /= _np.max(input) / (max - min)
    input += min
    return input


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        breakpoints = _np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = _psi_scale_range(breakpoints, _np.min(expected_array), _np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = _np.stack([_np.percentile(expected_array, b) for b in breakpoints])
        
        breakpoints[0] = _np.min([_np.min(expected_array), _np.min(actual_array)])
        breakpoints[-1] = _np.max([_np.max(expected_array), _np.max(actual_array)])

        expected_percents = _np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = _np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * _np.log(e_perc / a_perc)
            return (value)

        psi_value = _np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return (psi_value)

    if len(expected.shape) == 1:
        psi_values = _np.empty(len(expected.shape))
    else:
        psi_values = _np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return (psi_values)


def calculate_psi_v2(expected, actual, buckettype='bins', buckets=10, axis=0, featuretype=None, preagg=False):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: pandas dataframe or a series or an array of original values
       actual: pandas dataframe or a series or an array of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
       featuretype: str or dict defining featuretype of individual features which can be "num" or "cate"
       preagg: bool indicating whether the values are already aggregated e.g values of bins for numeric and value counts for categorical features. 

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected, actual, buckets, featuretype):
        '''Calculate the PSI for a single variable

        Args:
           expected: series of original values
           actual: series of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''
        if featuretype == 'cate':
            if not preagg:
                expected = _pd.Series(expected).value_counts()
                actual = _pd.Series(actual).value_counts()
            # concat and dropna to remove non-overlapped categories
            df = _pd.concat([expected, actual], axis=1).dropna()
            df.columns = ['expected', 'actual']
            expected_percents = (df['expected']/df['expected'].sum()).to_list()
            actual_percents = (df['actual']/df['actual'].sum()).to_list()
        elif featuretype == 'num':
            if preagg:
                expected_percents = expected/expected.sum()
                actual_percents = actual/actual.sum()
            else:
                breakpoints = _np.arange(0, buckets + 1) / (buckets) * 100

                if buckettype == 'bins':
                    breakpoints = _psi_scale_range(breakpoints, _np.min(expected), _np.max(expected))
                elif buckettype == 'quantiles':
                    breakpoints = _np.stack([_np.percentile(expected, b) for b in breakpoints])
                
                breakpoints[0] = _np.min([_np.min(expected), _np.min(actual)])
                breakpoints[-1] = _np.max([_np.max(expected), _np.max(actual)])

                expected_percents = _np.histogram(expected, breakpoints)[0] / len(expected)
                actual_percents = _np.histogram(actual, breakpoints)[0] / len(actual)
        else:
            raise Exception(f"Invalid value for featuretype. Valid values are `num`, `cate` but received {featuretype}.")

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * _np.log(e_perc / a_perc)
            return (value)

        if len(expected_percents) == 0:
            return -1
        
        psi_value = _np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return (psi_value)
    
    featuretype = featuretype or dict()
    if isinstance(expected, list):
        expected = _np.array(expected)
        actual = _np.array(actual)

    if isinstance(expected, _pd.DataFrame) and isinstance(actual, _pd.DataFrame):
        if axis==0:
            common_feats = list(set(expected.columns).intersection(set(actual.columns)))
        elif axis==1:
            common_feats = list(set(expected.index).intersection(set(actual.index)))

    if len(expected.shape) == 1:
        psi_values = _np.empty(1)
    else:
        psi_values = _pd.Series(index=common_feats)

    if len(psi_values) == 1:
        psi_values = psi(expected, actual, buckets, featuretype=featuretype if isinstance(featuretype, str) else 'num')
    else:
        for i in common_feats:
            if axis == 0:
                psi_values[i] = psi(expected.loc[:, i], actual.loc[:, i], 
                                    buckets, featuretype=featuretype.get(i, 'num'))
            elif axis == 1:
                psi_values[i] = psi(expected.loc[i, :], actual.loc[i, :], 
                                    buckets, featuretype=featuretype.get(i, 'num'))

    return (psi_values)



def check_psi(df_train, df_test, interest_cols, buckets=10):
    """
        PSI < 0.1 - No change. You can continue using existing model.
        PSI >=0.1 but less than 0.2 - Slight change is required.
        PSI >=0.2 - Significant change is required. Ideally, you should not use this model any more
    """
    psi_df = []
    for col_name in interest_cols:
        feature_train = df_train[col_name]
        feature_test = df_test[col_name]

        try:
            keep_na_score = calculate_psi(feature_train.fillna(-3), feature_test.fillna(-3), buckets=buckets)
            drop_na_score = calculate_psi(feature_train.dropna(), feature_test.dropna(), buckets=buckets)
            psi_df.append([col_name, keep_na_score, drop_na_score])
        except:
            print("skipped {}".format(col_name))

    psi_df = _pd.DataFrame(data=psi_df, columns=["col_name", "psi_keep_na", "psi_drop_na"])
    return psi_df


def config_plotly_bar_chart(fig, figsize=None, **kwargs):
    """Modify layout of a Plotly Bar chart plot.
    Args:
       fig: plotly figure
       figsize: a tuple display width and height
       **kwargs: additional keyword args for plotly fig layout
    """

    def get_plotly_bar_chart_figwidth(n):
        """Heuristic to determine suitable plotly figwidth base on number of category."""
        width = 300 if (n <= 2) else n * max(50, 140 * _np.power(0.92, n / 2))
        return width

    if figsize:
        width, height = figsize
    else:
        # Get number of categories
        n_bars = len(fig.data[0].x)
        width = get_plotly_bar_chart_figwidth(n_bars)
        height = 500

    return fig.update_traces(
        width=0.7,
        texttemplate="%{text:.0%}",
        # textposition='outside',
        marker_color="#1F77B4",  # xkcd:cerulean mai: #0485d1
    ).update_layout(
        width=width,
        height=height,
        autosize=True,
        title=dict(
            x=0,  # modify title's position
            font_size=18,
            font=dict(color="dimgray")
        ),
        xaxis=dict(
            # title=dict(text=col, font=dict(color="dimgray")),
            type="category",
            color="dimgray"
            # categoryorder='total descending'
        ),
        yaxis=dict(
            color="dimgray"
        ),
        **kwargs
    ).update_xaxes(
        ticks="",
        showline=True,
        linecolor="gray"
    ).update_yaxes(
        showline=True,
        linecolor="gray"
    )