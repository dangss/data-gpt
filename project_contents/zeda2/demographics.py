import pandas as _pd
import plotly.express as _px
import seaborn as _sns
import pathlib
_sns.set_style("whitegrid")
from pybloqs import Block as _Block, VStack as _VStack

from zeda2 import describe_utils as _describe_utils, data_format_helper as _data_format_helper
from zeda2.describe import describe_1d_category_with_percentiles as _describe_1d_category_with_percentiles, describe_1d_category_with_plot as _describe_1d_category_with_plot, plot_vietnam_province_map as _plot_vietnam_province_map

import pathlib
root_dir = str(pathlib.Path(__file__).parent.resolve())


# Fix bug not show chart
# _init_notebook_mode(connected=False)
_px.defaults.template = 'simple_white'
_pd.options.mode.chained_assignment = None
# occupation constant
_OCCUPATION_LV0_NAME="occupation_color"
_OCCUPATION_LV1_NAME="major_occupation_group"
prev_select = None
# living constants
_LIVING_THRESHOLD = 13


def describe_occupation(
        df, col,
        null_value='nan',
        threshold=(None, None),
        return_json=False,
        precomputed_kwargs=None
        ):
    if df is not None and col is not None:
        df_occupation = _data_format_helper.parse_job_id_v2(df, col)
        df_occupation['occupation_merge'] = df_occupation[_OCCUPATION_LV0_NAME].astype(str) + '|' + df_occupation[_OCCUPATION_LV1_NAME].astype(str)
        df_occupation.loc[df_occupation[_OCCUPATION_LV1_NAME].isna(), 'occupation_merge'] = 'Others|nan'
        # occupation color
        df_describe_color, _ = _describe_1d_category_with_percentiles(
            df_occupation,
            col=_OCCUPATION_LV0_NAME, 
            dropna=False, 
            threshold=threshold, 
            list_keep=None,
            category_orders=None,
            null_value=null_value)
        # occupation major group
        df_describe_lv1, _ = _describe_1d_category_with_percentiles(
            df_occupation, 
            col='occupation_merge', 
            dropna=False, 
            threshold=threshold, 
            list_keep=None, 
            category_orders=None,
            null_value=null_value)
    elif precomputed_kwargs is not None and 'df_color' in precomputed_kwargs and 'df_major' in precomputed_kwargs:
        # occupation color
        df_describe_color, _ = _describe_1d_category_with_percentiles(
            df=None,
            col=None,
            value_count_df=precomputed_kwargs['df_color'],
            count_col='count',
            dropna=False, 
            threshold=threshold, 
            list_keep=None,
            category_orders=None,
            null_value=null_value)
        # major occupation group
        df_major = precomputed_kwargs['df_major']
        df_major['occupation_merge'] = df_major[_OCCUPATION_LV0_NAME].astype(str) + '|' + df_major[_OCCUPATION_LV1_NAME].astype(str)
        df_major.loc[df_major[_OCCUPATION_LV1_NAME].isna(), 'occupation_merge'] = 'Others|nan'
        df_describe_lv1, _ = _describe_1d_category_with_percentiles(
            df=None, 
            col=None, 
            value_count_df=df_major.drop(columns=['occupation_color', 'major_occupation_group']).set_index('occupation_merge'),
            count_col='count',
            dropna=False, 
            threshold=threshold, 
            list_keep=None,
            category_orders=None,
            null_value='Others|nan')
    df_describe_lv1['occupation_color'] = df_describe_lv1.index.map(lambda x: str.split(x, '|')[0] if x!= 'Others' else 'Others')
    df_describe_lv1['major_occupation_group'] = df_describe_lv1.index.map(lambda x: str.split(x, '|')[1] if x!= 'Others' else 'Others')
    # to html
    df_describe_color_html = _describe_utils.format_table_cate_html(df_describe_color).rename(
        columns={'stat': 'Occupation Color'})
    # to html
    df_describe_lv1 = _describe_utils.bring_null_to_end(df_describe_lv1, 'nan')
    df_describe_lv1_html = _describe_utils.format_table_cate_html(df_describe_lv1, 'mix_color_major').rename(columns={'occupation_color': 'Occupation Color', 
            'major_occupation_group':'Major Occupation Group'})
    df_describe_lv1_html = df_describe_lv1_html.set_index("Major Occupation Group")
    df_describe_lv1_html.index = [f"<b>{idx}</b>" for idx in df_describe_lv1_html.index]
    df_describe_lv1_html.iloc[-1] = df_describe_lv1_html.iloc[-1].apply(lambda x: f'<div class="total_row">{x}</div>')
    current_index = list(df_describe_lv1_html.index)
    current_index[-1] = f'<div class="total_row">Total</div>'
    df_describe_lv1_html.index = current_index
    df_describe_lv1_html = df_describe_lv1_html.rename_axis("Major Occupation Group").reset_index()[['Major Occupation Group', 'Occupation Color', 'count', 'frequency (%)']]
    if return_json:
        return _describe_utils.df_to_json(df_describe_color_html), _describe_utils.df_to_json(df_describe_lv1_html)
    else:
        # app = JupyterDash(__name__)
        # app.layout = layout
        # app.run_server(mode="inline")
        # return app
        return _describe_utils.CustomHStack([df_describe_color_html, df_describe_lv1_html])
    

def describe_living(
        df, 
        col_province,
        col_country,
        dropna=False,
        threshold=(_LIVING_THRESHOLD, None),
        list_keep=None,
        null_value='nan',
        plot_kwargs={},
        orientation='v',
        show_percentile=False,
        fillna=None,
        outliers=['Hồ Chí Minh', 'Hà Nội'],
        return_json=False,
        table_index_name=None,
        ):
    df_temp = df[[col_province, col_country]].copy()
    if fillna:
        df_temp[col_province] = df_temp[col_province].fillna(fillna)
    df_temp.loc[(df_temp[col_country] != 'Vietnam') & (df_temp[col_country] != null_value) & (~df_temp[col_country].isna()), col_province] = 'Other Countries'
    # table and adhoc way to keep 'Other Countries' value
    if not list_keep and threshold[0] is not None:
        list_keep = list(df_temp[col_province].value_counts().index[:threshold[0]])
    if 'Other Countries' not in list_keep:
        list_keep.append('Other Countries')
    # map
    m = _plot_vietnam_province_map(df_temp, col_province=col_province, outliers=outliers, return_json=return_json)
    # precompute list_keep to keep other_country
    categorical_describe = _describe_1d_category_with_plot(
        df_temp, col_province, 
        threshold=threshold,
        list_keep=list_keep,
        orientation=orientation, 
        plot_kwargs=plot_kwargs,
        show_percentile=show_percentile,
        null_value=null_value,
        fillna=fillna,
        dropna=dropna,
        table_index_name=table_index_name,
        return_json=return_json
    )
    if return_json:
        return (*categorical_describe, m)
    return _VStack([categorical_describe, _Block(m._repr_html_())])