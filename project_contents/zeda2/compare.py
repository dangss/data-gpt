import pandas as pd
from zeda2 import describe, data_format_helper
from autoeda_report.bigdata.utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from autoeda_report.render_report_utils.utils import pyarrow_read_dir, pyarrow_read_csv, detect_feature_type, pyarrow_listdir
from autoeda_report.bigdata.demographics_utils import detect_feature_type_from_dir
from autoeda_report.bigdata.utils import transform_bin

from si_prefix import si_format
import os
import re


# CONSTANTS
_MAIN_COLOR = '#1F77B4'
_SECOND_COLOR = '#FF7F0E'
_GREY_COLOR = '#A9A9A9'
_NULL_COLOR = '#E45756'
_FIG_WIDTH=700

list_null_type = ['null', 'zero', 'negative', 'empty']
list_stat = ['#distinct', 'distinct (%)', 'null (%)', 'zero (%)', 'negative (%)', 'empty (%)']
list_stat_num = ['min', 'max', 'mean', 'std', 'median']
list_stat_num_left = [f'{stat}_left' for stat in list_stat_num]
list_func = [min, max, np.mean, np.std, np.median]

left_align_index_styles = [
    {
        'selector': 'tr th:first-child',
        'props': [('text-align', 'left !important')]
    },
    {
        'selector': 'tr td:first-child',
        'props': [('text-align', 'left !important')]
    },
    {
        'selector': 'thead th', 
        'props': [("text-align", "center !important")]
    }
]

inline_styles = [{
    'selector': 'th.col_heading',
    'props': [
        ('text-overflow', 'ellipsis'),
        ('overflow', 'hidden'),
        ('white-space', 'nowrap')
    ]
}]


def util_get_fig_height(n):
    """Heuristic to determine suitable plotly figwidth base on number of category."""
    height = 400 if (n <= 5) else n*max(50, 140*np.power(0.92, n/2))/1.9
    return height


def _compare_preprocess_dataset(asset_dict):
    # dataset type
    if 'df' in asset_dict:
        asset_dict['type'] = 'small'
    elif 'path' in asset_dict:
        path = asset_dict['path']
        if re.match('file://(.*)', path):
            asset_dict['type'] = 'small'
            asset_dict['df'] = _local_read_file(path)
        elif 'overview' in pyarrow_listdir(path):
            asset_dict['type'] = 'big'
        else:
            asset_dict['type'] = 'small'
            asset_dict['df'] = pyarrow_read_dir(path)
    else:
        raise Exception('Invalid assest dict. Either df or path must be provided.')
    # overview
    if asset_dict['type'] == 'small':
        df = asset_dict['df']
        df_overview = describe.data_overview(df, return_df=True)
        df_overview['distinct (%)'] = df_overview['#distinct']/df_overview['total']
        asset_dict['df_overview'] = df_overview
    elif asset_dict['type'] == 'big':
        df_overview = pyarrow_read_csv(f'{path}/overview').set_index('feature').rename(columns={'distinct': '#distinct'})
        for null_type in list_null_type:
            df_overview[f'{null_type} (%)'] = df_overview[f'missing{null_type.capitalize()}']/df_overview['count']
        df_overview['median'] = df_overview['percentiles'].apply(lambda x: ast.literal_eval(x)[49] if x != 'nan' else x)
        df_overview['distinct (%)'] = df_overview['#distinct']/df_overview['count']
        asset_dict['df_overview'] = df_overview
    # feature type
    if asset_dict['type'] == 'small':
        feature_type_map = asset_dict.get('feature_type_map', dict()).copy()
        asset_dict['declared_feature_type_map'] = feature_type_map.copy()
        
        df = asset_dict['df']
        for feat in df.columns:
            if feat not in feature_type_map:
                feature_type_map[feat] = detect_feature_type(df, feat)
        
        to_pop = []
        for feat in feature_type_map.keys():
            if feat not in df.columns:
                to_pop.append(feat)
        for pop in to_pop:
            feature_type_map.pop(pop)
        asset_dict['feature_type_map'] = feature_type_map
    elif asset_dict['type'] == 'big':
        # as for big, if feature_type_map is present it has all features
        if 'feature_type_map' not in asset_dict:
            root_dir = asset_dict["path"]
            list_feature = pyarrow_listdir(f'{root_dir}/1D/cate')+pyarrow_listdir(f'{root_dir}/1D/num')
            feature_type_map = detect_feature_type_from_dir(root_dir, {}, list_feature)
            asset_dict['feature_type_map'] = feature_type_map
            asset_dict['declared_feature_type_map'] = feature_type_map
            # adjust zero in big data
            asset_dict['df_overview'].loc[[feat for feat in list_feature if feature_type_map[feat] == 'cate'], 'zero (%)'] = 0

    return asset_dict


def _compare_preprocess_2_datasets(dict_left, dict_right):
    if 'common_feats' in dict_left and 'common_feats' in dict_right:
        return dict_left, dict_right
    
    dict_left = _compare_preprocess_dataset(dict_left.copy())
    dict_right = _compare_preprocess_dataset(dict_right.copy())
    common_feats = set(dict_left['feature_type_map'].keys()).intersection(set(dict_right['feature_type_map'].keys()))
    common_feats_cate = []
    common_feats_num = []
    for feat in common_feats:
        if dict_left['feature_type_map'][feat] == dict_right['feature_type_map'][feat]:
            if dict_left['feature_type_map'][feat] == 'cate':
                common_feats_cate.append(feat)
            elif dict_left['feature_type_map'][feat] == 'num':
                common_feats_num.append(feat)
    dict_left['common_feats'] = dict_right['common_feats'] = common_feats
    dict_left['common_feats_num'] = dict_right['common_feats_num'] = common_feats_num
    dict_left['common_feats_cate'] = dict_right['common_feats_cate'] = common_feats_cate
    return dict_left, dict_right


def _local_read_file(path):
    df = None
    if os.path.isfile(path):
        if re.match('.*csv$', path):
            df = pd.read_csv(path)
    else:
        try:
            df = pd.read_parquet(path)
        except:
            pass
    if df is None:
        raise Exception(f'Failed to read local file at {path}.')
    return df


def _compare_get_overview_unit(asset_dict):
    feature_type_map = asset_dict['feature_type_map']
    set_name = asset_dict['name']
    
    if asset_dict['type'] == "small":
        df = asset_dict['df']
        
        no_rows, no_features = df.shape
        no_cate = len([feat for feat in df.columns if feature_type_map[feat] == "cate"])
        no_num = len([feat for feat in df.columns if feature_type_map[feat] == "num"])
        no_dup = df.shape[0] - df.drop_duplicates().shape[0]
        
    if asset_dict['type'] == 'big':
        df_overview = asset_dict['df_overview']
        
        list_feature = [feat for feat in feature_type_map.keys() if 'UF_' not in feat]
        no_rows, no_features = df_overview['count'][0], len(list_feature)
        no_cate = len([feat for feat in list_feature if feature_type_map[feat] == 'cate'])
        no_num = len([feat for feat in list_feature if feature_type_map[feat] == 'num'])
    df_metadata = pd.DataFrame({set_name: [no_rows, no_features, no_cate, no_num]}, 
                               index=['#rows', '#features', '#categorical', '#numerical'])
    if asset_dict['type'] == 'small':
        df_metadata.loc['#duplicates', set_name] = no_dup
    return df_metadata


def compare_dataset_metadata(dict_left, dict_right):
    dict_left, dict_right = _compare_preprocess_2_datasets(dict_left, dict_right)
    common_feats = dict_left['common_feats']
    
    overview_left = _compare_get_overview_unit(asset_dict=dict_left)
    overview_right = _compare_get_overview_unit(asset_dict=dict_right)
    
    df_overview_compare = overview_left.merge(overview_right, left_index=True, right_index=True, how='right')
    # add no common_features row
    no_common_feat = len(common_feats)
    add_line = pd.DataFrame({dict_left['name']: no_common_feat, dict_right['name']: no_common_feat}, index=["#common_features"])
    df_overview_compare = pd.concat([df_overview_compare.iloc[:2], add_line, df_overview_compare.iloc[2:]])

    return (df_overview_compare
            .style
            .set_table_styles(left_align_index_styles + inline_styles)
            .format(formatter='{:,.0f}', na_rep='-'))


def compare_ftpye(dict_left, dict_right):
    dict_left, dict_right = _compare_preprocess_2_datasets(dict_left, dict_right)

    name_left = dict_left['name']
    name_right = dict_right['name']
    df_ftype = pd.DataFrame(columns=[name_left, name_right])
    for asset_dict in [dict_left, dict_right]:
        for feat, ftype in asset_dict['feature_type_map'].items():
            df_ftype.loc[feat, asset_dict['name']] = ftype
    df_ftype['is_diff'] = df_ftype.apply(lambda x: x[name_left] != x[name_right], axis=1)
        
    def highlight_dtype(row):
        """Heuristics to add visual cues."""
        ret = ['' for _ in row.index]
        style_left = ''
        style_right = ''
        is_diff = (row[name_left] != row[name_right])
        is_detected1 = row.name not in dict_left['declared_feature_type_map']
        is_detected2 = row.name not in dict_right['declared_feature_type_map']
        style_left += is_diff*f'font-weight: bold;' + is_detected1*'color: blue;'
        style_right += is_diff*f'font-weight: bold;' + is_detected2*'color: blue;'
        ret[row.index.get_loc(name_left)] = style_left
        ret[row.index.get_loc(name_right)] = style_right
        return ret
    
    list_red_styles = [{'selector': f'.row{idx:.0f}.level0', 'props': 'color: red'} 
                       for idx, is_diff in enumerate(df_ftype.sort_values(['is_diff', name_right], ascending=[True, False])['is_diff']) if is_diff]

    df_ftype_style = (df_ftype
            .rename_axis("feature", axis=1)
            .fillna('-')
            .sort_values(['is_diff', name_right], ascending=[True, False])
            .style
            .apply(highlight_dtype, axis=1).set_caption(f'''
                <div style="color: blue;">blue: auto detected</div>
                <div style="color: black">black: user declared</div>
                <div style="color: {_NULL_COLOR}; font-weight: bold;">red: excluded in further comparisons</div>''')
            .set_table_styles(left_align_index_styles+list_red_styles+inline_styles)
           )
    # to handle exception Styler does not have method "hide()"
    try:
        df_ftype_style = df_ftype_style.hide(subset=['is_diff'], axis=1)
    except:
        df_ftype_style = df_ftype_style.hide_columns(['is_diff'])
    
    return df_ftype_style


def compare_feature_metadata(left, right, name_left, name_right, metadata_title, fmt, return_raw=False):
    df_compare = pd.concat([left, right], axis=1).dropna()
    df_compare.columns=[name_left, name_right]
    df_compare['diff'] = df_compare[name_left]-df_compare[name_right]
    df_compare = df_compare.sort_values('diff', ascending=True)
    fig = make_subplots(cols=1, shared_yaxes=True)
    for col, color in zip(df_compare[[name_left, name_right]].columns, [_MAIN_COLOR, _SECOND_COLOR]):
        fig.add_trace(go.Bar(
            y=df_compare.index,
            x=df_compare[col],
            orientation='h',
            name=col,
            hoverinfo='x',
            hovertemplate=f'set={col}<br>feature = %{{y}}<br>null percentage= %{{x:.2%}}',
            marker=dict(color=color)
        ), row=1, col=1)
    # fig.add_trace(
    #     go.Bar(x=df_missing['abs(diff)'], 
    #            y=df_missing.index, 
    #            orientation='h',
    #            text=df_missing['abs(diff)'],
    #            texttemplate='%{text:.2%}',
    #            marker=dict(color=["#54A24B" if val < 0 else '#E45756' for val in df_missing['diff']]),
    #            name='abs(diff)'
    #           ), row=1, col=2)
    fig.update_layout(
        # height=490,
        height=util_get_fig_height(df_compare.shape[0]),
        width=600,
        title=f'Compare <b>{metadata_title}</b> of {name_left} and {name_right} dataset',
        xaxis_title=metadata_title,
        yaxis_title='feature'
    )
    vmin = df_compare['diff'].min()
    vmax = df_compare['diff'].max()
    df_compare.columns = pd.MultiIndex.from_tuples([(metadata_title, col) for col in df_compare.columns])
    #     diff_pos = pd.IndexSlice[df_compare.loc[df_compare[(metadata_title, 'diff')]>=0].index, (metadata_title, 'diff')]
    #     diff_neg = pd.IndexSlice[df_compare.loc[df_compare[(metadata_title, 'diff')]<0].index, (metadata_title, 'diff')]
    is_stat = any([stat in metadata_title for stat in list_stat_num+['#distinct']])
    if not is_stat:
        abs_diff = df_compare[(metadata_title, 'diff')].abs()
    else:
        abs_diff = (df_compare[(metadata_title, 'diff')]/df_compare[(metadata_title, name_left)]).abs()
        
    diff_red = pd.IndexSlice[df_compare.loc[abs_diff>=0.1].index, (metadata_title, 'diff')]
    diff_orange = pd.IndexSlice[df_compare.loc[(abs_diff>=0.05)&(abs_diff<0.1)].index, (metadata_title, 'diff')]
    diff_grey = pd.IndexSlice[df_compare.loc[abs_diff<0.05].index, (metadata_title, 'diff')]
    
    df_compare_style = (
        df_compare
        .rename_axis(("", "feature"), axis=1)
        .sort_values([(metadata_title, 'diff')], ascending=False)
        .style
        .format(formatter=fmt)
        .set_table_styles(left_align_index_styles+inline_styles)
        # .bar(subset=diff_red, color=_NULL_COLOR, vmin=vmin, vmax=vmax, height=70, align='zero')
        # .bar(subset=diff_orange, color='orange', vmin=vmin, vmax=vmax, height=70, align='zero')
        # .bar(subset=diff_grey, color=_GREY_COLOR, vmin=vmin, vmax=vmax, height=70, align='zero')
        .bar(subset=diff_red, color=_NULL_COLOR, vmin=vmin, vmax=vmax, align='zero')
        .bar(subset=diff_orange, color='orange', vmin=vmin, vmax=vmax, align='zero')
        .bar(subset=diff_grey, color=_GREY_COLOR, vmin=vmin, vmax=vmax, align='zero')
        .set_caption(f'''
                        <div style="display: inline-block; width: 10px; height: 10px; background-color: {_NULL_COLOR};"></div>
                        <div style="display: inline-block;">&nbsp; abs(diff) &ge; 10% {is_stat*f"of {name_left} value"}</div><br>
                        <div style="display: inline-block; width: 10px; height: 10px; background-color: orange;"></div>
                        <div style="display: inline-block;">&nbsp; abs(diff) &ge; 5% {is_stat*f"of {name_left} value"}</div><br>
                        <div style="display: inline-block; width: 10px; height: 10px; background-color: {_GREY_COLOR};"></div>
                        <div style="display: inline-block;">&nbsp; abs(diff) < 5% {is_stat*f"of {name_left} value"}</div>''')
    )

    if return_raw:
        return df_compare_style, fig
    return describe_utils.CustomHStack([describe_utils.render_html(df_compare_style), fig])


def compare_feature_metadata_overview(dict_left, dict_right, mode='both'):
    dict_left, dict_right = _compare_preprocess_2_datasets(dict_left, dict_right)
    common_feats = dict_left['common_feats']
    common_feats_num = dict_left['common_feats_num']
    common_feats_cate = dict_left['common_feats_cate']
    
    def highlight(row):
        """Heuristics to add visual cues."""
        ret = ['' for _ in row.index]
        is_stat = mode in ['both', 'stat']
        is_general = mode in ['both', 'general']
        
        if is_general:
            list_stat = ['distinct (%)', 'null (%)', 'zero (%)', 'negative (%)', 'empty (%)']
            for stat in list_stat:
                if abs(row[stat]) >= 0.1:
                    ret[row.index.get_loc(stat)] = f'color: {_NULL_COLOR}; font-weight: bold'
                elif abs(row[stat]) >= 0.05:
                    ret[row.index.get_loc(stat)] = 'color: orange; font-weight: bold'
            
            if row['ftype']=='cate':
                for cate_exclude in ['distinct (%)']:
                    ret[row.index.get_loc(cate_exclude)] = 'background-color: #C8C8C8;'

        if is_stat:
            for stat in list_stat_num:
                if row['ftype'] == 'num':
                    # handle float division by zero
                    if row[f'{stat}_left'] != 0:
                        diff_ratio = abs(row[stat]/row[f'{stat}_left'])
                    else:
                        diff_ratio = float('nan')

                    if diff_ratio >= 0.1:
                        ret[row.index.get_loc(stat)] = f'color: {_NULL_COLOR}; font-weight: bold'
                    elif diff_ratio >= 0.05:
                        ret[row.index.get_loc(stat)] = 'color: orange; font-weight: bold'
            
            if row['ftype']=='cate':
                for cate_exclude in list_stat_num:
                    ret[row.index.get_loc(cate_exclude)] = 'background-color: #C8C8C8;'

        return ret

    df_overview_left = dict_left['df_overview'].copy()
    df_overview_right = dict_right['df_overview'].copy()
    feature_type_map = dict_left['feature_type_map'].copy()
    if mode in ['both', 'general']:
        df_diff = pd.DataFrame(index=list(common_feats))
    elif mode == 'stat':
        df_diff = pd.DataFrame(index=list(common_feats_num))
    name_left = dict_left['name']
    name_right = dict_right['name']
    
    if mode in ['both', 'general']:
        for stat in list_stat:
            df_diff[stat] = df_overview_left.loc[df_diff.index, stat] - df_overview_right.loc[df_diff.index, stat]
    if mode in ['both', 'stat']:
        for feat in common_feats_num:
            list_values = []
            for asset_dict, df_overview in zip([dict_left, dict_right], [df_overview_left, df_overview_right]):
                if asset_dict['type'] == 'small':
                    stat_values = np.array([func(asset_dict['df'][feat].dropna()) for func in list_func])
                    df_overview.loc[feat, list_stat_num] = stat_values
                    list_values.append(stat_values)
                elif asset_dict['type'] == 'big':
                    df_overview.loc[feat, list_stat_num] = df_overview.loc[feat, list_stat_num].astype(float)
                    list_values.append(df_overview.loc[feat, list_stat_num])
            list_diff = list_values[0] - list_values[1]
            df_diff.loc[feat, list_stat_num] = list_diff
            df_diff.loc[feat, [f'{stat}_left' for stat in list_stat_num]] = list(list_values[0])
    
    df_ftype = pd.DataFrame(index=df_diff.index)
    if len(common_feats_num) > 0:
        df_ftype.loc[common_feats_num, 'ftype'] = 'num'
    if mode in ['both', 'stat']:
        df_diff[(df_diff==0)[df_diff.columns.drop(list_stat_num+list_stat_num_left)]]=float('nan')
    if mode in ['both', 'general']:
        if len(common_feats_cate) > 0:
            df_ftype.loc[common_feats_cate, 'ftype'] = 'cate'
            df_diff.loc[common_feats_cate, 'distinct (%)'] = float('nan')
    df_diff = pd.concat([df_ftype, df_diff], axis=1)
    
    def get_fmt(col):
        if '%' in col:
            return '.2%'
        elif '#' in col:
            return ',.0f'
        else:
            return ',.2f'
        
    df_tool_tips = pd.DataFrame(index=df_diff.index, columns=df_diff.columns)
    for idx in df_tool_tips.index:
        for col in df_tool_tips.columns:
            if (col == 'ftype' or  
                ((col in ['distinct (%)']+list_stat_num) and feature_type_map[idx]=='cate')
               ):
                df_tool_tips.loc[idx, col] = None
            elif col in df_overview_left.columns:
                fmt = get_fmt(col)
                left_value = df_overview_left.loc[idx, col]
                right_value = df_overview_right.loc[idx, col]
                try:
                    left_value = f'{left_value:{fmt}}'
                    right_value = f'{right_value:{fmt}}'
                except:
                    pass
                df_tool_tips.loc[idx, col] = f'{name_left} | {left_value} - {right_value} | {name_right}'
    
    tooltip_props = [('visibility', 'hidden'),
                     ('position', 'absolute'),
                     ('z-index', 1),
                     ('background-color', 'black'),
                     ('color', 'white'),
                     ('transform', 'translate(-20px, -20px)'),
                     ('padding', '5px'),
                     ('opacity', '0.8')
                    ]
    
    border_props = [
        ("border-left-color", "black"),
        ("border-left-style", "dashed"),
        ("border-left-width", "thin"),
    ]
    
    list_border_props = [
        dict(selector='th.col_heading.level0.col1', props=border_props),
        dict(selector='th.col_heading.level0.col3', props=border_props),
        dict(selector='th.col_heading.level0.col7', props=border_props)]
    
    is_stat = mode == 'stat'
    if mode == 'stat':
        df_diff = df_diff.sort_values('ftype', ascending=False)
    else:
        df_diff = df_diff.sort_values(['ftype', 'null (%)'], ascending=False)
        
    df_style = (
        df_diff
        .rename_axis("feature", axis=1)
        .style
        .set_caption(f'''<div style="color: black;font-weight: bold;">diff = {name_left} -  {name_right}</div>
                        <div style="display: inline-block; width: 10px; height: 10px; background-color: {_NULL_COLOR};"></div>
                        <div style="display: inline-block;">&nbsp; abs(diff) &ge; 10% {is_stat*f"of {name_left} value"}</div><br>
                        <div style="display: inline-block; width: 10px; height: 10px; background-color: orange;"></div>
                        <div style="display: inline-block;">&nbsp; abs(diff) &ge; 5% {is_stat*f"of {name_left} value"}</div><br>''')
        .set_table_styles(left_align_index_styles+list_border_props+inline_styles)
        .set_tooltips(df_tool_tips, props=tooltip_props)
        .set_properties(**dict(border_props), subset=['#distinct', 'null (%)', 'min'])
        .apply(highlight, axis=1)
    )
    if mode in ['both', 'general']:
        df_style = (df_style            
            .format(formatter='{:.2%}', subset=pd.IndexSlice[common_feats_num, 'distinct (%)'], na_rep='-')
            .format(formatter='{:,.0f}', subset=['#distinct'], na_rep='-')
            .format(formatter='{:.2%}', subset=list_stat[1:], na_rep='-')
            .format(formatter=None, subset=pd.IndexSlice[common_feats_cate, 'distinct (%)'], na_rep='')
        )
    if mode in ['both', 'stat']:
        df_style = (df_style
            .format(formatter='{:,.2f}', subset=pd.IndexSlice[common_feats_num, list_stat_num], na_rep='-')
        )
        try:
            df_style = df_style.hide(subset=[f'{stat}_left' for stat in list_stat_num], axis=1)
        except:
            df_style = df_style.hide_columns([f'{stat}_left' for stat in list_stat_num])
        
    if mode == 'both':
        df_style = df_style.format(formatter=None, subset=pd.IndexSlice[common_feats_cate, list_stat_num], na_rep='')
    return df_style


def _psi_cate_preagg(col, df=None, root_dir=None, category_rename=None):
    # feature to handle specifically
    col_major = 'major_occupation_group'
    col_color = 'occupation_color'
    if root_dir is not None:
        df = pyarrow_read_dir(f'{root_dir}/1D/cate/{col}')
        if category_rename and col in category_rename:
            df[col] = df[col].apply(lambda x: category_rename[col].get(x, x))
        if col == col_major:
            df[col_major] = (df[col_major]*1e+4).apply(data_format_helper.get_job_name)
            df[col_color] = (df[col_color]).apply(data_format_helper.get_color_name)
            # preprocess name
            df = (data_format_helper
                  ._preprocess_occupation_name(df, col_color, col_major)
                  [[col_color, col_major, 'count']]
                  .dropna(subset=[col_major])
                 )
            # reassign col_major aka col
            df[col] = df.apply(lambda x: f'{x[col_color]}|{x[col_major]}', axis=1)
        return df.dropna(subset=[col]).groupby(col).agg({'count': 'sum'})['count']
    elif df is not None:
        df_temp = df[[col]].copy()
        if category_rename and col in category_rename:
            df_temp[col] = df_temp[col].apply(lambda x: category_rename[col].get(x, x))
        if col == col_major:
            df_temp[col_color] = df[col_color]
            df_temp[col] = df_temp.apply(lambda x: f'{x[col_color]}|{x[col_major]}', axis=1)
        return df_temp[col].dropna().value_counts()
    else:
        raise Exception(f"_psi_cate_preagg: Either df or root_dir must be provided.")
    

def _compare_psi_table(dict_left, dict_right, df_psi_prev=None, nbins=10):
    common_feats = dict_left['common_feats']
    if df_psi_prev is not None:
        common_feats = list(set(common_feats).difference(set(df_psi_prev.index)))
        df_psi = df_psi_prev.copy()
    else:
        df_psi = pd.DataFrame(columns=['type', 'psi'])

    for feat in common_feats:
        featuretype = dict_left['feature_type_map'].get(feat, None)
        if featuretype == 'cate':
            list_value_counts = []
            for asset_dict in [dict_left, dict_right]:
                if asset_dict['type'] == 'small':
                    value_counts = _psi_cate_preagg(df=asset_dict['df'], col=feat, category_rename=asset_dict.get('category_rename', None))
                elif asset_dict['type'] == 'big':
                    value_counts = _psi_cate_preagg(root_dir=asset_dict['path'], col=feat, category_rename=asset_dict.get('category_rename', None))
                list_value_counts.append(value_counts)
            psi = describe_utils.calculate_psi_v2(list_value_counts[0], list_value_counts[1], preagg=True, featuretype='cate')
        elif featuretype == 'num':
            # if big - small, big is always in 0 index
            # if big - big, return -1
            # if small - small, any order
            ordered_list = [None, None]
            for asset_dict in [dict_left, dict_right]:
                if asset_dict['type'] == 'big':
                    # big - big
                    if ordered_list[0]:
                        raise Exception('Not support for big - big comparision yet.')
                    # big - small
                    else:
                        ordered_list[0] = asset_dict
                # small - small
                elif asset_dict['type'] == 'small':
                    if ordered_list[1]:
                        ordered_list[0] = asset_dict
                    else:
                        ordered_list[1] = asset_dict

            if ordered_list[0]['type'] == 'big':
                df_left = pyarrow_read_dir(f'{ordered_list[0]["path"]}/1D/num/{feat}/bin{nbins}_exact')
                _, bin_edges_left, bin_edges_refer = get_refer_nbins(feat, ordered_list[0]['df_overview'], 100)
                _, bin_values_left = transform_bin(bin_edges_left, bin_edges_refer, df_left, f'{feat}_bin{nbins}_exact')
            elif ordered_list[0]['type'] == 'small':
                bin_edges_left, bin_values_left, _, _ = describe_utils.calculate_bin(
                    ordered_list[0]['df'][feat].dropna(), 
                    nbins=nbins, 
                    binsize=None, 
                    non_empty_bins=True)

            # adjust bin_edges_left and compute bin_values_right accordingly
            df_right = ordered_list[1]['df']
            bin_edges_left[0] = min(bin_edges_left[0], df_right[feat].min())
            bin_edges_left[-1] = max(df_right[feat].max(), bin_edges_left[-1])
            bin_values_right = np.histogram(df_right[feat], bin_edges_left)[0]

            # compute psi
            psi = describe_utils.calculate_psi_v2(bin_values_left, bin_values_right, featuretype='num', preagg=True)
        df_psi.loc[feat] = [featuretype, psi]
    return df_psi


def _compare_psi_table_style(df_psi):
    
    def highlight_psi(row):
        """Heuristics to add visual cues."""
        ret = ['' for _ in row.index]

        if row['psi'] == -1:
            ret[row.index.get_loc('psi')] = 'background-color: grey; font-weight: bold'
        elif abs(row['psi']) < 0.25 and abs(row['psi']) >= 0.1:
            ret[row.index.get_loc('psi')] = 'background-color: orange; font-weight: bold'
        elif abs(row['psi']) >= 0.25:
            ret[row.index.get_loc('psi')] = f'background-color: {_NULL_COLOR}; font-weight: bold'

        return ret

    df_psi_style = (df_psi
     .sort_values('psi', ascending=False)
     .style
     .format(formatter='{:.4f}', subset=['psi'])
     .apply(highlight_psi, axis=1)
     .set_table_styles(inline_styles + left_align_index_styles)
     .set_caption(f'''<div>PSI Value Interpretation</div>
                     <div style="display: inline-block; width: 10px; height: 10px; background-color: {_NULL_COLOR};"></div>
                     <div style="display: inline-block;">&nbsp; PSI &ge; 0.25: Significant difference</div><br>
                     <div style="display: inline-block; width: 10px; height: 10px; background-color: orange;"></div>
                     <div style="display: inline-block;">&nbsp; PSI < 0.25: Moderate difference</div><br>
                     <div style="display: inline-block; width: 10px; height: 10px; background-color: #f0f0f0;"></div>
                     <div style="display: inline-block;">&nbsp; PSI < 0.1: Insignificant difference</div><br>
                     <div style="display: inline-block; width: 10px; height: 10px; background-color: grey;"></div>
                     <div style="display: inline-block;">&nbsp; PSI = -1: No common categories</div>''')
    )
    return df_psi_style


def _compare_psi_plot(df_psi):
    df_psi = df_psi.sort_values('psi', ascending=True, key=abs).drop(index=df_psi[df_psi['psi']==-1].index)
    list_color = [_NULL_COLOR if psi >= 0.25 else 'orange' if psi >=0.1 else '#d0d0d0' for psi in df_psi['psi']]
    fig = (
        go.Figure(
            go.Bar(x=df_psi.psi, y=df_psi.index, 
                   orientation='h', marker=dict(color=list_color),
                   texttemplate='%{x:.4f}'
                  )
        )).update_layout(
        width=_FIG_WIDTH,
        # height=_FIG_HEIGHT,
        height=util_get_fig_height(df_psi.shape[0]),
        title=f'PSI values of common features between {df_psi.columns[0]} and {df_psi.columns[1]} datasets'
    )
    return fig


def compare_psi(dict_left, dict_right, return_raw=False):
    dict_left, dict_right = _compare_preprocess_2_datasets(dict_left, dict_right)
    common_feats_cate = dict_left['common_feats_cate']

    df_psi_10bins = _compare_psi_table(dict_left, dict_right, None, 10)
    df_psi_10bins_style = _compare_psi_table_style(df_psi_10bins)
    fig_10bins = _compare_psi_plot(df_psi_10bins)
    # 20 bins, reuse categorical features
    df_psi_20bins = _compare_psi_table(dict_left, dict_right, df_psi_10bins.loc[common_feats_cate], 20)
    df_psi_20bins_style = _compare_psi_table_style(df_psi_20bins)
    fig_20bins = _compare_psi_plot(df_psi_20bins)

    if return_raw:
        return df_psi_10bins_style, fig_10bins, df_psi_20bins_style, fig_20bins
    
    return describe_utils.CustomHStack(
        ['<h4>10bins</h4>', 
         describe_utils.render_html(df_psi_10bins_style), 
         fig_10bins,
         '<h4>20bins</h4>',
         describe_utils.render_html(df_psi_20bins_style), 
         fig_20bins])


def compare_cate(dict_left, dict_right, col, threshold=10, fmt='.2%', sort_key=None, return_raw=False):
    dict_left, dict_right = _compare_preprocess_2_datasets(dict_left, dict_right)

    sort_col = 'diff' if not sort_key else 'abs(diff)'
    name_left = dict_left['name']
    name_right = dict_right['name']
    df_meta_cate = pd.DataFrame(columns = [name_left, name_right], index=["count", "null (%)", "#distinct", "#common_cate", "#distinct_cate", "mode", "mode (%)"])
    list_set_cate = []
    list_df = []
    
    # table metadata
    for asset_dict in [dict_left, dict_right]:
        name = asset_dict['name']
        df_overview = asset_dict['df_overview']
        
        if asset_dict['type'] == 'small':
            df = asset_dict['df']
            # process df_cate
            df_cate, _ = describe.describe_1d_category_with_percentiles(df, col, threshold=None)
            df_cate = df_cate.rename(columns={'frequency (%)': name})[[name]]
            # extract count total
            count_val = df_overview.loc[col, 'total']
        elif asset_dict['type'] == 'big':
            path = asset_dict['path']
            category_rename = asset_dict.get('category_rename', dict())
            # process df_cate
            df_cate = pyarrow_read_dir(f'{path}/1D/cate/{col}')
            df_cate[col] = df_cate[col].apply(lambda x: category_rename.get(col, dict()).get(x, x)).fillna("nan").map(str)
            # to handle fillna failure with NaN in Float64 column
            # handle Worker -> Blue
            df_cate = df_cate.groupby(col, as_index=False).agg({'count': 'sum'})
            
            df_cate['freq'] = df_cate['count']/df_cate['count'].sum()
            df_cate['freq'] = df_cate['freq'].astype(pd.Float64Dtype())
            df_cate = df_cate.rename(columns={'freq': name}).assign(**{col: lambda x: x[col].fillna('nan')}).set_index(col)[[name]]
            # extract count total
            count_val = df_overview.loc[col, 'count']
        
        df_cate = df_cate.sort_values(name, ascending=False)
        list_df.append(df_cate)
        list_set_cate.append(set(df_cate.index))
        null_val = asset_dict['df_overview'].loc[col, 'null (%)']
        
        mode_val = None
        mode_perc = None
        if df_cate.shape[0] > 0:
            mode_val = df_cate.iloc[0].name
            mode_perc = df_cate.iloc[0][0]
        df_meta_cate[name] = [count_val, null_val, df_cate.shape[0], None, None, mode_val, mode_perc]
    
    no_common_cates = len(list_set_cate[0].intersection(list_set_cate[1]))
    no_distinct_left = len(list_set_cate[0].difference(list_set_cate[1]))
    no_distinct_right = len(list_set_cate[1].difference(list_set_cate[0]))
    df_meta_cate.loc['#common_cate'] = no_common_cates
    df_meta_cate.loc['#distinct_cate'] = [no_distinct_left, no_distinct_right]

    df_meta_cate_style = (
        df_meta_cate
        .rename_axis("metadata", axis=1)
        .style
        .format(formatter='{:.2%}', subset=pd.IndexSlice[['null (%)', 'mode (%)'], :])
        .format(formatter='{:,.0f}', subset=pd.IndexSlice[['count'], :])
        .set_table_styles(left_align_index_styles+inline_styles)
    )
    
    # table
    df_merge = list_df[1].merge(list_df[0], left_index=True, right_index=True, how="outer").fillna(0)
    df_merge['diff']=df_merge[name_left]-df_merge[name_right]
    df_merge['abs(diff)']=df_merge['diff'].abs()
    df_merge[['diff', 'abs(diff)']] = df_merge[['diff', 'abs(diff)']].fillna(0)
    df_merge = df_merge.sort_values('abs(diff)', ascending=False)[:threshold].sort_values(sort_col, ascending=False, key=sort_key)
    df_merge.index.name=None
    
    # diff_pos = pd.IndexSlice[df.loc[df['diff']>0].index, 'diff']
    # diff_neg = pd.IndexSlice[df.loc[df['diff']<=0].index, 'diff']
    df_table = df_merge.copy()
    df_table.index = df_table.index.map(lambda x: x.replace('-', '&#8209;').replace(' ', '&nbsp;'))
    
    abs_diff = df_table[sort_col].abs()
    diff_red = pd.IndexSlice[df_table.loc[abs_diff>=0.1].index, sort_col]
    diff_orange = pd.IndexSlice[df_table.loc[(abs_diff>=0.05)&(abs_diff<0.1)].index, sort_col]
    diff_grey = pd.IndexSlice[df_table.loc[abs_diff<0.05].index, sort_col]
    vmin = df_table[sort_col].min()
    vmax = df_table[sort_col].max()

    df_style = (
            df_table[[name_left, name_right, sort_col]]
            .rename_axis(None, axis=0)
            .style
            .format(formatter=f'{{:{fmt}}}')
            # .bar(subset=diff_red, color=_NULL_COLOR, vmin=vmin, vmax=vmax, height=70, align='zero')
            # .bar(subset=diff_orange, color='orange', vmin=vmin, vmax=vmax, height=70, align='zero')
            # .bar(subset=diff_grey, color=_GREY_COLOR, vmin=vmin, vmax=vmax, height=70, align='zero')
            .bar(subset=diff_red, color=_NULL_COLOR, vmin=vmin, vmax=vmax, align='zero')
            .bar(subset=diff_orange, color='orange', vmin=vmin, vmax=vmax, align='zero')
            .bar(subset=diff_grey, color=_GREY_COLOR, vmin=vmin, vmax=vmax, align='zero')
            .set_table_styles(left_align_index_styles+inline_styles)
            .set_caption(f'''
                            <div style="display: inline-block; width: 10px; height: 10px; background-color: {_NULL_COLOR};"></div>
                            <div style="display: inline-block;">&nbsp; abs(diff) &ge; 10%</div><br>
                            <div style="display: inline-block; width: 10px; height: 10px; background-color: orange;"></div>
                            <div style="display: inline-block;">&nbsp; abs(diff) &ge; 5%</div><br>
                            <div style="display: inline-block; width: 10px; height: 10px; background-color: {_GREY_COLOR};"></div>
                            <div style="display: inline-block;">&nbsp; abs(diff) < 5%</div>''')
        )
    
    # fig
    fig = go.Figure()
    df_merge = df_merge[::-1]
    for set_name, color in zip(df_merge.columns, [_MAIN_COLOR, _SECOND_COLOR]):
        trace = go.Bar(
            y=df_merge.index,
            x=df_merge[set_name],
            orientation='h',
            name=set_name,
            hoverinfo='x',
            hovertemplate=f'set={set_name}<br>{col}=%{{y}}<br>freq=%{{x:.2%}}',
            marker=dict(color=color)
        )
        fig.add_trace(trace)
    fig.update_layout(
        width=700,
        # height=400,
        height=util_get_fig_height(df_merge.shape[0]),
        title=f'Compare {col} of {name_left} and {name_right} dataset',
        xaxis_title='frequency',
        yaxis_title=col,
        xaxis_tickformat=',.0%',
        legend=dict(
            xanchor='right',
            yanchor='top',
            x=1,
            y=1.1)
    )

    if return_raw:
        return df_meta_cate_style, df_style, fig
    
    return describe_utils.CustomHStack([describe_utils.render_html(df_meta_cate_style), describe_utils.render_html(df_style), fig], grid_cell_style='margin-left:35px')


def compare_num(dict_left, dict_right, col, nbins=100, fmt=',.2f', return_raw=False):
    dict_left, dict_right = _compare_preprocess_2_datasets(dict_left, dict_right)

    ordered_list = [None, None]
    for asset_dict in [dict_right, dict_left]:
        if asset_dict['type'] == 'big':
            # big - big
            if ordered_list[0]:
                raise Exception('Not support for big - big comparision yet.')
            # big - small
            else:
                ordered_list[0] = asset_dict
        # small - small
        elif asset_dict['type'] == 'small':
            if ordered_list[1]:
                ordered_list[0] = asset_dict
            else:
                ordered_list[1] = asset_dict
                
    name_left = ordered_list[0]['name']
    name_right = ordered_list[1]['name']

    def highlight_num(row):
        """Heuristics to add visual cues."""
        ret = ['' for _ in row.index]

        if abs(row['diff']/row[name_left]) >= 0.1:
            ret[row.index.get_loc('diff')] = f'color: {_NULL_COLOR}; font-weight: bold'
        elif abs(row['diff']/row[name_left]) >= 0.05:
            ret[row.index.get_loc('diff')] = f'color: orange; font-weight: bold'

        return ret
    
    binsize = None
    list_df_stat = []
    box_traces = []
    hist_traces = []
    scale_factors = []
    for asset_dict in ordered_list:
        name = asset_dict['name']

        if asset_dict['type'] == 'big':
            path = asset_dict['path']
            df_overview = asset_dict['df_overview']
            scale_factor = df_overview["count"][0]

            # histogram
            df_num = pyarrow_read_dir(f'{path}/1D/num/{col}/bin100')
            _, bin_edges, bin_edges_refer = get_refer_nbins(col, df_overview, 100)
            _, bin_values = transform_bin(bin_edges, bin_edges_refer, df_num, f'{col}_bin100')
            bin_centers = describe_utils.get_bin_centers_from_edges(bin_edges)
            binsize = bin_edges[1]-bin_edges[0]
            # table
            quantiles = ast.literal_eval(df_overview['percentiles'].loc[col])
            df_percentile = pd.DataFrame({
                'value': quantiles if quantiles else float('nan'), 
                'stat': [f'{percentile/100:.0%}' for percentile in range(1, 100)]}).set_index('stat')
            for stat in ['count', 'min', 'max', 'mean', 'std']:
                df_percentile.loc[stat] = df_overview[stat].loc[col]
            df_percentile.loc['null'] = df_overview['missingNull'].loc[col]/df_overview['count'].loc[col]
            df_percentile.loc['non_null'] = df_overview['count'].loc[col]-df_overview['missingNull'].loc[col]
            df_stat = describe.describe_1d_numeric(
                None, col, 
                percentile_df=df_percentile, 
                fmt='.0f',
                is_simple=True, raw=True).astype(float)
            box_trace = describe_utils.get_box_trace(series=None, series_percentile=df_percentile['value'], fmt=fmt)
        elif asset_dict['type'] == 'small':
            df = asset_dict['df']
            scale_factor = df.shape[0]

            # histogram
            bin_edges, bin_values, _, _ = describe_utils.calculate_bin(
                df[col].dropna(),
                nbins=nbins,
                binsize=binsize,
                non_empty_bins=binsize is None
            )
            bin_centers = describe_utils.get_bin_centers_from_edges(bin_edges)
            # table
            df_stat = describe.describe_1d_numeric(
                df, col, 
                fmt='.0f',
                is_simple=True,
                raw=True
            )
            box_trace = describe_utils.get_box_trace(df[col], fmt='.0f')
        
        if len(bin_edges)>=2:
            text = [f'{name}<br>{col}={left_edge}-{right_edge}' for left_edge, right_edge in zip(bin_edges[:-1],bin_edges[1:])]
        else:
            text = []

        hist_trace = go.Bar(x=bin_centers, y=np.array(bin_values)/scale_factor, customdata=np.array(bin_values), name=name, hovertext=text, hovertemplate='%{hovertext}<br>count=%{customdata:,.0f}')

        box_traces.append(box_trace)
        hist_traces.append(hist_trace)
        list_df_stat.append(df_stat)
        scale_factors.append(scale_factor)
            
    df_merge_num = list_df_stat[0].merge(list_df_stat[1], left_index=True, right_index=True)
    df_merge_num.columns=[name_left, name_right]
    df_merge_num['diff'] = df_merge_num[name_left] - df_merge_num[name_right]
    
    # format table
    df_merge_num_style = (df_merge_num
     .rename_axis("stat", axis=1)
     .style
     .format(formatter='{:,.0f}', subset=pd.IndexSlice[['count', 'null', 'non_null'], :])
     .format(formatter='{:,.0f}', subset=pd.IndexSlice[['count', 'null', 'non_null'], :])
     .format(formatter=f'{{:{fmt}}}', subset=pd.IndexSlice[df_merge_num.index.drop(['skew', 'kurt', 'count', 'null', 'non_null'], errors='ignore'), :])
     .apply(highlight_num, axis=1)
     .set_caption(f'''<div style="display: inline-block; width: 10px; height: 10px; background-color: {_NULL_COLOR};"></div>
                     <div style="display: inline-block;">&nbsp; abs(diff) &ge; 10% of {name_left} value</div><br>
                     <div style="display: inline-block; width: 10px; height: 10px; background-color: orange;"></div>
                     <div style="display: inline-block;">&nbsp; abs(diff) &ge; 5% of {name_left} value</div><br>''')
    )
    if all(idx in df_merge_num.index for idx in ['skew', 'kurt']):
        df_merge_num_style = df_merge_num_style.format(formatter='{:.2f}', subset=pd.IndexSlice[['skew', 'kurt'], :])
    
    # fig
    hist_traces[0].update(xaxis="x", yaxis="y", showlegend=True, opacity=0.6)
    hist_traces[1].update(xaxis="x2", yaxis="y2", showlegend=True, opacity=0.6)
    box_traces[0].update(xaxis="x3", yaxis="y3", name=name_left, showlegend=False)
    box_traces[1].update(xaxis="x4", yaxis="y4", name=name_right, showlegend=False, marker=dict(color=_SECOND_COLOR))
    
    # fig = go.Figure([kde_small, kde_big, box_small, box_big])
    fig = go.Figure([*hist_traces, *box_traces])
    fig.layout.xaxis = {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': col}}
    fig.layout.xaxis2 = {'anchor': 'y2', 'domain': [0.0, 1.0], 'showticklabels': False, 'overlaying': 'x', 'matches': 'x'}
    fig.layout.xaxis3 = {'anchor': 'y3', 'domain': [0.0, 1.0], 'matches': 'x', 'showticklabels': False}
    fig.layout.xaxis4 = {'anchor': 'y4', 'domain': [0.0, 1.0], 'matches': 'x', 'showticklabels': False, 'visible': False}
    n_ticks = 50
    fig.layout.yaxis = {'anchor': 'x', 'domain': [0.0, 0.7], 'title': {'text': f'<span style="color: {_MAIN_COLOR}">count {name_left}</span>'}, 
                        'tickvals': [i/n_ticks for i in range(1, n_ticks+1)], 
                        'ticktext': [si_format(scale_factors[0]*i/n_ticks, 2) for i in range(1, n_ticks+1)]}
    fig.layout.yaxis2 = {'anchor': 'x2', 'domain': [0.0, 0.7], 'title': {'text': f'<span style="color: {_SECOND_COLOR}">count {name_right}</span>'}, 
                         'side': 'right', 'overlaying': 'y', 'matches': 'y',
                         'tickvals': [i/n_ticks for i in range(1, n_ticks+1)], 
                         'ticktext': [si_format(scale_factors[1]*i/n_ticks, 2) for i in range(1, n_ticks+1)]
                        }
    fig.layout.yaxis3 = {'anchor': 'x3',
                'domain': [0.72, 0.86],
                'matches': 'y3',
                'visible': False,
                'showticklabels': False}
    fig.layout.yaxis4 = {'anchor': 'x4',
                'domain': [0.86, 1.0],
                'matches': 'y4',
                'visible': False,
                'showticklabels': False
    }
    
    fig.update_layout(
        title=f'Compare {col} distribution of {name_left} and {name_right} dataset',
        width=800,
        height=650,
    )
    
    if return_raw:
        return df_merge_num_style, fig
    
    return describe_utils.CustomHStack([describe_utils.render_html(df_merge_num_style), fig])