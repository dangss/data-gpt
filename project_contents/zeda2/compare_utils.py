import pandas as _pd


def describe_overview(col):
    df_describe = _pd.DataFrame([], columns=[col.name])
    df_describe.loc['dtype'] = col.dtype
    df_describe.loc['count'] = col.size
    df_describe.loc['distinct'] = col.nunique()
    df_describe.loc['distinct (%)'] = df_describe.loc['distinct'] / df_describe.loc['count']
    mode_series = col.mode()
    if mode_series.size > 0:
        df_describe.loc['mode']=mode_series[0]
    else:
        df_describe.loc['mode']=None
    mode_val = df_describe.loc['mode', col.name]
    mode_count = (col==mode_val).sum()
    df_describe.loc['mode count'] = mode_count
    df_describe.loc['mode (%)'] = df_describe.loc['mode count'] / df_describe.loc['count']
    df_describe.loc['null'] = col.isnull().sum()
    df_describe.loc['null (%)'] = df_describe.loc['null'] / df_describe.loc['count']
    df_describe.loc['empty'] = (col=='').sum()
    df_describe.loc['empty (%)'] = df_describe.loc['empty'] / df_describe.loc['count']
    if _pd.api.types.is_numeric_dtype(col):
        df_describe.loc['zero'] = (col==0).sum()
        df_describe.loc['zero (%)'] = df_describe.loc['zero'] / df_describe.loc['count']
        df_describe.loc['negative'] = (col<0).sum()
        df_describe.loc['negative (%)'] = df_describe.loc['negative'] / df_describe.loc['count']
    return df_describe

def describe_percentiles(col, percentiles=None):
    if not percentiles:
        percentiles = [i * 0.1 for i in range(1, 10)] + [0.01, 0.05, 0.95, 0.99]
    quartiles = [0.25, 0.75]
    percentile_df = col.describe(percentiles=percentiles+quartiles).to_frame().rename(columns={col.name: "value"})
    return percentile_df

def compare_cate_table(col1, col2):
    list_stat = ['dtype', 'count', 'distinct', 'distinct (%)', 'mode', 'mode count', 'mode (%)', 'null', 'null (%)', 'empty', 'empty (%)']
    df_describe1 = describe_overview(col1).loc[list_stat]
    df_describe2 = describe_overview(col2).loc[list_stat]
    df_describe = df_describe1.merge(df_describe2, left_index=True, right_index=True)
    # common and distinct categories
    unique_cate1 = set(col1.dropna().unique())
    unique_cate2 = set(col2.dropna().unique())
    common_value = len(unique_cate1.intersection(unique_cate2))
    distinct_value1 = len(unique_cate1.difference(unique_cate2))
    distinct_value2 = len(unique_cate2.difference(unique_cate1))
    df_describe.loc['common cate'] = common_value
    df_describe.loc['distinct cate', col1.name] = distinct_value1
    df_describe.loc['distinct cate', col2.name] = distinct_value2
    return df_describe

def util_compare_cc_arrange_cate(df_describe1, df_describe2):
    list_cate1 = df_describe1.index
    list_cate2 = df_describe2.index
    in1_not_in2 = [cate for cate in list_cate1 if cate not in list_cate2]
    in2_not_in1 = [cate for cate in list_cate2 if cate not in list_cate1]
    new_idx1 = df_describe1.index
    new_idx2 = df_describe2.index
    if in1_not_in2 and not in2_not_in1:
        in_both = [cate for cate in list_cate1 if cate in list_cate2]
        new_idx1 = in1_not_in2+in_both
    elif in2_not_in1 and in1_not_in2:
        in_both = [cate for cate in list_cate2 if cate in list_cate1]
        new_idx2 = in2_not_in1+in_both
    else:
        in_both = [cate for cate in list_cate1 if cate in list_cate2]
        new_idx1 = in1_not_in2+in_both
        new_idx2 = in2_not_in1+in_both
    df_describe1 = df_describe1.loc[new_idx1, :]
    df_describe2 = df_describe2.loc[new_idx2, :]
    return df_describe1, df_describe2


def util_compare_cc_bring_html_null_to_end(df, null_value='Null'):
    null_idx = None
    for idx in df.index:
        if null_value in idx:
            null_idx = idx
    new_idx = []
    if null_idx:
        new_idx = [idx for idx in df.index if idx!=null_idx]
        new_idx += [null_idx]
    else:
        new_idx = df.index
    return df.loc[new_idx, :]


def util_compare_get_processed_names(name1, name2, 
                               alternative_name1, 
                               alternative_name2, 
                               prefix1, 
                               prefix2, 
                               suffix1, 
                               suffix2):
    # name processing
    prefix1 = prefix1 or ''
    prefix2 = prefix2 or ''
    if name1 == name2 and not alternative_name1 and not alternative_name2:
        suffix1 = suffix1 or '1'
        suffix2 = suffix2 or '2'
    else:
        suffix1 = suffix1 or ''
        suffix2 = suffix2 or ''
    # preprocess name
    new_name1 = alternative_name1 or f'{prefix1}_{name1}_{suffix1}'.strip('_')
    new_name2 = alternative_name2 or f'{prefix2}_{name2}_{suffix2}'.strip('_')
    # double check the name
    if new_name1 == new_name2:
        new_name1 = f'{new_name1}_1'
        new_name2 = f'{new_name2}_2'
    return new_name1, new_name2