from .utils import *


def autoeda_describe_interest_from_path(path):
    df_interest = pd.read_parquet(path)
    df_interest = (df_interest
        .merge(
            (df_interest.groupby(['cate']).agg({'num_user': 'sum'})
            .rename(columns={'num_user': 'sum_num_user'})
            .reset_index()),
            left_on='cate',
            right_on='cate')
        )
    df_interest['pct'] = df_interest['num_user']/df_interest['sum_num_user']
    df_interest_pivot = df_interest.pivot(index='cate', columns='cls', values=['num_user', 'pct'])
    color_sequence = ['#114b5f', '#1a936f', '#88d498', '#c6dabf', '#f3e9d2', '#8b9890','#6c6c6c','#020f0b']

    fig = px.bar(
        df_interest,
        x='cate',
        y='pct',
        text='pct',
        color='cls',
        color_discrete_sequence=color_sequence,
        category_orders={'cls':['HIGH','MEDIUM','LOW']}
    )

    fig.update_traces(texttemplate="%{text:,.2%}")
    fig.update_layout(width=550,
                    height=500,
                    yaxis=dict(tickformat=',.0%'),
                    legend=dict(xanchor='left',
                                yanchor='top',
                                x=0.78,
                                y=1.28
                                )
                    )
    fmt_ts_0 = tf.FmtThousandSeparator(0, columns=pd.MultiIndex.from_tuples((('num_user', 'HIGH'), 
                                                            ('num_user', 'LOW'), 
                                                            ('num_user', 'MEDIUM'))))
    fmt_percent = tf.FmtPercent(2, columns=pd.MultiIndex.from_tuples((('pct', 'HIGH'), 
                                                            ('pct', 'LOW'), 
                                                            ('pct', 'MEDIUM'))))
    table = Block(df_interest_pivot, formatters=[fmt_ts_0, fmt_percent, fmt_fontsize])
    return VStack([
        Block('<h1 style="margin-bottom:10px">Interest EDA Report</h1>'),
        Block('<h3 style="margin-bottom:10px">User Rating by Categories</h3>'),
        describe_utils.CustomHStack([table, fig], grid_cell_style='margin-right:20px')
    ])