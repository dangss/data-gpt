import plotly.graph_objects as _go
import plotly.express as _px

import matplotlib.pyplot as _plt
import seaborn as _sns
from matplotlib import gridspec as _gridspec

_px.defaults.template = 'simple_white'


def sns_plot_2d_boxplot(df_input, x, y):
    """
        Plot boxplot chart
        Arguments:
            df_input: data frame has column need plot
            x: name of column x
            y: name of column y
    """
    _plt.figure(figsize=(20, 10))
    ax = _sns.boxplot(data=df_input, x=x, y=y)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    _plt.tight_layout()
    _plt.show()


def sns_plot_2d_categories_nested_bar_count_chart(df_input, x, hue):
    """
        Plot 2d nested bar chart by sns
        Arguments:
            df_input: data frame has column need plot
            x: name of column x
            hue: name of color
    """
    _plt.figure(figsize=(10, 8))
    _sns.set(style='darkgrid')
    ax = _sns.countplot(x=x, hue=hue, data=df_input)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    _plt.tight_layout()
    _plt.show()


def sns_plot_2d_histogram(series_1, series_2):
    """
        Plot 2D Historgram using seaborn
    """
    n_size_max = 1000000
    sample_series_1 = series_1
    if len(series_1) > n_size_max:
        sample_series_1 = series_1.sample(n_size_max)
    gs = _gridspec.GridSpec(1, 2)
    fig = _plt.figure(figsize=(25, 8))
    ax1 = _plt.subplot(gs[0, 0])
    ax1 = _plt.hist(sample_series_1, bins=100, density=True)
    ax1 = _sns.kdeplot(sample_series_1)
    ax1 = _plt.ylabel('Count')
    ax1 = _plt.xlabel(sample_series_1.name)
    ax1 = _plt.title("Histogram Probability of " + sample_series_1.name)

    sample_series_2 = series_2
    if len(series_2) > n_size_max:
        sample_series_2 = series_2.sample(n_size_max)
    ax2 = _plt.subplot(gs[0, 1])
    ax2 = _plt.hist(sample_series_2, bins=100, density=True)
    ax2 = _sns.kdeplot(sample_series_2)
    ax2 = _plt.ylabel('Count')
    ax2 = _plt.xlabel(sample_series_2.name)
    ax2 = _plt.title("Histogram Probability of " + sample_series_2.name)

    _plt.show()


def plotly_plot_2d_line_chart_mean(df_input, column1, column2):
    """
        Plot 2d line chart mean data
        Arguments:
            df_input: data frame has column need plot
            column1: x_axis
            column2: y_axis
    """
    df_input = df_input.groupby([column1], as_index=False).mean()
    fig = _px.line(df_input, x=column1, y=column2, title=''.join([column1, ' + ', column2]))
    fig.show()


def plotly_plot_2d_histogram(df_input, x_axis, color, **kwargs):
    """
        Plot 2d histogram chart
        Arguments:
            df_input: data frame has column need plot
            x_axis: x_axis
            color: color
    """
    df_draw = df_input.dropna(subset=[x_axis, color])
    fig = _px.histogram(df_draw, x=x_axis, color=color, barmode='relative', hover_data=df_input.columns)
    fig.update_layout(**kwargs)
    fig.show()


def plotly_plot_2d_bar_chart(df_input, x_axis, color):
    """
        Plot 2d bar chart
        Arguments:
            df_input: data frame has column need plot
            x_axis: x_axis
            color: color
    """
    fig = _px.bar(df_input, x=x_axis, color=color,
                  barmode='relative',
                  hover_data=df_input.columns)
    fig.show()


def plotly_plot_2d_categories_bar_chart(df_input, columnX, columnY, type="group"):
    """
        Plot 2d bar chart category with category
        Arguments:
            df_input: data frame has column need plot
            columnX: x_axis
            columnY: y_axis and color
            type: "group" or "relative"
    """
    barnorm = None
    if type == "group":
        barnorm = 'percent'
    input_df = df_input[(df_input[columnX].notnull()) & (df_input[columnY].notnull())]
    fig = _px.histogram(input_df, x=columnX, y=columnY, color=columnY, histfunc="count", barmode=type, barnorm=barnorm)
    fig.show()


def plotly_plot_2d_category_with_label(df_input, df_groupby, column_name, label_name='label', **kwargs):
    """
        Plot 2d bar chart category with label column
        Arguments:
            df_input: dataframe has column need plot
            df_groupby: dataframe group by get from function describe.describe_2d_category_data_extend()
            column_name: category column
            label_name: label column name
            kwargs: additional keyword args for plotly layout
    """
    names = df_groupby[column_name + ' / ' + label_name]
    data = []
    values = df_input[label_name].dropna().unique()
    for value in values:
        data.append(_go.Bar(name='Label ' + str(value), x=names, y=df_groupby['percentile_hor_' + str(value)]))
    y_values = [df_groupby.loc[len(df_groupby) - 1, 'percentile_hor_' + str(values[0])] for i in range(len(names))]
    data.append(_go.Scatter(x=names, y=y_values))
    fig = _go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode='stack', **kwargs)
    return fig


def sns_plot_2d_numeric_with_label(df, feature_name, label_name="label", figsize=(32, 5)):
    """
        Plot 2d histogram chart and line chart numeric with label column
        Arguments:
            df: dataframe has column need plot
            feature_name: numeric column
            label_name: label column name
            figsize: size of figure
    """
    fig, ax = _plt.subplots(1, 2, figsize=figsize)
    if isinstance(df[label_name], (int, float, complex)):
        label_values = sorted(df[label_name].unique())
    else:
        label_values = df[label_name].unique()
    _sns.kdeplot(x=df[feature_name], hue=df[label_name], hue_order=label_values, common_norm=False, ax=ax[0])
    if len(df[feature_name].unique()) > 1000:
        _sns.histplot(x=df[feature_name], hue=df[label_name], hue_order=label_values, multiple='stack', ax=ax[1],
                      element="step")
    else:
        _sns.histplot(x=df[feature_name], hue=df[label_name], hue_order=label_values, discrete=True, multiple='stack',
                      ax=ax[1])
    return fig, ax


def matplotlib_plot_2d_numeric_hist_compare(series_1, series_2):
    '''Plot 2D Historgram using matplotlib'''
    n_size_max = 1000000
    sample_series_1 = series_1
    if len(series_1) > n_size_max:
        sample_series_1 = series_1.sample(n_size_max)
    gs = _gridspec.GridSpec(1, 2)
    fig = _plt.figure(figsize=(25, 8))
    ax1 = _plt.subplot(gs[0, 0])
    ax1 = _plt.hist(sample_series_1, bins=100, density=True)
    ax1 = _sns.kdeplot(sample_series_1)
    ax1 = _plt.ylabel('Count')
    ax1 = _plt.xlabel(sample_series_1.name)
    ax1 = _plt.title("Histogram Probability of " + sample_series_1.name)

    sample_series_2 = series_2
    if len(series_2) > n_size_max:
        sample_series_2 = series_2.sample(n_size_max)
    ax2 = _plt.subplot(gs[0, 1])
    ax2 = _plt.hist(sample_series_2, bins=100, density=True)
    ax2 = _sns.kdeplot(sample_series_2)
    ax2 = _plt.ylabel('Count')
    ax2 = _plt.xlabel(sample_series_2.name)
    ax2 = _plt.title("Histogram Probability of " + sample_series_2.name)

    _plt.show()
