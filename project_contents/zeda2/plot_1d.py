import pandas as _pd
import plotly.graph_objects as _go

import plotly.express as _px

_px.defaults.template = 'simple_white'

import matplotlib.pyplot as _plt
import seaborn as _sns

# See more: https://matplotlib.org/stable/tutorials/introductory/customizing.html
_STYLE_1D_HISTPLOT = {
    'figure.figsize': (5, 4),

    'axes.facecolor': 'white',
    'axes.edgecolor': 'grey',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 'medium',
    'axes.labelcolor': 'grey',
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'major',

    'grid.color': 'lightgray',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.75,

    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'xtick.minor.size': 0,
    'ytick.minor.size': 0,
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'xtick.color': 'grey',
    'ytick.color': 'grey'
}


def sns_plot_1d_numeric(df, column_name, figsize=(6, 4), title="", xlabel=None, ylabel=None, style=None, box_plot=False, **kwargs):
    if not style:
        style = _STYLE_1D_HISTPLOT
    # Default plot settings
    plot_settings = {
        "stat": "probability",
        "bins": 25,
        "kde": True,
        "alpha": 0.6,
        "color": "xkcd:azure",
    }
    plot_settings.update(kwargs)
    if box_plot:
        fig, (ax_box, ax_hist) = _plt.subplots(nrows=2, sharex=True, figsize=figsize,
                                               gridspec_kw={"height_ratios": (.15, .85)})
        with _plt.style.context(style):
            _sns.boxplot(x=df[column_name], ax=ax_box, color=plot_settings['color'])
            _sns.histplot(df[column_name], ax=ax_hist, **plot_settings)
            ax_box.set(yticks=[])
            if xlabel:
                ax_box.xlabel = xlabel
            if ylabel:
                ax_box.ylabel = ylabel
            return fig
    with _plt.style.context(style):
        _, ax = _plt.subplots(1, 1, figsize=figsize, dpi=100)
        ax = _sns.histplot(df[column_name], ax=ax, **plot_settings)
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel=xlabel)
        if ylabel:
            ax.set_ylabel(ylabel=ylabel)
        return ax


def sns_plot_1d_numeric_lite(df, column_name, figsize=(32, 5)):
    fig, ax = _plt.subplots(1, 2, figsize=figsize)
    _sns.kdeplot(x=df[column_name], common_norm=False, ax=ax[0])
    _sns.histplot(x=df[column_name], discrete=True, ax=ax[1])
    _plt.show()


def matplotlib_plot_1d_hist_norm(x, xTitle, title):
    """
    Plot Historgram using matplotlib
    """
    _plt.figure(figsize=(10, 8))
    _plt.hist(x, bins=30, label=title)
    _plt.ylabel('Count')
    _plt.xlabel(xTitle)
    _plt.title("Histogram Count")
    _plt.show()

    _plt.figure(figsize=(10, 8))
    _plt.hist(x, bins=30, density=True)
    _sns.kdeplot(x)
    _plt.ylabel('Probability')
    _plt.xlabel(xTitle)
    _plt.title("Histogram Probability")
    _plt.show()


def matplotlib_plot_1d_bar_chart_with_orders(df_input, column_name, order_labels):
    """
        Plot bar chart with orders input label values
        Arguments:
            df_input: data frame has column need plot
            column_name: name of column
            order_labels: array label of column need order
    """
    _plt.figure(figsize=(10, 8))
    _sns.set(style='darkgrid')
    ax = _sns.countplot(x=column_name, data=df_input, order=order_labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    _plt.tight_layout()
    _plt.show()


def plotly_plot_1d_line_time_series(df, column_name):
    """
        Plot time series data to chart line
        Arguments:
            df: data frame has column need plot
            column_name: name of time series column
    """
    column_datetime_format = df[column_name].dt.strftime('%Y/%m')
    time_series = _pd.DataFrame(column_datetime_format.value_counts().reset_index())
    time_series.columns = ['date', 'count']
    time_series = time_series.sort_values('date', ascending=True)
    fig = _px.line(time_series, x="date", y="count", title='Count by ' + column_name)
    fig.show()


def plotly_plot_1d_heatmap_bytime(df, column_name, xaxis_type="dayofweek", yaxis_type="hour"):
    """
    Visualize heatmap by time
    Contributor: TuHV
    Parameters:
        - df: DataFrame need to visualize
        - column_name: Name of time column need to visualize
        - xaxis_type: type of x_axis summarize includes: hour, dayofweek, date, day, month, year
        - yaxis_type: type of y_axis summarize includes: hour, dayofweek, date, day, month, year
    Returns:
        - Visualize a heatmap of number of log by time
    """
    summary_df = df.groupby([getattr(df[column_name].dt, yaxis_type),
                             getattr(df[column_name].dt, xaxis_type)])[column_name].count().rename_axis(
        index=[xaxis_type, yaxis_type]).unstack(level=1)
    if xaxis_type == "dayofweek":
        x_axis = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]
    else:
        x_axis = summary_df.columns.values
    y_axis = summary_df.index.values
    z = summary_df.values

    fig = _go.Figure(data=_go.Heatmap(
        z=z,
        x=x_axis,
        y=y_axis,
        colorscale='Blues'))

    fig.update_layout(
        title='Describe ' + yaxis_type + " per " + xaxis_type + " of " + column_name,
        yaxis=dict(
            title=yaxis_type,
            tickmode='linear'),
        xaxis=dict(
            title=xaxis_type,
            tickmode='linear'),
        width=800,
        height=1000)

    fig.show()