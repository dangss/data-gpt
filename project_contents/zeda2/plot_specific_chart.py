"""
Modify from https://github.com/wcipriano/pretty-print-confusion-matrix
"""

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import plotly.graph_objs as _go
import seaborn as _sns

import matplotlib.font_manager as _fm
from matplotlib.collections import QuadMesh as _QuadMesh
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.utils.multiclass import unique_labels as _unique_labels

from zeda2.describe_utils import calculate_psi


def plot_corr_features(df, figsize=(16, 6)):
    """
        Contribute by baopng
        Draw correlation heatmap between all columns in dataframe
        Params:
            df: dataframe
            figsize: size of fig draw, default (16,6)
    """
    _plt.figure(figsize=figsize)
    heatmap = _sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12);
    # save heatmap as .png file
    # dpi - sets the resolution of the saved image in dots/inches
    # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    # _plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


def _add_margins(df, margins_name="ALL"):
    """Add row and column margins (subtotals)."""
    if isinstance(margins_name, str):
        margins_name = [margins_name] * 2
    elif isinstance(margins_name, list):
        length = len(margins_name)
        if length == 1:
            margins_name = [margins_name] * 2
        elif length < 1 or length > 2:
            raise ValueError("Length of margins_name is {}. "
                             "Expected length is 1 or 2.".format(length))
    else:
        raise ValueError("margins_name argument must be a string or a list.")

    # Check for name conflicts
    row_name, col_name = margins_name
    if row_name in df.index.values:
        raise ValueError('Index name "{}" already existed.'.format(row_name))
    if col_name in df.columns.values:
        raise ValueError('Column name "{}" already existed.'.format(col_name))

    # Compute subtotal row and column
    sum_rows = _np.sum(df, axis=0)
    sum_cols = _np.sum(df, axis=1)
    grand_total = _np.sum(sum_rows)

    result = df.copy()
    result[col_name] = sum_cols
    sum_rows = _np.append(sum_rows, grand_total)
    result.loc[row_name] = sum_rows
    return result


def _config_cell_properties(data, x, y, position, text, fontsize, fmt,
                            facecolors, show_null_values=False):
    text_add = []
    text_del = []

    cell_value = data[y][x]
    total = data[-1][-1]
    cell_percentage = (float(cell_value) / total) * 100

    current_column = data[:, y]
    column_length = len(current_column)

    # Config for last row and/or last column
    if (x == column_length - 1) or (y == column_length - 1):
        if cell_value != 0:
            if (x == column_length - 1) and (y == column_length - 1):
                total_right = _np.sum(data.diagonal()[:-1])
            elif x == column_length - 1:
                total_right = data[y][y]
            elif y == column_length - 1:
                total_right = data[x][x]
            right_percentage = (float(total_right) / cell_value) * 100
            wrong_percentage = 100 - right_percentage
        else:
            right_percentage = wrong_percentage = 0

        # Delete old text
        text_del.append(text)

        # Add new text
        list_text = [
            format(cell_value, ",d"),
            format(right_percentage, fmt) + "%",
            format(wrong_percentage, fmt) + "%"
        ]

        text_properties = dict(
            color="w", ha="center", va="center", gid="sum",
            fontproperties=_fm.FontProperties(weight="bold", size=fontsize)
        )
        list_text_properties = [text_properties.copy() for i in range(3)]
        list_text_properties[1]["color"] = "xkcd:vibrant green"
        list_text_properties[2]["color"] = "red"

        text_x, text_y = text.get_position()
        list_positions = [
            (text_x, text_y - 0.3),
            (text_x, text_y),
            (text_x, text_y + 0.3)
        ]

        for i, text in enumerate(list_text):
            new_text = dict(
                x=list_positions[i][0], y=list_positions[i][1],
                text=list_text[i], kw=list_text_properties[i]
            )
            text_add.append(new_text)

        # Set background color for cells at the margin
        background_color = [0.27, 0.30, 0.27, 0.85]  # RGBA
        if (x == column_length - 1) and (y == column_length - 1):
            background_color = [0.17, 0.20, 0.17, 0.85]
        facecolors[position] = background_color

    else:
        if cell_percentage > 0:
            new_text = format(cell_value, ",d") + \
                       "\n" + format(cell_percentage, fmt) + "%"
        elif show_null_values:
            new_text = "0\n" + format(0, fmt)
        else:
            new_text = ""
        text.set_text(new_text)

        # Main diagonal
        if (x == y):
            # Set color of the text in the diagonal cells
            text.set_color("w")
            # Set background color for the diagonal cells
            facecolors[position] = [0.35, 0.8, 0.55, 1.0]
        else:
            text.set_color("r")

    return text_add, text_del


def compute_psi_heatmap(df, col, subset, na_replace=None):
    """
    Contributed by thanhnm3.
    :param df: input dataframe
    :param col: column to calculate psi
    :param subset: calculate psi by subset column
    :param na_replace: value to fillna
    """
    def highlight_psi(v):
        if v < 0.1:
            props = "color: green"
        elif v < 0.2:
            props = "color: orange"
        else:
            props = "color: red"
        return props

    unique_groups = sorted(df[subset].unique())
    result_df = _pd.DataFrame(_np.nan, index=unique_groups, columns=unique_groups)
    tooltips_df = _pd.DataFrame("", index=unique_groups, columns=unique_groups)

    for src_idx in range(len(unique_groups)):
        for des_idx in range(src_idx+1, len(unique_groups)):
            src_group = unique_groups[src_idx]
            des_group = unique_groups[des_idx]
            src_series = df[df[subset] == src_group][col]
            des_series = df[df[subset] == des_group][col]

            if na_replace:
                src_series = src_series.fillna(na_replace)
                des_series = des_series.fillna(na_replace)
            else:
                src_series = src_series.dropna()
                des_series = des_series.dropna()

            psi = calculate_psi(src_series, des_series, buckettype="quantiles", buckets=10)
            result_df.at[src_group, des_group] = psi
            result_df.at[des_group, src_group] = psi

            tooltip_value = f"{len(src_series)} - {len(des_series)}"
            tooltips_df.at[src_group, des_group] = tooltip_value
            tooltips_df.at[des_group, src_group] = tooltip_value

    return (
        result_df
            .style
            .format("{0:.3f}", na_rep="")
            .applymap(highlight_psi)
            .set_tooltips(
            tooltips_df,
            props=[
                ("visibility", "hidden"),
                ("position", "absolute"),
                ("background-color", "lightgrey"),
                ("color", "black"),
                ("transform", "translate(-20px, -20px)")
            ]
        )
    )


def plot_precomputed_confusion_matrix(
    cm, title="Confusion Matrix", figsize=(10, 10),
    axis_labels=("Predicted", "Actual"), predict_axis="x",
    fontsize=10, fmt=",.2f", cmap="Oranges", cbar=False,
    linewidths=0.5, show_null_values=False, ax=None):
    """Plot confusion matrix from precomputed data.

    Parameters
    ----------
    cm : ndarray, Pandas DataFrame
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is
        is provided, the index/column information will be used to label the
        columns and rows.

    title : str, optional, default: "Confusion Matrix"
        Text to use for the title.

    figsize : (float, float), optional, default: (10, 10)
        Figure width and height in inches.

    axis_labels : (str, str), optional, default: ("Predicted", "Actual")
        A tuple of explicit labels for the x-axis and y-axis.

    predict_axis : {"x", "X", "y", "Y"}, optional, default: "x"
        Whether to use x-axis or y-axis as predict axis.

    fontsize : {size in points, "xx-small", "x-small", "small", "medium", \
"large", "x-large", "xx-large"}, optional, default: 10

    fmt : str, optional, default: ",.2f"
        String formatting code to use when adding percentage annotations.

    cmap : matplotlib colormap name/object, optional, default: "Oranges"
        The mapping from data values to color space.

    cbar : boolean, optional, default: False
        Whether to draw a colorbar.

    linewidths : float, optional, default: 0.5
        Width of the lines that will divide each cell.

    show_null_values : boolean, optional, default: False
        Whether to annotate cell with null values.

    ax : matplotlib Axes, optional, default: None
        Axes in which to draw the plot, otherwise create a new Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the confusion matrix.
    """
    if predict_axis in ("x", "X"):
        x_label, y_label = axis_labels
        margins_name = ["PRECISION", "RECALL"]
    elif predict_axis in ("y", "Y"):
        cm = cm.T
        y_label, x_label = axis_labels
        margins_name = ["RECALL", "PRECISION"]
    else:
        raise ValueError('predict_axis argument must be "x" or "y"')

    if ax is None:
        fig = _plt.figure(title, figsize=figsize)
        ax = fig.gca()  # get current axis
        ax.cla()  # clear existing plot

    # Insert summary row and column (subtotals)
    cm = _add_margins(cm, margins_name=margins_name)

    # Plot confusion matrix
    ax = _sns.heatmap(cm, annot=True, annot_kws={"size": fontsize}, fmt=fmt,
                      square=True, cbar=cbar, cmap=cmap,
                      linecolor="w", linewidths=linewidths, ax=ax)

    # Turn of all the ticks
    ax.tick_params(bottom=False, top=False, left=False, right=False)

    # Set labels rotation
    ax.tick_params(axis='x', rotation=90, labelsize=fontsize + 2)
    ax.tick_params(axis='y', rotation=0, labelsize=fontsize + 2)

    # Face color list
    quadmesh = ax.findobj(_QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # Text annotation
    cm = cm.to_numpy()
    text_add = []
    text_del = []
    position = -1  # from left to right, bottom to top

    for text in ax.collections[0].axes.texts:
        pos = text.get_position() - _np.array([0.5, 0.5])
        x, y = pos.astype(int)
        position += 1

        # This will modify `text` and `facecolors` inplace
        text_to_add, text_to_delete = \
            _config_cell_properties(cm, x=x, y=y, position=position,
                                    text=text, fontsize=fontsize, fmt=fmt,
                                    facecolors=facecolors,
                                    show_null_values=show_null_values)
        text_add.extend(text_to_add)
        text_del.extend(text_to_delete)

    # Remove old text and add new text
    for text in text_del:
        text.remove()
    for text in text_add:
        ax.text(text["x"], text["y"], text["text"], **text["kw"])

    # Set title and legends
    ax.set_title(title, fontsize=fontsize + 7, pad=20)
    ax.set_xlabel(x_label, fontsize=fontsize + 5)
    ax.set_ylabel(y_label, fontsize=fontsize + 5)
    return ax


def plot_confusion_matrix(y_true, y_pred, **kwargs):
    """Plot confusion matrix.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    title : str, optional, default: "Confusion Matrix"
        Text to use for the title.

    figsize : (float, float), optional, default: (10, 10)
        Figure width and height in inches.

    axis_labels : (str, str), optional, default: ("Predicted", "Actual")
        A tuple of explicit labels for the x-axis and y-axis.

    predict_axis : {"x", "X", "y", "Y"}, optional, default: "x"
        Whether to use x-axis or y-axis as predict axis.

    fontsize : {size in points, "xx-small", "x-small", "small", "medium", \
"large", "x-large", "xx-large"}, optional, default: 10

    fmt : str, optional, default: ",.2f"
        String formatting code to use when adding percentage annotations.

    cmap : matplotlib colormap name/object, optional, default: "Oranges"
        The mapping from data values to color space.

    cbar : boolean, optional, default: False
        Whether to draw a colorbar.

    linewidths : float, optional, default: 0.5
        Width of the lines that will divide each cell.

    show_null_values : boolean, optional, default: False
        Whether to annotate cell with null values.

    ax : matplotlib Axes, optional, default: None
        Axes in which to draw the plot, otherwise create a new Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the confusion matrix.
    """
    classes = _unique_labels(y_true, y_pred)
    cm = _pd.DataFrame(_confusion_matrix(y_true, y_pred),
                       columns=classes, index=classes)
    return plot_precomputed_confusion_matrix(cm, **kwargs)


def plot_sankey(df, cate_cols, value_col=None, title="Sankey"):
    def to_source_target_df(df, cate_cols, value_col=None):
        df = df.copy()
        if not value_col:
            value_col = "count"
            df = df.groupby(list(cate_cols)).size().to_frame("count").reset_index()

        st_df = None
        for i in range(len(cate_cols) - 1):
            temp_df = df[[cate_cols[i], cate_cols[i + 1], value_col]]
            temp_df.columns = ["source", "target", "count"]
            st_df = _pd.concat([st_df, temp_df])
            st_df = st_df.groupby(["source", "target"]).agg({"count": "sum"}).reset_index()
        return st_df

    color_palete = ["#4B8BBE", "#306998", "#FFE873", "#FFD43B", "#646464"]
    label_list = []
    color_num_list = []

    # Define colors based on number of levels
    for c in cate_cols:
        labels = list(set(df[c].values))
        color_num_list.append(len(labels))
        label_list = label_list + labels
    label_list = list(set(label_list))  # remove duplicates

    #     color_list = []
    #     for idx, color_num in enumerate(color_num_list):
    #         color_list = color_list + [color_palete[idx]]*color_num

    source_target_df = to_source_target_df(df, cate_cols, value_col)

    # Add index for source-target pair
    source_target_df["source_id"] = source_target_df["source"].apply(lambda x: label_list.index(x))
    source_target_df["target_id"] = source_target_df["target"].apply(lambda x: label_list.index(x))

    fig = _go.Figure(data=[
        _go.Sankey(
            node=dict(
                label=label_list,
                #                 color=color_list,  # DISABLE CUSTOM COLOR
                pad=15,
                thickness=20,
                line=dict(
                    color="black",
                    width=0.5,
                ),
            ),
            link=dict(
                source=source_target_df["source_id"],
                target=source_target_df["target_id"],
                value=source_target_df["count"]
            )
        )
    ])

    fig.update_layout(
        title=title,
        font=dict(size=10)
    )

    return fig
