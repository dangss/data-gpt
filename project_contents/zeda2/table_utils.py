import numbers
import pybloqs.block.table_formatters as tf
from pybloqs.block import colors as colors
import re

_NULL_COL_NAME = "Null"
_NULL_COLOR = "#D62728"
_MAIN_COLOR = "#1F77B4"
CSS_BOLD = 'font-weight:bold'
CSS_BACKGROUND_COLOR = 'background-color:'
CSS_COLOR = 'color:'


class TableFormatter(object):
    def __init__(self, rows=None, columns=None, apply_to_index=False, apply_to_header=False):
            """Initialise formatter and specify which rows and columns it is applied to. Default None applies to all.
            boolean or 2-tuple of booleans can be supplied to apply_to_header_and_index.
            """
            self.rows = rows
            self.columns = columns
            self.apply_to_index = apply_to_index
            self.apply_to_header = apply_to_header
            return self
    
    def validate_params(self, df):
        if self.columns is None:
            self.columns = df.columns
        if self.rows is None:
            self.rows = list(df.index)
    
    def modify_dataframe(self, df):
        raise NotImplementedError('format_dataframe')
  

class FmtPercent(TableFormatter):
    def __init__(self, n_decimals, columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtPercent, self).__init__(rows, columns, apply_to_index, apply_to_header)
        self.n_decimals = n_decimals
        return
    
    def modify_dataframe(self, df):
        self.validate_params(df)
        for col in self.columns:
            try:
                df[col].astype('double')
                fmt_string = f'{{:.{self.n_decimals}%}}'
                df.loc[self.rows, col] = df.loc[self.rows, col].apply(lambda x: fmt_string.format(x))
            except:
                pass
        return df


class FmtFormat(TableFormatter):
    def __init__(self, fmt, columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtFormat, self).__init__(rows, columns, apply_to_index, apply_to_header)
        self.fmt = fmt
        return
    

    def modify_dataframe(self, df):
        self.validate_params(df)
        for col in self.columns:
            try:
                df[col].astype('double')
                fmt_string = f'{{:{self.fmt}}}'
                df.loc[self.rows, col] = df.loc[self.rows, col].apply(lambda x: fmt_string.format(x))
            except:
                pass
        return df


class FmtDecimals(TableFormatter):
    def __init__(self, n_decimals, columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtDecimals, self).__init__(rows, columns, apply_to_index, apply_to_header)
        self.n_decimals = n_decimals
        return
    
    def modify_dataframe(self, df):
        self.validate_params(df)
        for col in self.columns:
            try:
                df[col].astype('double')
                fmt_string = f'{{:.{self.n_decimals}f}}'
                df.loc[self.rows, col] = df.loc[self.rows, col].apply(lambda x: fmt_string.format(x))
            except:
                pass
        return df


class FmtThousandSeperator(TableFormatter):
    def __init__(self, columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtThousandSeperator, self).__init__(rows, columns, apply_to_index, apply_to_header)
        return
    
    def modify_dataframe(self, df):
        self.validate_params(df)
        for col in self.columns:
            try:
                df[col].astype('double')
                fmt_string = f'{{:,.0f}}'
                df.loc[self.rows, col] = df.loc[self.rows, col].apply(lambda x: fmt_string.format(x))
            except:
                pass
        return df


class FmtPercentHtml(TableFormatter):
    def __init__(self, n_decimals, red_null=False, null_value=None, main_color=_MAIN_COLOR, columns=None, rows=None, apply_to_index=False, apply_to_header=False, vmin=0, vmax=100):
        super(FmtPercentHtml, self).__init__(rows, columns, apply_to_index, apply_to_header)
        self.n_decimals = n_decimals
        self.red_null = red_null
        self.main_color = main_color
        self.null_value = null_value
        # multiply 100 in percent system
        self.vmin = vmin
        self.vmax = vmax
        return
    
    def modify_dataframe(self, df):
        self.validate_params(df)
        numeric_cols = []
        for col in self.columns:
            try:
                df.loc[self.rows, col].astype('double')
                numeric_cols.append(col)
            except:
                pass
        df.loc[self.rows, numeric_cols] = df.loc[self.rows, numeric_cols].apply(lambda x: get_html_percent(x, self.rows, self.n_decimals, self.red_null, self.main_color, self.null_value, self.vmin, self.vmax), axis=1)
        return df


class FmtBold(TableFormatter):
    def __init__(self, columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtBold, self).__init__(rows, columns, apply_to_index, apply_to_header)
        return
    
    def modify_dataframe(self, df):
        self.validate_params(df)
        for col in self.columns:
            df.loc[self.rows, col] = df.loc[self.rows, col].apply(lambda x: f"<b>{x}</b>")
        return df


class FmtRed(TableFormatter):
    def __init__(self, columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtRed, self).__init__(rows, columns, apply_to_index, apply_to_header)
        return
    
    def modify_dataframe(self, df):
        self.validate_params(df)
        df[self.columns] = df[self.columns].astype(object)
        for col in self.columns:
            df.loc[self.rows, col] = df.loc[self.rows, col].apply(lambda x: f"<div style='color:red'>{x}</div>" if 'div' not in str(x) else x)
        if self.apply_to_index is not None:
            original_index = df.index
            new_index = []
            for idx in original_index:
                if idx in self.rows:
                    new_index.append(f"<div style='color:red'>{idx}</div>")
                else:
                    new_index.append(idx)
            df.index = new_index
        return df
        

class FmtAppendTotalRow(TableFormatter):

    def __init__(self, row_name='Total', columns=None, rows=None, apply_to_index=False, apply_to_header=False):
        super(FmtAppendTotalRow, self).__init__(rows, columns, apply_to_index, apply_to_header)
        self.row_name = row_name
        return


    def modify_dataframe(self, df):
        """Add row to dataframe, containing numbers aggregated with self.operator."""
        last_row = []
        for col in df.columns:
            try:
                last_row.append(df[col].astype('double').sum())
            except:
                last_row.append('')
        df.loc[self.row_name] = last_row
        return df


def get_html_percent(row, selected_rows, n_decimals, red_null, main_color, null_value, vmin=0, vmax=100):
    if row.name == 'Total':
        return ['{:.2%}'.format(element) for element in row]
    elif row.name in selected_rows:
        if red_null and row.name==null_value:
            color = _NULL_COLOR
        else:
            color = main_color
        new_row = row.apply(lambda x: get_html_percent_content(x, color, n_decimals, vmin=vmin, vmax=vmax))
        return new_row
    else:
        return row.astype(object)
        

def get_html_percent_content(percent, color, n_decimals, vmin=0, vmax=100):
    percent = round(percent*100, n_decimals)
    if color == _NULL_COLOR:
        text_color = _NULL_COLOR
    else:
        text_color = "#000000"
    scaled_percent = ((percent-vmin)/(vmax-vmin))*100
    if scaled_percent > 45:
        return f"<div style='width:{scaled_percent+8}%;color:#FFFFFF;background:{color};border-radius:3px;text-align:center;'>{percent:.2f}%</div>"
    else:
        return f"""<div style='display:flex;flex-direction:row'><div style='width:{scaled_percent+5}%;height:100%;color:{color};background:{color};border-radius:3px;text-align:center;'><div><span style="padding-left:5px"></span></div></div><div style='margin-left:3px;color:{text_color}'>{percent:.2f}%</div></div>"""


class EdaTable(object):
    def __init__(self, df, formatters):
        self.df = df
        self.formatters = formatters
        if isinstance(formatters, list):
            self.formatters = formatters
        else:
            self.formatters = [formatters]


    def get_dataframe(self):
        self.formatted_df = self.df.copy()
        # add class total_row for html export purpose
        has_total_row = False
        for formatter in self.formatters:
            if type(formatter) == FmtAppendTotalRow:
                has_total_row = True
            try:
                self.formatted_df = formatter.modify_dataframe(self.formatted_df)
            except NotImplementedError:
                raise NotImplementedError('format_dataframe')
        self.formatted_df.index = [f"<b>{idx}</b>" for idx in self.formatted_df.index]
        if has_total_row and self.formatted_df.shape[0] > 0:
            self.formatted_df.iloc[-1] = self.formatted_df.iloc[-1].apply(lambda x: f'<div class="total_row">{x}</div>')
            current_index = list(self.formatted_df.index)
            current_index[-1] = f'<div class="total_row">{current_index[-1]}</div>'
            self.formatted_df.index = current_index
        return self.formatted_df


class FmtAppendTotalsRowCustom(tf.TableFormatter):
    """Add another row at table bottom containing sum/mean/etc. of specified columns"""

    def __init__(self, row_name='Total', bold=True, background_color=colors.LIGHT_GREY,
                 font_color=None, total_columns=None, hline_color=colors.DARK_BLUE, hline_style='1px solid', fmt={}):
        self.row_name = row_name
        # Operate on all columns: Set self.columns to None
        super(FmtAppendTotalsRowCustom, self).__init__([row_name], None)

        if total_columns is None:
            total_columns = []
        self.total_columns = total_columns
        self.bold = bold
        self.background_color = background_color
        self.font_color = font_color
        self.hline_color = hline_color
        self.hline_style = hline_style
        self.fmt = fmt
        return
    
    def p2f(self, x):
        return float(x.strip('%'))/100
    
    def _get_number_from_string(self, s):
        if s is None or s != s:
            return s
        try:
            s+1
            return s
        except:
            pass
        if 'div' in s:
            pattern = r'.*>([0-9,%.]+)<.*'
            matches = re.search(pattern, s)
            if matches:
                s = matches.groups(1)[0]
            else:
                return None
        if '%' in s:
            return self.p2f(s)
        if s.replace(',', '').replace('.', '').isnumeric():
            return float(s.replace(',', ''))
        return None


    def _modify_dataframe(self, df):
        """Add row to dataframe, containing numbers aggregated with self.operator."""
        last_row = []
        for col in df.columns:
            fmt_col = self.fmt.get(col, '.2f')
            series_numeric = df[col].apply(lambda x: self._get_number_from_string(x))
            if fmt_col == 'html_percent':
                fmt_col = '.0%'
            sum_val = series_numeric.sum()
            if sum_val is None:
                last_row.append('')
            last_row.append(f'{{:{fmt_col}}}'.format(sum_val))

        df.loc[self.row_name] = last_row
        return df

    def _create_cell_level_css(self, data):
        """Set fontsize for cell as CSS format."""
        if data.row_name != self.row_name:
            return None

        css_substrings = []
        if self.bold is not None:
            css_substrings.append(CSS_BOLD)
        if self.background_color is not None:
            css_substrings.append(CSS_BACKGROUND_COLOR + colors.css_color(self.background_color))
        if self.font_color is not None:
            css_substrings.append(CSS_COLOR + colors.css_color(self.font_color))
        if self.hline_color is not None:
            css_substrings.append('border-top:' + self.hline_style + ' ' + colors.css_color(self.hline_color))
        if len(css_substrings) != 0:
            return "; ".join(css_substrings)
        else:
            return None


class FmtFormatTotalRow(tf.TableFormatter):
    """Add another row at table bottom containing sum/mean/etc. of specified columns"""

    def __init__(self, rows=['Total'], bold=True, background_color=colors.LIGHT_GREY,
                 font_color=None, total_columns=None, hline_color=colors.DARK_BLUE, hline_style='1px solid'):
        # Operate on all columns: Set self.columns to None
        super(FmtFormatTotalRow, self).__init__(rows, None)

        if total_columns is None:
            total_columns = []
        self.total_columns = total_columns
        self.bold = bold
        self.background_color = background_color
        self.font_color = font_color
        self.hline_color = hline_color
        self.hline_style = hline_style
        return


    def _create_cell_level_css(self, data):
        """Set fontsize for cell as CSS format."""
        if data.row_name not in self.rows:
            return None
        css_substrings = []
        if self.bold is not None:
            css_substrings.append(CSS_BOLD)
        if self.background_color is not None:
            css_substrings.append(CSS_BACKGROUND_COLOR + colors.css_color(self.background_color))
        if self.font_color is not None:
            css_substrings.append(CSS_COLOR + colors.css_color(self.font_color))
        if self.hline_color is not None:
            css_substrings.append('border-top:' + self.hline_style + ' ' + colors.css_color(self.hline_color))
        if len(css_substrings) != 0:
            return "; ".join(css_substrings)
        else:
            return None


class FmtVerticalLine(tf.TableFormatter):
    def __init__(self, columns=None, rows=None, vline_color=colors.BLACK, vline_style='1px dotted'):
        self.columns = columns
        # Operate on all columns: Set self.columns to None
        super(FmtVerticalLine, self).__init__(rows, columns)
        self.vline_color = vline_color
        self.vline_style = vline_style
        return
    
    def _create_cell_level_css(self, data):
        if data.column_name not in self.columns:
            return None
        css_substrings = []
        if self.vline_color is not None:
            css_substrings.append('border-left:' + self.vline_style + ' ' + colors.css_color(self.vline_color))
        if len(css_substrings) != 0:
            return "; ".join(css_substrings)
        else:
            return None