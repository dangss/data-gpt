from IPython.display import display_html as _display_html
from IPython.core.display import display as _display
from IPython.core.display import HTML as _HTML
from IPython.display import Markdown as _Markdown
import pandas as _pd

_pd.options.display.float_format = "{:.2f}".format
_pd.options.display.max_rows = 500
# _pd.set_option('max_columns', 50)


class _color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def display_side_by_side(*args):
    """
        Display pandas dataframe side by side
        args: pass all pandas dataframes you want to display
    """
    html_str = ''
    for df in args:
        html_str += df.to_html()
    _display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


def display_side_by_side_extend(dfs: list, captions: list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0" * 20
    _display(_HTML(output))


def display_df_with_title(df, title):
    """
       df : pandas dataframe
       title : title
    """
    output = df.style.set_table_attributes("style='font-size:100%'")._repr_html_()
    output = '<b>{}</b>'.format(title) + output
    _display(_HTML(output))


def display_markdown(text):
    """
       text : input text to show markdown
    """
    _display(_Markdown(text))


def display_full_df(df):
    """
        Show full dataframe (Warning: not input too big dataframe)
       df : pandas dataframe
    """
    _pd.set_option('display.max_rows', None)
    _pd.set_option('display.max_columns', None)
    _pd.set_option('display.width', 2000)
    _pd.set_option('display.float_format', '{:20,.2f}'.format)
    _pd.set_option('display.max_colwidth', None)
    _display(df)
    _pd.reset_option('display.max_rows')
    _pd.reset_option('display.max_columns')
    _pd.reset_option('display.width')
    _pd.reset_option('display.float_format')
    _pd.reset_option('display.max_colwidth')


def set_width_container(width: int):
    _display(_HTML(f"<style>.container {{ width:{width}% !important; }}</style>"))
