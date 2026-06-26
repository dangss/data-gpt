import os
from typing import Any, List, Dict
import pandas as _pd
from pandas.io.formats.style import Styler as _Styler
from zeda2.describe_utils import render_html as _render_html, get_plotly_fig_json as _get_plotly_fig_json, df_to_json as _df_to_json
from datetime import datetime as _datetime
import random as _random
import plotly.graph_objs as _go
from zeda2.table_utils import EdaTable as _EdaTable
from copy import deepcopy as _deepcopy
import json as _json
import os as _os
import pathlib as _pathlib
from jinja2 import Environment as _Environment, FileSystemLoader as _FileSystemLoader


class BaseBlock:
    def __init__(self) -> None:
        pass

    
    def to_json(self):
        raise Exception('to_json has not been implemented.')
    

    def gen_id(self, prefix):
        now_str = _datetime.now().strftime('%Y%m%dT%H%M%S%f')
        rand_number = _random.randint(0, 10000)
        self.id = f'{prefix}_{now_str}_{rand_number}'
        return self.id
    

    def to_html(self, save_path=None, return_string=False):
        '''
        Export report to HTML file.

        Parameters
        ----------
        save_path: str, optional
            Path to save the resulting HTML file. If not provided, the report is saved to the current working directory under filename `autoeda_report.html`.
        
        return_string: bool, default False, optional
            Whether to return the HTML file as a string and delete file.

        Return
        ----------
        str if return_string=True and None otherwise
        '''
        return _save_report(
            data_json=self.to_json(),
            save_path=save_path,
            return_string=return_string)
    

class BlockHtml(BaseBlock):
    def __init__(self, obj: Any):
        self.obj = obj
        self.id = self.gen_id('blockhtml')
    

    def to_json(self):
        
        if isinstance(self.obj, _Styler):
            self.instance = _render_html(self.obj)
        elif isinstance(self.obj, _pd.DataFrame):
            # preprocess layout before rendering
            inline_props = [
                ('overflow', 'hidden'),
                ('white-space', 'nowrap')]
            
            self.instance = _render_html(
                self.obj
                .style
                .set_table_styles([dict(selector='th.col_heading', props=inline_props)]))
        elif isinstance(self.obj, str):
            self.instance = self.obj
        else:
            raise Exception('Invalid datatype for class ZHtmlBlock. Valid types are (str, pandas.DataFrame, pandas.io.formats.style.Styler)')
        
        return {
			'id': self.id,
			'type': 'html',
			'data': self.instance}


class BlockPlotlyChart(BaseBlock):
    def __init__(self, plotly_fig: _go.Figure):
        self.plotly_fig = plotly_fig
        self.id = self.gen_id('blockplotlychart')
        
    
    def to_json(self):
        if isinstance(self.plotly_fig, _go.Figure):
            self.instance = _get_plotly_fig_json(self.plotly_fig)
        else:
            raise Exception('Invalid datatype for class BlockPlotlyChart. Valid types are (plotly.graph_objs.Figure)')
        return {
			'id': self.id,
			'type': 'plotly_chart',
			'data': self.instance}


class BlockTable(BaseBlock):
    def __init__(self, table: _EdaTable):
        self.table = table
        self.id = self.gen_id('blocktable')
        
    
    def to_json(self):
        if isinstance(self.table, _EdaTable):
            transformed_df = self.table.get_dataframe()
            self.instance = _df_to_json(transformed_df)
        else:
            raise Exception('Invalid datatype for class BlockTable. Valid types are (zeda2.table_utils.EdaTable')
        return {
			'id': self.id,
			'type': 'table',
			'data': self.instance}
    

class BlockWordcloud(BaseBlock):
    def __init__(self, series: Any):
        self.series = series
        self.id = self.gen_id('blockwordcloud')
        
    
    def to_json(self):
        if isinstance(self.series, _pd.Series):
            self.instance = [{'text': str(index), 'size': value} for index, value in self.series.to_dict().items()]
        else:
            raise Exception('Invalid datatype for class BlockWordcloud. Valid types are (zeda2.table_utils.EdaTable')
        return {
			'id': self.id,
			'type': 'wordcloud',
			'data': self.instance}
    

class ZBlock(BaseBlock):
    def __init__(self, obj: Any):
        if isinstance(obj, (str, _pd.DataFrame, _Styler)):
            self.instance = BlockHtml(obj)
        elif isinstance(obj, _go.Figure):
            self.instance = BlockPlotlyChart(obj)
        elif isinstance(obj, _EdaTable):
            self.instance = BlockTable(obj)
        elif isinstance(obj, _pd.Series):
            self.instance = BlockWordcloud(obj)
        else:
            raise Exception(f'Invalid type for `obj` if ZBlock. Type: {type(obj)}')
    

    def to_json(self):
        return self.instance.to_json()
    

class ZStack(BaseBlock):
    def __init__(self, children: List, styles: str = ""):
        self.children = children
        self.styles = styles
        for idx in range(len(self.children)):
            child = self.children[idx]
            if isinstance(child, list):
                self.children[idx] = ZHStack(child)
            elif not isinstance(child, BaseBlock):
                self.children[idx] = ZBlock(child)
        self.id = self.gen_id('zstack')
        
    
    def to_json(self):
        self.data = []
        for child in self.children:
            self.data.append(child.to_json())


class ZVStack(ZStack):
    def __init__(self, children: List, styles: str = ""):
        super(ZVStack, self).__init__(children, styles)
        
    
    def to_json(self):
        super(ZVStack, self).to_json()
        return {
			'id': self.id,
			'type': 'stack',
            'layout': 'column',
			'data': self.data,
            'style': self.styles
            }
    

class ZHStack(ZStack):
    def __init__(self, children: List, styles: str = ""):
        super(ZHStack, self).__init__(children, styles)
        
    
    def to_json(self):
        super(ZHStack, self).to_json()
        return {
			'id': self.id,
			'type': 'stack',
            'layout': 'row',
			'data': self.data,
            'style': self.styles
            }
    

class BaseFilter(BaseBlock):
    def __init__(self, items: Dict, default: str):
        self.items = items
        for key, value in self.items.items():
            if not isinstance(value, BaseBlock):
                if isinstance(value, list):
                    self.items[key] = ZHStack(value)
                else:
                    self.items[key] = ZBlock(value)
        self.default = default


    def to_json(self):
        self.data = []
        self.options = []
        child_data = None
        for text, child in self.items.items():
            # get child json
            if isinstance(child, BaseBlock):
                child_data = child.to_json()
            else:
                child_data = ZBlock(child).to_json()

            # if this child is default then extract its id
            if text == self.default:
                self.default = child_data['id']

            # append to options
            if child_data:
                self.data.append(child_data)
                self.options.append({'text': text, 'id': child_data['id']})
        
        # if default is not provided then choose the first one
        if len(self.options) > 0 and not self.default:
            self.default = self.options[0]['id']

    

class ZSelectDropdown(BaseFilter):
    def __init__(self, items: Dict, default: str = None):
        super(ZSelectDropdown, self).__init__(items, default)
        self.id = self.gen_id('zselect_dropdown')
        
    
    def to_json(self):
        super(ZSelectDropdown, self).to_json()
        return {
			'id': self.id,
			'type': 'select_dropdown',
			'data': self.data,
            'options': self.options,
            'default': self.default
            }
    

class ZSelectRadio(BaseFilter):
    def __init__(self, items: Dict, default: str = None):
        super(ZSelectRadio, self).__init__(items, default)
        self.id = self.gen_id('zselect_ratio')
        
    
    def to_json(self):
        super(ZSelectRadio, self).to_json()
        return {
			'id': self.id,
			'type': 'select_radio',
			'data': self.data,
            'options': self.options,
            'default': self.default
            }
    

class ZTab(BaseFilter):
    def __init__(self, items: Dict, default: str = None):
        super(ZTab, self).__init__(items, default)
        self.id = self.gen_id('ztab')
        
    
    def to_json(self):
        super(ZTab, self).to_json()
        return {
			'id': self.id,
			'type': 'tabs',
			'data': self.data,
            'tabs': self.options,
            'default': self.default
            }
    

class ZTitle(BaseBlock):
    def __init__(self, title: str, level: int = 0):
        self.title = title
        self.level = level
        self.id = self.gen_id('ztitle')
        
    
    def to_json(self):
        return {
			'id': self.id,
            'type': 'html',
			'data': f'<h{self.level+1}>{self.title}</h{self.level+1}>'
            }
    

class ZText(BaseBlock):
    def __init__(self, text: str):
        self.text = text
        self.id = self.gen_id('ztext')
        
    
    def to_json(self):
        return {
			'id': self.id,
            'type': 'html',
			'data': f'<div{self.level}>{self.text}</div>'
            }


def _process_title(data, dict_indices, list_menu):
    if isinstance(data, ZTitle):
        data = _deepcopy(data)
        level = data.level
        if level in dict_indices:
            current_idx = dict_indices[level] + 1
            dict_indices[level] += 1
            
            # reset sub level
            next_level = level + 1
            while dict_indices.get(next_level, None):
                dict_indices[next_level] = -1
                next_level += 1
        else:
            current_idx = 0
            dict_indices[level] = 0
        
        # get all previous parent level's current value
        parent_idx = '.'.join([str(dict_indices[prev_idx]+1) for prev_idx in range(1, level)])

        if level == 0:
            order = f'{chr(current_idx+65)} '
        elif level == 1:
            order = f'{current_idx+1:.0f} '
        else:
            order = f'{parent_idx}.{current_idx+1:.0f} '
        data.title = order + data.title
        list_menu.append({'id': data.id, 'text': data.title, 'level': data.level})

    elif isinstance(data, (ZHStack, ZVStack)):
        data.children = [_process_title(child, dict_indices, list_menu) for child in data.children]
    elif isinstance(data, BaseFilter):
        for key in data.items.keys():
            data.items[key] = _process_title(data.items[key], dict_indices, list_menu)
    return data


class ZReport(BaseBlock):
    def __init__(
            self, 
            data: Any, 
            title: str = 'AutoEDA Report', 
            layout: str = 'h'):
        if isinstance(data, BaseBlock):
            self.data = data
        elif isinstance(data, list):
            self.data = ZHStack(data)
        else:
            self.data = ZBlock(data)
        self.title = title
        self.layout = layout

    
    def to_json(self):
        list_memu = []

        # data of title will be changed, and list menu is appended
        self.data = _process_title(self.data, dict(), list_memu)

        # get json of data
        data_json = self.data.to_json()

        return {
            'title': self.title,
            'data': [data_json],
            'layout': self.layout,
            'menu': list_memu,
            'logo': 'https://res.cloudinary.com/uit-hcm-vn/image/upload/v1675322438/logo_bsc5zt.png',
        }
    
        
def _get_a_path():
    default_path = _os.getcwd() + '/autoeda_report'
    suffix_number = 1
    path = default_path
    while _os.path.exists(path+'.html'):
        path = f'{default_path}_{suffix_number}'
        suffix_number+=1
    return path + '.html'


def _save_report(data_json, save_path=None, return_string=False, report_type='common'):
    if not isinstance(data_json, dict):
        data_json = _json.loads(data_json)
    save_path = save_path or _get_a_path()

    # create parent directory if not exist
    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    
    # folder containing html assets
    root_dir = str(_pathlib.Path(__file__).parent.resolve())
    fs_generate = root_dir+f'/export_assets/{report_type}/templates'

    # load environment
    ENV_LOADER = _Environment(
        loader=_FileSystemLoader(searchpath=fs_generate),
    )
    TEMPLATE_FILE = "index.html"
    template_base = ENV_LOADER.get_template(TEMPLATE_FILE)
    report = template_base.render(context=data_json, path_exists=_os.path.exists)

    # save report
    if return_string:
        return report

        # save report
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f'Report has been saved at {save_path}')
    return save_path