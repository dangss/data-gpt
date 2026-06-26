from zeda2.common_report.base import BaseBlock, ZBlock, ZHStack, ZVStack, ZTab, ZSelectDropdown, ZSelectRadio, ZText, ZTitle, ZReport

def export_html_from_json(data_json):
    import json
    import sys
    import pathlib
    from datetime import datetime
    from jinja2 import Environment, FileSystemLoader
    import os

    def path_exists(path):
        return os.path.exists(path)

    TEMPLATE_PATH = f'{os.getcwd()}/zeda2/common_report/export_assets/common/templates/'
    # load environment
    ENV_LOADER = Environment(
        loader=FileSystemLoader(searchpath=TEMPLATE_PATH),
    )
    TEMPLATE_FILE = "index.html"
    template_base = ENV_LOADER.get_template(TEMPLATE_FILE)
    report = template_base.render(context=data_json, path_exists=path_exists)

    return report

def export_html(component:BaseBlock):
    data_json = component.to_json()
    return export_html_from_json(data_json)