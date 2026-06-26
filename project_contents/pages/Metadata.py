import json
import re
import streamlit as st
import pandas as pd

from common.app_utils import *
from common.chart_utils import *
from common.display_utils import *

page_prepare("Metadata Catalog", initial_sidebar_state='collapsed')
st.title("Metadata Catalog")
with st.container(border=True):
    if 'full_databases' not in st.session_state:
        st.session_state.full_databases = AppData()
    full_databases = st.session_state.full_databases
    db_path = full_databases.dataset_path
    db_index = full_databases.database_options.index(st.session_state['db_name']) if 'db_name' in st.session_state else 0
    db_name = st.selectbox("Choose a database:", options=full_databases.database_options, index=db_index)

st.header(f'DATABASE: {db_name}')
# get datasets
if 'db_name' not in st.session_state:
    full_databases.load_database(db_name=db_name)

st.caption(f'{full_databases.current_database["db_description"]}')
shown_tables = {table_name: None for table_name in full_databases.current_database['tables'].keys()}

# information_search = st.text_input(label="Quick search", placeholder="Type something to search in database")
# if information_search:
#     with st.spinner("Searching database"):
#         relevant_tables = full_databases.search_information(information=information_search, top_table=8, threshold=0.15,
#                                                             index_type='ip', apply_column_reranking=False, excludes=[])
#
#     if len(relevant_tables) > 0:
#         st.success(f'Here are the tables relevant to "{information_search}"')
#         shown_tables = {row['table']: row['relevant_information'] for idx, row in relevant_tables.iterrows()}
# else:
#     shown_tables = {table_name: None for table_name in full_databases.current_database['tables'].keys()}

if shown_tables is None:
    st.warning('No relevant information')
else:
    with st.container(border=True):
        chosen_table_names = st.multiselect('Table to show', options=list(full_databases.current_database['tables'].keys()), default=list(shown_tables.keys()))

    for table_name in chosen_table_names:
        table_info = full_databases.current_database['tables'][table_name]
        table_description = table_info["description"]
        st.subheader(table_name)

        if shown_tables[table_name] is not None:
            show_columns = shown_tables[table_name][:10]
        else:
            show_columns = table_info["column_description"].keys()

        with st.expander('Column information'):
            for column in show_columns:
                st.markdown(f'<b>{column}</b>', unsafe_allow_html=True)
                st.text(table_info['column_description'].get(column, 'No information').replace('\n', '  \n'))

        with st.expander(f'Show data of {table_name}'):
            with st.container(border=True):
                st.markdown(f'<b>Table Description:</b><br> <p style="text-align: justify;">{table_description}', unsafe_allow_html=True)
                st.markdown(f'<b>Primary Keys:</b>', unsafe_allow_html=True)
                st.write(list(table_info['key'].keys()))

                st.divider()
                if table_info['path'] != '':
                    data = read_path(table_info['path'])
                    data_tabs = None
                    if len(data.keys()) > 1:
                        data_tabs = st.tabs(list(data.keys()))

                    if data_tabs is not None:
                        for idx, dta in enumerate(data.values()):
                            with data_tabs[idx]:
                                st.markdown(f"<b>Rows:</b> {dta.shape[0]}; <b>Columns:</b> {dta.shape[1]}", unsafe_allow_html=True)
                                st.divider()
                                st.markdown(f'<b>Data report:</b>', unsafe_allow_html=True)
                                data_report = get_report(dta) # only get overview
                                show_report(data_report)
                    else:
                        for idx, dta in enumerate(data.values()):
                            st.markdown(f"<b>Rows:</b> {dta.shape[0]}; <b>Columns:</b> {dta.shape[1]}", unsafe_allow_html=True)
                            st.divider()
                            st.markdown(f'<b>Data report:</b>', unsafe_allow_html=True)
                            data_report = get_report(dta)  # only get overview
                            show_report(data_report)

        st.divider()