import copy
import json
import os
import glob
import time
import unidecode
from datetime import datetime
import streamlit as st
from stqdm import stqdm
import pandas as pd
import numpy as np
import ast
import faiss
from langchain.llms import OpenAI
from textwrap import dedent

from dotenv import load_dotenv
load_dotenv("env")

# print("OPENAI_API_KEY: ", os.getenv("OPENAI_API_KEY"))
# print("OPENAI_API_BASE: ", os.getenv("OPENAI_API_BASE"))
# print("HELICONE_API_KEY: ", os.getenv("HELICONE_API_KEY"))

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

from langchain.embeddings import OpenAIEmbeddings
from helpers.constants import DATABASE_NAME_LOGS_CHAT
from models.auth import Auth
from models.mongo_helper import MongoHelper
from agents.agent_utils import question_to_search_infos

llm_common = OpenAI(model_name='gpt-3.5-turbo-1106', temperature=0, top_p=0.001, seed=212)

dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(dir_path)
icon_file_path = os.path.join(dir_path, "img", "icon.png")
css_file_path = os.path.join(dir_path, "style.css")

## Common Error ##
db_not_exist = lambda db_name, all_db: KeyError(f"Database {db_name} does not exist. Available databases are: {all_db}")
table_not_exist = lambda table_name, all_tables: KeyError(f"Table {table_name} does not exist. Available tables are: {all_tables}")
db_none = ValueError(f"Database name must be specified, now it is None.")
table_none = ValueError(f"Table name must be specified, now it is None.")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

## Page config ##
def page_prepare(page_title="Data Insights", initial_sidebar_state="auto"):
    st.set_page_config(
        page_title=page_title,
        page_icon=icon_file_path,
        layout="wide",
        initial_sidebar_state=initial_sidebar_state,
    )
    st.set_option("deprecation.showPyplotGlobalUse", False)

    ### EMBED STYLE
    # Get the absolute path to the style.css file
    with open(css_file_path, "r") as css_file:
        css_styles = css_file.read()

    st.markdown(css_styles, unsafe_allow_html=True)
    # components.html(css_styles)

    ### TITLE
    st.sidebar.markdown('<div class="title-header">DI</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="caption-header">From <b>D</b>ata to <b>I</b>nsights</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
def st_warning(text):
    return st.warning(text, icon="⚠️")

## General data ##
def get_openai_embeddings(documents):
    """Get embeddings for a list of texts using OpenAI's API."""
    embeddings = openai.Embedding.create(
        input=documents,
        engine="text-embedding-3-large"  # Choose an appropriate engine for your task
    )
    # Extract the embeddings from the response
    return np.array([embedding['embedding'] for embedding in embeddings['data']])
def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def read_parquet_dir(path, from_date=None, to_date=None):
    spark = st.session_state.spark

    if from_date is not None and to_date is not None:
        date_pths = [os.path.join(path, d.strftime('%Y/%m/%d')) for d in pd.date_range(start=from_date, end=to_date)]
    else:
        date_pths = [path]

    full_df = None
    for pth in date_pths:
        data = spark.read.option("recursiveFileLookup", "true").parquet(pth)
        full_df = data if full_df is None else full_df.unionByName(data, allowMissingColumns=True)

    return full_df.toPandas()

def read_path(data_path, from_date=None, to_date=None):
    all_data = dict()
    if isinstance(data_path, str):
        data_name = data_path.split('/')[-1]
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
            all_data[data_name.replace('.csv', '')] = data.copy()
        elif data_path.endswith(".parquet"):
            data = pd.read_parquet(data_path)
            all_data[data_name.replace('.parquet', '')] = data.copy()
        else:
            data = read_parquet_dir(data_path, from_date=from_date, to_date=to_date)
            all_data[data_name] = data.copy()

    elif isinstance(data_path, dict):
        all_data = dict()
        for duration, pth in data_path.items():
            all_data[duration] = read_parquet_dir(pth)

    return all_data

class AppData:
    def __init__(self):
        self.db_info = json.load(open(os.path.join(dir_path, "database_info.json"), "r"))
        self.current_database = None
        self.dataset_path = os.path.join(project_path, "public_datasets")
        self.database_options = list(sorted(self.db_info.keys()))
        self.db_info = json.load(open(os.path.join(dir_path, "database_info.json"), "r"))
        self.DURATION = ['d00', 'd01', 'd07', 'd30']
        self.use_dt_folder = False

    def load_database(self, db_name: str = None):
        if db_name is None:
            raise db_none

        if db_name not in self.db_info.keys():
            raise db_not_exist(db_name, list(self.db_info.keys()))

        ## TODO: standardize the file system: duration/table_name/time_range
        if db_name == '00_Mekong_Delta':
            ## get all filepath in /db_name
            ## TODO: this is only work with parquet files
            table_files = glob.glob(
                os.path.join(self.dataset_path, db_name, "**/*.parquet"), recursive=True
            )

            ## get filename from file path
            table_path = {
                unidecode.unidecode(table_file.split("/")[-1].replace(".parquet", "")): table_file
                for table_file in table_files
            }

            ## assign path to table
            for table_name in self.db_info[db_name]['tables'].keys():
                if unidecode.unidecode(table_name) not in table_path.keys():
                    print(table_name)
                else:
                    self.db_info[db_name]['tables'][table_name]["path"] = table_path[unidecode.unidecode(table_name)]

        elif db_name == '01_RBT':
            self.use_dt_folder = True
            for table_name in self.db_info[db_name]['tables'].keys():
                self.db_info[db_name]['tables'][table_name]["path"] = dict()
                for duration in self.DURATION:
                    file_path = os.path.join(self.dataset_path, db_name, duration, table_name)
                    if os.path.exists(file_path):
                        self.db_info[db_name]['tables'][table_name]["path"][duration] = file_path

        self.current_database = self.db_info[db_name]
#         if self.use_dt_folder:
#             self.current_database['processing_notes'] = self.current_database.get('processing_notes', '') + '''
# While processing, you must consider the following information: for a specific duration, data at a given row is calculated and aggregated for x days leading up to the current date.
# For instance:
# - Duration 'd01': For a row dated April 11, the information aggregated pertains to that specific day alone, April 11.
# - Duration 'd07': For a row dated April 11, the information aggregated covers the preceding 7 days leading up to April 11, ranging from April 4 to April 11.
# - Duration 'd30': For a row dated April 11, the information aggregated spans the previous 30 days leading up to April 11, from March 12 to April 11.'''

    def load_data_into_table(self, table_name: str = None):
        if table_name is None:
            raise table_none

        if table_name not in self.current_database['tables'].keys():
            raise table_not_exist(table_name, list(self.current_database['tables'].keys()))

        if 'data' not in self.current_database['tables'][table_name].keys():
            data_path = self.current_database['tables'][table_name]['path']
            if data_path.endswith('.parquet'):
                self.current_database['tables'][table_name]['data'] = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                self.current_database['tables'][table_name]['data'] = pd.read_csv(data_path)
            else:
                raise NotImplementedError("This file format is not currently supported. Only '.csv' and '.parquet' files are supported.")

    def load_table(self, table_name: str = None):
        if table_name is None:
            raise table_none

        if table_name not in self.current_database['tables'].keys():
            raise table_not_exist(table_name, list(self.current_database['tables'].keys()))

        return self.current_database['tables'][table_name]

    def search_in_table(self, table_name: str = None, information=None, search_on:str="column_description", threshold: float = None, index_type:str='l2'):
        if table_name is None:
            raise table_none

        if information is None:
            raise ValueError("Information search must be specified, now it is None.")

        if isinstance(information, str): # if information is a text
            information = get_openai_embeddings(information) # convert to embeddings

        if table_name not in self.current_database['tables'].keys():
            raise table_not_exist(table_name, list(self.current_database['tables'].keys()))

        # document to search (columns)
        documents = copy.deepcopy(self.current_database['tables'][table_name][search_on])
        # documents.update({'table_description': self.current_database['tables'][table_name]['description']})
        # documents.update(self.current_database['tables'][table_name]['key'])

        nb_documents = len(documents)
        search_embedding = {
            "column_description": "vector_store" if "vector_store" in self.current_database['tables'][table_name].keys() else "column_embedding",
            "processing_logic": "logic_embedding"
        }
        # if 'vector_store' not in self.current_database['tables'][table_name].keys():
        #     self.current_database['tables'][table_name]['vector_store'] = get_openai_embeddings(list(documents.values()))

        vector_store = self.current_database['tables'][table_name][search_embedding[search_on]].copy()[:len(documents)]
        if isinstance(vector_store, list):
            vector_store = np.array(vector_store)

        vector_store = normalize_l2(vector_store)
        information = normalize_l2(information)

        dimension = vector_store.shape[1]
        if index_type == 'l2':
            index = faiss.IndexFlatL2((dimension))
        else:
            index = faiss.IndexFlatIP((dimension))
        index.add(vector_store)

        scores, indices = index.search(information, k=nb_documents)

        df_result = pd.DataFrame(zip(scores[0], indices[0]), columns=['score', 'indices'])
        if threshold is not None:
            if index_type == 'l2': # distance
                df_result = df_result[df_result['score'] <= threshold]
            else: # similarity
                df_result = df_result[df_result['score'] >= threshold]
        if isinstance(documents, dict):
            df_document = pd.DataFrame(zip(range(nb_documents), documents.keys(), documents.values()), columns=['indices', 'column', 'content'])
        elif isinstance(documents, list):
            df_document = pd.DataFrame(zip(range(nb_documents)), documents, columns=['indices', 'logic_code'])
        df_result = pd.merge(df_result, df_document, on='indices').drop('indices', axis=1)
        df_result['table'] = table_name
        return df_result

    def rerank_selected_tables(self, selected_tables:pd.DataFrame=None, user_question:str=""):
        current_database = self.current_database
        if selected_tables is None:
            raise ValueError("Parameter 'selected_tables' cannot be empty.")

        if not isinstance(selected_tables, pd.DataFrame):
            raise TypeError(f"'selected_tables' must be a Pandas dataframe, now its type is {type(selected_tables)}")

        PROMPT = dedent("""
        << TASK DESCRIPTION >>
        A list of tables is shown below. Next to each table is a name, its primary keys (marked by "+"), and columns relevant to the question provided.
        Each table includes attributes corresponding to its primary key. 
        For example:
        - A table with the primary key [X, Y] contains information for each Y for each X.
        - A table with the primary key [X, Y, Z] contains information for each Z within each Y for each X.

        Based on the question, return a list of tables re-ranked according to their relevance to the question. In case there is no relevant tables, honestly return [].

        << FORMAT >>
        Following strictly this format:
        TABLE_LIST: <List of tables, their primary keys, and some relevant columns>
        Question: <User's question>
        Final Answer: <list of reranked tables>

        << EXAMPLE>>
        Here are some typical examples:
        1. 
        TABLE_LIST:
        ---
        Table "buget_revenue_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        + "loại_ngân_sách": "budget type of revenue"
        Relevant Columns: ['budget_revenue', 'percentage_budget_revenue']
        ---
        ---
        Table "budget_expenditure_table":
        Primary Key:
        + "năm": "year",
        + "tỉnh": "province"
        + "loại_ngân_sách": "budget type of expenditure"
        Relevant Columns: ['budget_expense', 'percentage_budget_expense']
        ---
        ---
        Table "province_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        Relevant Columns: ['budget_revenue', 'budget_expense', 'GDP', 'budget_revenue_from_insurance']
        ---
        Question: "Total state budget revenue of province An Giang"
        Final Answer: ["province_table", "budget_revenue_table", "budget_expenditure_table"]

        2. 
        TABLE_LIST:
        ---
        Table "buget_revenue_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        + "loại_ngân_sách": "budget type of revenue"
        Relevant Columns: ['budget_revenue', 'percentage_budget_revenue']
        ---
        ---
        Table "budget_expenditure_table":
        Primary Key:
        + "năm": "year",
        + "tỉnh": "province"
        + "loại_ngân_sách": "budget type of expenditure"
        Relevant Columns: ['budget_expense', 'percentage_budget_expense']
        ---
        ---
        Table "province_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        Relevant Columns: ['budget_revenue', 'budget_expense', 'GDP', 'budget_revenue_from_insurance']
        ---
        Question: "Total state budget revenue by each budget type"
        Final Answer: ["budget_revenue_table", "province_table", "budget_expenditure_table"]

        3. 
        TABLE_LIST:
        ---
        Table "buget_revenue_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        + "loại_ngân_sách": "budget type of revenue"
        Relevant Columns: ['budget_revenue', 'percentage_budget_revenue']
        ---
        ---
        Table "budget_expenditure_table":
        Primary Key:
        + "năm": "year",
        + "tỉnh": "province"
        + "loại_ngân_sách": "budget type of expenditure"
        Relevant Columns: ['budget_expense', 'percentage_budget_expense']
        ---
        ---
        Table "province_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        Relevant Columns: ['budget_revenue', 'budget_expense', 'GDP', 'budget_revenue_from_insurance']
        ---
        ---
        Table "district_ward_table":
        Primary Key:
        + "năm": "year"
        + "tỉnh": "province"
        + "huyện": "district/ward"
        Relevant Columns: ['budget_revenue', 'budget_expense', 'GDP', 'budget_revenue_from_insurance']
        ---
        Question: "budget revenue changes of each province"
        Final Answer: ["province_table", "district_ward_table", "budget_revenue_table", "budget_expenditure_table"]

        << YOUR TURN >>
        Let's try this now:
        TABLE_LIST:
        {table_list}
        Question: "{user_question}"
        """)

        table_list = ""
        original_tables = selected_tables['table'].tolist().copy()
        for idx, table in selected_tables.iterrows():
            table_list += f"---\nTable \"{table['table']}\":\nPrimary Key:\n"
            for key, des in current_database['tables'][table['table']]['key'].items():
                key_des = f"+ '{key}': '{des}'\n"
                table_list += key_des
            table_list += f"Relevant Columns: {table['column']}\n---\n"

        prompt_to_use = PROMPT.format(table_list=table_list, user_question=user_question)
        response = llm_common(prompt_to_use)
        try:
            reranked_tables = ast.literal_eval(response.split("Final Answer:")[-1].strip())
            reranked_tables = reranked_tables + [table for table in selected_tables['table'] if table not in reranked_tables]
            return [table for table in reranked_tables if table in original_tables]

        except Exception as err:
            print(err)
            return selected_tables['table'].tolist()
    def rerank_tables_windowsize(self, selected_tables: pd.DataFrame, user_question: str = "", window_size: int = 5, loops: int = 1, step:int=4):
        origin_tables = selected_tables[['table']]
        origin_tables['column'] = selected_tables['relevant_information'].map(lambda x: x[:5])

        if len(origin_tables) > window_size:
            loop = 0
            while loop < loops:
                for i in stqdm(range(len(origin_tables), 0, -step)):
                    start_idx = max(0, i - window_size)
                    end_idx = i
                    to_rerank_tables = origin_tables.iloc[start_idx:end_idx]
                    reranked_tables = self.rerank_selected_tables(to_rerank_tables, user_question)
                    origin_tables.iloc[start_idx:end_idx, 0] = reranked_tables.copy()
                loop += 1
        else:
            reranked_tables = self.rerank_selected_tables(origin_tables, user_question)
            origin_tables.iloc[:, 0] = reranked_tables.copy()

        return origin_tables.merge(selected_tables, on='table', how='left')

    def get_relevant_columns(self, question: str, table: str, column_list: list):
        def split_list(input_list, size=5):
            return [input_list[i:i + size] for i in range(0, len(input_list), size)]

        PROMPT = dedent("""
        <<TASK DESCRIPTION>>
        A list of columns is shown below. Next to each column is the its description. A question is also provided.

        Based on the question, return a list of column index corresponding to the relevant columns that are similar or useful for the answer of the question.

        <<FORMAT>>
        Using this following format:
        '''
        Column list: <list of provided columns>
        Question: <given question>
        Final answer: 
        RELEVANT_COLUMNS: [<list column indexes that presents exactly the information>]

        <<EXAMPLE>>
        Here is some typical examples:
        1/
        '''
        Column list:
        ---
            1. 'nang_suat_lao_dong': '''productivity of all provinces'''
        ---
        Question: 'Provided the productivity per capita of Long An in 2018'
        Final answer:
        RELEVANT_COLUMNS: [1]
        '''

        2/
        '''
        Column list:
        ---
            1. 'number_of_marriage_urban': '''number of marriage in urban area'''
            2. 'count_marriage_total': '''total count of marriage in all provinces'''
            3. 'number_of_first_marriage': '''number of marriage who get first married'''
            4. 'married_average_age': '''the average age at which the citizens get married'''
        ---
        Question: 'Show me number of marriage of each province over years'
        Final answer:
        RELEVANT_COLUMNS: [2, 1, 3]
        '''

        3/
        '''
        Column list:
        ---
            1. 'number_of_marriage_urban': '''number of marriage in urban area'''
            2. 'count_marriage_total': '''total count of marriage in all provinces'''
            3. 'number_of_first_marriage': '''number of marriage who get first married'''
            4. 'married_average_age': '''the average age at which the citizens get married'''
            5. 'total_budget': '''the total budget of each province'''
        ---
        Question: 'How does GDP changes in period 2018 - 2022?'
        Final answer:
        RELEVANT_COLUMNS: []
        '''

        <<YOUR TURN>>
        Column list:
        ---
        {columns}
        ---
        {hints}
        Question: "{question}"
        Final answer:
        """)

        column_sublist = split_list(column_list, 5)
        relevant_columns = []
        for sublist in column_sublist:
            columns = ""
            for idx, column in enumerate(sublist):
                columns += f"\n{idx + 1}. '{column}': '''{self.current_database['tables'][table]['column_description'].get(column, self.current_database['tables'][table]['key'].get(column, ''))}'''\n"
            hints = ""
            if 'processing_logic' in self.current_database['tables'][table].keys():
                processing_logic = '\n- '.join(self.current_database['tables'][table]['processing_logic'])
                hints = f"Hints:\n---\n{processing_logic}\n---\n"
            prompt_to_use = PROMPT.format(question=question, columns=columns, hints=hints)

            response = llm_common(prompt_to_use)
            thought, results = response.split("RELEVANT_COLUMNS:")
            results = [int(x) for x in ast.literal_eval(results.strip())]
            relevant_columns += [sublist[idx - 1] for idx in results].copy()

        new_relevant_columns = []
        for col in relevant_columns:
            if col not in new_relevant_columns: new_relevant_columns.append(col)
        return new_relevant_columns

    def search_information(self, information:str = None, top_table:int=None, threshold=None, index_type='l2', apply_column_reranking=True):
        if information is None or information == "":
            raise ValueError('Information search must not be empty.')

        if self.current_database is None:
            raise AttributeError("Database is not loaded.")

        full_result = None
        try:
            information_embedding = get_openai_embeddings([information])
        except Exception as err:
            with st.spinner('Service is overloaded. Please wait'):
                time.sleep(5)
            information_embedding = get_openai_embeddings([information])

        for table_name in stqdm(self.current_database['tables'].keys()):
            result = self.search_in_table(table_name=table_name, information=information_embedding, threshold=threshold, index_type=index_type)
            if full_result is None:
                full_result = result.copy()
            else:
                full_result = pd.concat([full_result, result], axis=0)

        full_result = full_result.sort_values(['table', 'score'], ascending=index_type=='l2')
        # exlucde the table description
        relevant_tables = full_result.groupby('table').agg(
            score=('score', 'min' if index_type == 'l2' else 'max'),
            relevant_information=('column', list),
            relevant_scores=('score', list)
        ).sort_values('score', ascending=index_type=='l2').reset_index()

        if top_table is not None:
            relevant_tables = relevant_tables.iloc[:top_table]

        final_tables = relevant_tables
        if len(relevant_tables) > 3:
            with st.spinner('Reranking tables'):
                reranked_relevant_tables = self.rerank_tables_windowsize(relevant_tables, user_question=information)
            # get the score 0.45 above the others to avoid wrong reranking
            reranked_relevant_tables = pd.concat([
                reranked_relevant_tables[reranked_relevant_tables['score'] >= 0.45],
                reranked_relevant_tables[reranked_relevant_tables['score'] < 0.45]
            ], axis=0)
            final_tables = reranked_relevant_tables

        # reranking columns
        final_tables['high_similar_columns'] = [[] for _ in range(len(final_tables))]
        if apply_column_reranking:
            nb_table_rerank = min(4, len(final_tables))
            with st.spinner(f'Reranking columns on top {nb_table_rerank} tables:'):
                for idx_table in stqdm(range(nb_table_rerank)):
                    table_row = final_tables.iloc[idx_table:idx_table+1]
                    try:
                        table_row['high_similar_columns'] = table_row.apply(lambda x: self.get_relevant_columns(question=information, table=x['table'], column_list=x['relevant_information'][:30]), axis=1)
                        final_tables.update(table_row)
                    except:
                        continue

        return final_tables

    def search_question(self, question:str=None, split_info:str=True):
        if not split_info:
            full_information = {'target': question, 'factor': ''}
        else:
            full_information = question_to_search_infos(question=question)
            if full_information is None:
                full_information = {'target': question, 'factor': ''}
            else:
                full_information = {'target': '; '.join(full_information['target']), 'factor': '; '.join(full_information['factor'])}

        full_tables = pd.DataFrame()
        for information_type in ['target', 'factor']:
            information = full_information[information_type]
            if information == '': continue
            with st.spinner(f'Searching {information}'):
                relevant_tables = self.search_information(information=information, top_table=8, threshold=0.15, index_type='ip')
                relevant_tables['information_type'] = information_type
                relevant_tables['information'] = information
                full_tables = relevant_tables.copy() if full_tables is None else pd.concat([full_tables, relevant_tables], axis=0)
        return full_tables

## Table of content ##
class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = st.sidebar

    def remove_items(self):
        self._items = []

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2")

    def subheader(self, text):
        self._markdown(text, "h3")

    def generate(self, title='Conversation'):
        if len(self._items) > 0:
            with self._placeholder:
                st.subheader(title)
                item_str = "\n".join([f'{item}' for item in self._items])
                st.markdown(f"<ul style='list-style-type: none;'>{item_str}</ul>", unsafe_allow_html=True)
                st.divider()
    def add_item(self, text:str, status:str):
        """ Add an item with its status: 'done', 'in progress', 'pending' """
        key_text = text.replace(' ', '-').lower()
        if status == 'done':
            self._items.append(f"<li style='margin: 0;'>✔&emsp;<a href='#{key_text}'>{text}</a></li>")
        elif status == 'in progress':
            self._items.append(f"<li style='margin: 0;'>╰┈➤&emsp;<b>{text}</b>")
        else:
            self._items.append(f"<li style='margin: 0;'>▪&emsp;{text}</span></li>")

    def _markdown(self, text, level):
        key_text = text.replace(' ', '-').lower()
        st.markdown(f"<{level} id='{key_text}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"<a href='#{key_text}'>{text}</a>")

## Mongo DB Action ##
def get_current_user():
    auth_model = Auth()
    access_token = st.query_params["accessToken"]
    current_user = auth_model.get_current_user_from_token(
        token=access_token
    )

    return current_user

## Feedback & Log ##
def save_conversation(question:str, part:str):
    global_conversation = json.load(open(os.getcwd()))

def is_exist_in_conversation(order_number, list_history):
    for history in list_history:
        if history['order'] == order_number:
            return True
    return False
def save_history(username, conversation_id, messages, curr_session_id="test"):
    ## get user info
    db = MongoHelper()
    current_user = get_current_user()

    if conversation_id not in db.get_list_collection_in_database(
        database_name=DATABASE_NAME_LOGS_CHAT
    ):
        db.create_collection_in_database(
            database_name=DATABASE_NAME_LOGS_CHAT, collection_name=conversation_id
        )

    ## Save history ##
    curr_timestamp = datetime.now().timestamp()
    ## save message ##
    history_file_name = f"history_{curr_session_id}.csv"
    for idx_message, message in enumerate(messages):
        msg_str = get_message_string(message)
        list_history_in_conversation_id = db.get_documents_in_collection(collection_name=conversation_id, database_name=DATABASE_NAME_LOGS_CHAT)
        if not is_exist_in_conversation(idx_message, list_history_in_conversation_id):
            db.create_document_in_collection(
                collection_name=conversation_id,
                database_name=DATABASE_NAME_LOGS_CHAT,
                param={
                    "username": current_user["username"],
                    "role": msg_str.split("|")[0],
                    "order": idx_message,
                    "content": msg_str.split("|")[1],
                },
            )
def get_message_string(message):
    role = message.role
    message_content = message.content

    messages = []
    if isinstance(message_content, str):
        content_string = ["", "", message_content]
        messages.append(content_string)
    else:
        for msg in message_content:
            section_type = msg.type
            tool_name = msg.tool_name
            message_string = msg.content
            ## history of planner agents
            if isinstance(message_string, str):
                content_string = [tool_name, section_type, message_string]
                messages.append(content_string)
            else:
                ## history of expert agents
                sub_content = message_string.content
                for cnt in sub_content:
                    section_type = cnt.type
                    message_string = cnt.content
                    content_string = [tool_name, section_type, message_string]
                    messages.append(content_string)

    return f"""{role}|{messages}""".replace("\n", "\\n")
def save_feedback(feedbacks, conversation_id, curr_session_id="test"):
    ## Save feedback ##
    feedback_file = f"feedback_{curr_session_id}.json"
    curr_timestamp = datetime.now().timestamp()
    feedbacks["created_time"] = curr_timestamp
    feedback_data = json.dumps([feedbacks])

    current_user = get_current_user()
    db = MongoHelper()
    if current_user is None:
        return

    data_feedback = db.get_document_in_collection_by_unique_value(
        database_name=DATABASE_NAME_LOGS_CHAT,
        collection_name="feedbacks-chat",
        param={"conversation_id": conversation_id},
    )

    if feedbacks[conversation_id][1]["Overall Result"]["score"] == 1:
        data_append = {
            "order": feedbacks[conversation_id][0]["Order Number"],
            "user_question": feedbacks[conversation_id][0]["User question"],
            "score": feedbacks[conversation_id][1]["Overall Result"]["score"],
            "comment": feedbacks[conversation_id][1]["Overall Result"]["text"],
        }

    else:
        data_append = {
            "order": feedbacks[conversation_id][0]["Order Number"],
            "user_question": feedbacks[conversation_id][0]["User question"],
            "score": feedbacks[conversation_id][1]["Overall Result"]["score"],
            "comment": feedbacks[conversation_id][1]["Overall Result"]["text"],
            "metrics": [
                {
                    "type": "accuracy",
                    "score": feedbacks[conversation_id][2]["accuracy"]["score"],
                    "steps": feedbacks[conversation_id][2]["accuracy"]["steps"],
                },
                {
                    "type": "completeness",
                    "score": feedbacks[conversation_id][3]["completeness"]["score"],
                    "text": feedbacks[conversation_id][3]["completeness"]["text"],
                },
                {
                    "type": "conciseness",
                    "score": feedbacks[conversation_id][4]["conciseness"]["score"],
                    "text": feedbacks[conversation_id][4]["conciseness"]["text"],
                },
                {
                    "type": "response_time",
                    "score": feedbacks[conversation_id][5]["response_time"]["score"],
                    "text": feedbacks[conversation_id][5]["response_time"]["text"],
                },
                {
                    "type": "other_feedback",
                    "score": feedbacks[conversation_id][6]["other_feedback"]["score"],
                    "text": feedbacks[conversation_id][6]["other_feedback"]["text"],
                },
            ],
        }

    if data_feedback is None:
        list_feedback = [data_append]
        new_data_feedback = {
            "conversation_id": conversation_id,
            "feedbacks": list_feedback,
        }

        db.create_document_in_collection(
            database_name=DATABASE_NAME_LOGS_CHAT,
            collection_name="feedbacks-chat",
            param={**new_data_feedback},
        )
    else:
        new_list_feedbacks = data_feedback["feedbacks"]
        new_list_feedbacks.append(data_append)
        new_data_feedback = {
            "conversation_id": conversation_id,
            "feedbacks": new_list_feedbacks,
        }
        db.update_documents_in_collection_by_property(
            collection_name="feedbacks-chat",
            database_name=DATABASE_NAME_LOGS_CHAT,
            filter={"conversation_id": conversation_id},
            params=new_data_feedback,
        )