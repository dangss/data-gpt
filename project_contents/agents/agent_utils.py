import ast
import copy
import json
import os
import re
import time
from textwrap import dedent
from traceback import format_exc
from typing import List, Union

import openai
import pandas as pd
import streamlit as st
from langchain.llms import OpenAI
from pandas.api.types import (
    is_categorical_dtype,
    is_object_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_datetime64_any_dtype,
)

import numpy as np
from datetime import datetime
import faiss

base_model = "gpt-4o-mini"
llm_common = OpenAI(model_name=base_model, temperature=0, top_p=0.001, seed=212, max_tokens=2048)

## ChatGPT_HANDLER
# class ChatGPT_Handler:  # designed for ChatCompletion API
#     def __init__(self, model_name: str = base_model, extract_patterns: list = [], max_response_tokens: int = None,
#                  token_limit: int = None, temperature: float = 0) -> None:
#         self.max_response_tokens = max_response_tokens
#         self.token_limit = token_limit
#         self.temperature = temperature
#         self.extract_patterns = extract_patterns
#         # Longest pattern first
#         self.__extract_pattern_heads = '|'.join(sorted(
#             ['^' + p[:p.find('\\n')] + r'(\r|\n|\r\n)' for name, p in self.extract_patterns],
#             key=lambda pair: len(pair[1]),
#             reverse=True
#         ))
#         self.model_name = model_name
#         self.conversation_history = []
#         self.last_messages = FullMessage(role='assistant')
#         self.msg_type_notes = {
#             'agree': 'This is an "agree" message allowing you to continue your process.',
#             'give_feedback': 'This is an "give_feedback" message requesting you to modify your code.',
#             'ask_information': 'This is an "ask_information" message requesting you more information to make decision.'
#         }
#         if 'HELICONE_API_KEY' in os.environ.keys():
#             self.headers = {
#                 "Helicone-Auth": f"Bearer {os.environ['HELICONE_API_KEY']}",
#             }
#         else:
#             self.headers = {}
#
#     def _call_llm(self, agent: str = None, history: List[dict]=None, stop: List[str]=None):
#         """Call LLM to produce a response.
#
#         Parameters
#         ----------
#         history: List[str]
#             The user prompt to answer.
#         stop: List[str]
#             Stop condition, will be forwarded to ChatCompletion.create.
#
#         Returns
#         -------
#         msg: ChatMessage
#             The response packed in a ChatMessage object.
#         """
#         message_placeholder = st.empty()
#         # full_response = FullMessage(role='assistant', agent=agent, in_progress=True)
#         self.last_messages.agent = agent
#         self.last_messages.in_progress = True
#
#         current_length = len(self.last_messages.messages)
#         for response in openai.ChatCompletion.create(
#                 model=self.model_name,
#                 messages=history,
#                 temperature=self.temperature,
#                 max_tokens=self.max_response_tokens,
#                 stop=stop, stream=True,
#                 presence_penalty=0.9,
#                 seed=212,
#                 headers={
#                     "Helicone-Auth": f"Bearer {os.environ['HELICONE_API_KEY']}",
#                 },
#                 top_p=0.0001
#         ):
#             delta = response.choices[0].delta.get("content", "")
#             self.last_messages.add_delta(delta)
#             # Display the message as it is being streamed, start from current length (at this time, last_messages still include the old messages)
#             # if len(self.last_messages.messages) == current_length: # not having new Message
#             #     current_length -= 1
#             #     self.last_messages.show(message_placeholder, start_index=current_length)
#             # else:
#             #     self.last_messages.show(message_placeholder, start_index=current_length)
#
#         # Display the complete message.
#         self.last_messages.in_progress = False
#         if len(self.last_messages.messages) == current_length: # not having new Message
#             current_length -= 1
#
#         # print(self.last_messages.messages)
#         self.last_messages.messages = self.last_messages.messages[current_length:]
#         self.last_messages.show(message_placeholder)
#
#         return self.last_messages
#
#     def update_history(self, updated_message:'FullMessage'):
#         """Update conversation history if any change is applied"""
#         content = "".join([msg.content for msg in updated_message.messages])
#         role = updated_message.role
#         self.conversation_history[-1] = {"role": role, "content": content}
#
#     def get_next_steps(self, user_message: str = "", assistant_response: str = "", agent: str = None, stop: List[str]=[]):
#         if len(user_message) > 0:
#             self.conversation_history.append({"role": "user", "content": user_message})
#             # self.conversation_history.append({"role": "assistant", "content": "\nThought: I "})
#             self.last_messages.messages = []
#
#         if len(assistant_response) > 0:
#             new_history = add_assistant_msg(self.conversation_history, assistant_response)
#             self.conversation_history = copy.deepcopy(new_history)
#         try:
#             llm_output = self._call_llm(agent=agent, history=self.conversation_history, stop=stop)
#         except:
#             error_msg = format_exc()
#             llm_output = f"OPENAI_ERROR: {error_msg}"
#             self.conversation_history.append({"role": "assistant", "content": llm_output})
#
#         return llm_output

## FUNCTION TOOLS ##
def get_dtype_feature(column):
    # check numeric
    if is_integer_dtype(column):  # such as hours
        return "numeric"
    if is_float_dtype(column):
        return "numeric"
    if is_datetime64_any_dtype(column):
        return "datetime"
    if is_categorical_dtype(column) or is_object_dtype(column):
        try:
            pd.to_datetime(column)
            return "datetime"
        except:
            return "categorical"

def get_column_values(column:pd.Series, column_dtype:str, nb_categories:int=None):
    new_info = ''
    if column_dtype == 'numeric':
        new_info += f'Value ranges: {column.round(2).min()} - {column.round(2).max()}'
    elif column_dtype == 'categorical':
        unique_values = column.dropna().unique().tolist()
        if len(unique_values) < 2:
            new_info += f"Unique values: {unique_values} (No need filtering on this column)"
        else:
            if nb_categories is not None:
                unique_values = f'{unique_values[:nb_categories]}, and other values.'
            new_info += f"Unique values: {unique_values}"
    else: # datetime
        if column.min() != column.max():
            new_info += f"Time ranges: {column.min()} => {column.max()}"
        else:
            new_info += f"Specific Datetime: {column.min()} (No need filtering on this column)"

    return new_info

def get_column_info(df:pd.DataFrame, nb_categories:int=None):
    info = f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns\nColumns information:"
    for column in df.columns:
        dtype = get_dtype_feature(df[column])
        new_info = f"Column '{column}', datatype '{dtype}'"
        new_info += f'\n\t\t- {get_column_values(df[column], column_dtype=dtype, nb_categories=nb_categories)}'
        info += f'\n\t- {new_info}'
    return info + "\n"

def get_full_tables(relevant_tables: dict, return_columns: bool = False) -> Union[dict, list]:
    """Get all the chosen tables and columns (including 'target' and 'factor' table), along with duration of table"""
    all_tables = dict()
    for feature_type in relevant_tables.keys():
        for table in relevant_tables[feature_type].keys():
            if table not in all_tables.keys():
                all_tables[table] = dict()
            if feature_type != 'time_condition':
                current_columns = all_tables.get(table, dict()).get('columns', [])
                all_tables[table]['columns'] = current_columns + [c for c in
                                                                  relevant_tables[feature_type].get(table, []) if
                                                                  c not in current_columns]
            else:
                all_tables[table]['time_condition'] = relevant_tables[feature_type][table]
    if not return_columns:
        return list(all_tables.keys())
    return all_tables

def get_selected_tables(selected_tables: dict, return_columns: bool = False) -> Union[dict, list]:
    """Get all the chosen tables and columns (including 'target' and 'factor' table), along with time condition of chosen tables"""
    all_tables = dict()
    for feature_type in selected_tables.keys():
        for table in selected_tables[feature_type].keys():
            if table not in all_tables.keys():
                all_tables[table] = dict()
            if feature_type != 'time_condition':
                current_columns = all_tables.get(table, dict()).get('columns', [])
                all_tables[table]['columns'] = current_columns + [c for c in
                                                                  selected_tables[feature_type].get(table, []) if
                                                                  c not in current_columns]
            else:
                all_tables[table]['time_condition'] = selected_tables[feature_type][table]
    if not return_columns:
        return list(all_tables.keys())
    return all_tables

def normalize_message(old_message:str, new_message:str, is_concat:bool=True):
    """
    :param old_message: the previous message
    :param new_message: the message to concat
    """

    if new_message is None:
        return old_message

    last_thought_index = old_message.rfind("Thought:")
    last_observation_index = old_message.rfind("Observation:")

    # Check if a manual thought has been added and the new input is a thought
    if last_thought_index > last_observation_index and new_message.strip().startswith("Thought"):
        old_message = old_message[:last_thought_index] # remove manual Thought of previous assistant message

    if is_concat and isinstance(new_message, str):
        old_message += new_message

    return old_message

def add_assistant_msg(history:List[dict]=[], new_input:str=""):
    last_msg = history[-1]
    if last_msg['role'] == 'assistant':
        append_to_last_msg = True
        last_assistant_msg = last_msg['content']
    elif last_msg['role'] == 'user' and history[-2]['role'] == 'assistant':
        append_to_last_msg = False
        last_assistant_msg = history[-2]['content']
    else: # system before user - simply append new input to assistant
        history.append({"role": "assistant", "content": new_input})
        return history

    last_assistant_msg = normalize_message(old_message=last_assistant_msg, new_message=new_input, is_concat=append_to_last_msg)

    if append_to_last_msg: # last msg is assistant msg
        history[-1]['content'] = last_assistant_msg
    else:
        history[-2]['content'] = last_assistant_msg # replace the previous assistant msg with normalize version
        history.append({"role": "assistant", "content": new_input})

    return history

def extract_python_code(response: str = ""):
    extracted_code = re.search(r'```python(.*?)```', response, re.DOTALL)
    return extracted_code.group(1).strip() if extracted_code else None
def format_code(code:str="", result_name='df_result'):
    """Print the variables if they are not printed in the code"""
    # Regular expression pattern to match variable assignments
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)')

    # Find all variable assignmentst
    matches = re.findall(pattern, code)

    # Loop through matches and insert print statements
    variable_names = []
    for match in matches:
        variable_name, _ = match
        variable_names.append(variable_name)

    new_code = []
    for line in code.split('\n'):
        if line in variable_names:
            line = f'print({line})'
        new_code.append(line)

    new_code = dedent('\n'.join(new_code))

    # append print(df_result.head()) if not exists
    if f'{result_name} =' in new_code and f'print({result_name}' not in new_code:
        new_code += f"\nprint({result_name})"
    new_code = new_code.replace('print(df_result)', 'print(df_result.head())') # avoid print all the data
    return new_code

def read_table(table_name: str, tables: dict):
    if table_name is None:
        raise table_none

    if table_name not in tables.keys():
        raise table_not_exist(table_name=table_name, all_tables=list(tables.keys()))

    if 'data' not in tables[table_name].keys():
        data_path = tables[table_name]['path']

        if data_path.endswith(".csv"):
            tables[table_name]['data'] = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            tables[table_name]['data'] = pd.read_parquet(data_path)
        else:
            raise NotImplementedError("This file format is not supported.")

    return tables[table_name]['data']
def merge_tables(tables:List[pd.DataFrame], keys:List[str]=[], how='inner', missing_threshold=0.9):
    """Assume all the tables having identical primary keys"""
    for idx, table in enumerate(tables):
        for key in keys:
            if key not in table.columns.tolist():
                raise KeyError(f"Key {key} is not in the table at index {idx}'s column list. Review your column selection to make sure the keys are included in the data.")

    if len(tables) == 0:
        raise ValueError(f'''There is no table in the database having the specified keys: {keys}. Please review your keys.''')

    df_final = tables[0].copy()
    for table in tables[1:]:
        new_data = table[[c for c in table.columns if c in keys or c not in df_final.columns.tolist()]] # remove duplicated columns
        df_final = df_final.merge(new_data, on=keys, how=how)

    missing_ratios = df_final.isnull().mean()
    columns_to_drop = missing_ratios[missing_ratios > missing_threshold].index
    df_final.drop(columns=columns_to_drop, inplace=True)

    print('The datasets have been merged successfully.')
    return df_final

def show_history(history):
    for item in history:
        if item['role'] != 'system':
            print(item['role'])
            print(item['content'])
            print('-----')
## PROMPT TOOLS ##

def correct_indentation(code):
    # Split the code into lines
    lines = code.split('\n')

    # Find the minimum indentation level (ignore empty lines)
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())

    # Remove the minimum indentation from all lines
    corrected_lines = [line[min_indent:] if len(line.strip()) > 0 else line for line in lines]

    # Re-join the lines back into a single string
    corrected_code = '\n'.join(corrected_lines)

    return corrected_code

VALID_INTENTS = ['Descriptive', 'Diagnostic', 'Predictive', 'Others']
def get_response(prompt:str, model_name:str='gpt-3.5-turbo-1106'):
    llm_common.model_name = model_name
    response = llm_common(prompt)
    llm_common.model_name = base_model
    return response

@st.cache_data(ttl=60*60)
def combine_result(previous_result:str, refined_result:str, model_name:str=base_model, is_code:bool=False):
    PROMPT = """
    You are an assistant who excels at combining information.
    Given two objects, one being the previous object and the other the refined result, your task is to combine the two objects into one, incorporating the modifications from the refined object. The object can be text or a Python code snippet. Simply return the combined result without any explanation.

    [Previous Object]
    {previous_result}

    [Refined Object]
    {refined_result}

    [Combined Object]
    """
    if is_code:
        previous_result = f"```python\n{previous_result}\n```"
        refined_result = f"```python\n{refined_result}\n```"
    prompt_to_use = PROMPT.format(previous_result=previous_result, refined_result=refined_result)
    return get_response(prompt_to_use, model_name=model_name)

@st.cache_data(ttl=60*60)
def get_question_intent(user_question:str="", model_name:str=base_model):
    if user_question is None or user_question == "":
        raise ValueError("Parameter 'user_question' cannot be empty. Please specify the user_question to get its intent.")

    PROMPT = """
You are an agent tasked with detecting the type of questions asked by users.
Each question is classified into one of these categories:
    1. Descriptive Question: Inquires about past events, situations or explores the relationships / correlations / trends in the past data. Also, this question can ask about the overview of past data.
    2. Diagnostic Question: Seeks explanations or reasons for past events or situations, or idenfities the impact of many factors on a specific target.
    3. Predictive Question: Requests predictions or forecasts based on historical data. This question can be determined with the keywords such as: 'will', 'predict', 'would', 'could'...
    4. Others: Questions that cannot be categorized into any of the aforementioned types.

Given a question, your task is to categorize it into the appropriate type based on its content.
Let's think step by step.

Use this following format:
'''
Question: <user question>
Thought: your short explanation for the question category that you chose.
Final Answer: [[type of question]]
'''

For example:
'''
1.
Question: Identify and visualize the provinces that are in top 3 the highest average GRDP in 2018 - 2022.
Thought: This question asks about the data in the past, without any explanation or inferences, so this is a Descriptive question.
Final Answer: [[Descriptive Question]]

2.
Question: Explain the decrease of income in 2022
Thought: This question wants to know the reason why income in 2022 decreased, therefore, this is a Diagnostic question.
Final Answer: [[Diagnostic Question]]

3.
Question: What is the GRDP per capita in next year?
Thought: This question wants to predict the GRDP per capita in the future (next year), therefore, this is a Predictive question.
Final Answer: [[Predictive Question]]

4.
Question: Are there any differences between users from iOS and Android?
Thought: This question wants to seek the differences between iOS user and Android user, which needs a conclusion if they are different or not. Therefore, this is a Diagnostic question.
Final Answer: [[Diagnostic Question]]

5.
Question: 'What should I do if the sales go up?'
Thought: This question needs to draw conclusion about an action, which cannot be categorized as Descriptive, Diagnostic or Predictive Question. Therefore, it is an Others question.
Final Answer: [[Others]]
'''

Begin!
Question: '{user_question}'
"""

    prompt_to_use = PROMPT.format(user_question=user_question)
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    pattern = r"Thought:\s*(.*?)\s*Final Answer:\s*\[\[(.*?)\]\]"
    match = re.search(pattern, response)

    if match:
        thought_part = match.group(1)
        final_answer_part = match.group(2)
        extracted_parts = [thought_part, final_answer_part.replace('Question', '').strip()]

        return extracted_parts

    return ['Unable to detect question intent.', 'Others']
@st.cache_data(ttl=60*60)
def question_to_search_infos(question: str = None, example_objects: list = [], model_name:str=base_model):
    '''
    This function splits the user's question into smaller components/entities, facilitating information retrieval and response.
    '''

    if question is None or question == "":
        raise ValueError("Question must be specified. Now it is empty.")

    PROMPT = dedent("""
    << TASK DESCRIPTION >>
    Given a question, your task is to determine the values for entities (information type) in this list:
    - TIME: [the interested time period, not always the period explicit in the question but may be implicit]
    - TARGET: [list of the information of interest to search, maybe general or breakdown by a particular object, including: {example_objects}]
    - FACTOR: [list of independent variables that is used to predict or explain the TARGET]
    - GOAL: [the action to be taken regarding the TARGET, should be clear, concise and start with: "show information", "give explanation", "predict", "compare", "show relationship", "calculate"]
    - EXTERNAL_KNOWLEDGE: [the external information provided by user, such as formula, or news]

    << FORMAT >>
    Follow strictly this format:
    '''
    User question: <question from user>
    Thought: think about the entity that the question contains, and assign the value for the detected entity.
    Final Answer: 
    ---
    list of [ENTITY_NAME: VALUE]
    ---
    '''

    << EXAMPLE >>
    Here are some examples:
    0. 
    '''
    User question: 'Is there any difference between provinces with a GRDP of industry higher than that of agriculture, and those where it is reversed?'
    Thought: The question aims to detect the differences between provinces having higher GRDP of industry than agriculture and the control group, therefore, the TARGET of this question is a label indicating if a province meet the requirement or not, of object 'province'.
    Final Answer:
    ---
    - TIME: []
    - TARGET: ['GRDP of industry', 'GRDP of agriculture']
    - FACTOR: []
    - GOAL: ["compare"]
    - EXTERNAL_KNOWLEDGE: []
    ---
    '''
    1.
    '''
    User question: 'In Can Tho and An Giang, does the social indicators have an impact on the decrease in the negative growth rate of outstanding liquidity of credit institutions, branches of foreign bank?'
    Thought: The question asks about the impact of "social indicators" of object 'province' with value 'Can Tho', 'An Giang' on the negativity of "growth rate of outstanding liquidity of credit institutions, branches of foreign bank" of object 'province' with value 'Can Tho', 'An Giang' (TARGET).
    Final Answer:
    ---
    - TIME: []
    - TARGET: ["growth rate of outstanding liquidity of credit institutions, branches of foreign bank of each province (An Giang, Can Tho)"]
    - FACTOR: ["social indicators of province (Can Tho, An Giang)"]
    - GOAL: ["show relationship"]
    - EXTERNAL_KNOWLEDGE: []
    ---
    '''

    2.
    '''
    User question: 'Visualize the changes of total state budget in VND and in USD of MekongDelta region for the period 2018 - 2020, breakdown by frequent and investment expenditure'
    Thought: The question asks about the total state budget of the whole Mekong Delta region, broken down by the object 'budget type': frequent budget and investment budget (TARGET). Regarding TIME, since it mentions the 'change' of a value over years, while the interested period is 2018 - 2022, I consider to get period from 2017 to get changes in 2018. Therefore, TIME will be '2017 - 2022' instead of '2018 - 2022'.
    Final Answer:
    ---
    - TIME: ["2017-2022"]
    - TARGET: ["total state budget in VND and in USD of Mekong Delta region of budget type (frequent and investment)"]
    - FACTOR: []
    - GOAL: ["show information"]
    - EXTERNAL_KNOWLEDGE: []
    ---
    '''

    3.
    '''
    User question: 'In 2015, what is the relationship between total domestic revenue and labor size of each province?'
    Thought: The question asks about the relationship between (1) total revenue of object 'budget type' with value 'Domestic', and (2) the labor size of object 'province'. Since there is no particular TARGET mentionned in the question, I will assume the first factor as TARGET, and the other as FACTOR. TIME of question is 2015.
    Final Answer:
    ---
    - TIME: ["2015"]
    - TARGET: ["total revenue of budget type Domestic of each province"]
    - FACTOR: ["labor size of each province"]
    - GOAL: ["show relationship"]
    - EXTERNAL_KNOWLEDGE: []
    ---
    '''

    4.
    '''
    User question: 'Which factors best explain the changes of GDP per capita over years'
    # Thought: the question asks about an explanation (GOAL) on the changes of GDP per capita (TARGET) without mentionning any specific FACTOR. The object of this question is not mentionned either, so we can assume it is largest-level object 'province'.
    Final Answer:
    ---
    - TIME: []
    - TARGET: ['GDP per capita of each province']
    - FACTOR: []
    - GOAL: ['give explanation']
    - EXTERNAL_KNOWLEDGE: []
    ---
    '''
    
    << REQUIREMENTS >>
    - GOAL and TARGET are essential components that every question must include.
    - For any ENTITIES not mentioned, initialize an empty list.
    - TARGET and FACTOR should be clearly defined. It is advisable to extract TARGET and FACTOR as individual components, rather than as multiple features.
    Let's think step by step.

    << YOUR TURN >>
    Begin!
    User question: {question}
    Thought:
    """)
    if len(example_objects) == 0:
        # example_objects = ['user', 'ward/district', 'household', 'province', 'product']
        # TODO: modify the example objects for specific database
        example_objects = [
            'province', 'ward/district', 'economic sector', 'economic ownership', 'economic activities',
            'occupation', 'employment status', 'state budget type', 'mobilization balance', 'credit liquidity'
        ]

    prompt_to_use = PROMPT.format(question=question, example_objects=example_objects)
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    pattern = r'- ([A-Z_]+): (\[.*?\])'

    # Using re.findall() to find all matches in the input string
    # This returns a list of tuples (key, value)
    extracted_info = re.findall(pattern, response)

    # Post-process the values to split by comma and strip quotes and whitespace
    try:
        processed_info = {key.lower(): ast.literal_eval(value.strip()) for key, value in extracted_info}
        return processed_info
    except:
        print(response)
        return None
@st.cache_data(ttl=60*60)
def suggest_time_range(question: str, time_range: str, model_name='gpt-3.5-turbo-1106'):
    PROMPT = """
[Instruction]
Given a question and a specified time range, your task is to detect the time format (based on the Python datetime.datetime), and to suggest the appropriate start time and end time to the user.

Your answer should follow this format:
[Thouhgt]
<your explanation about the time range you chose>
[Final Answer]
Format: <time format>
From: <start time point>
To: <end time point>

[Example]
Here are some examples:
<<Example 1>>
[Data Time Range]
'year': 2015 - 2023

[Question]
Show me the revenue in 2018 - 2022

[Thought]
The data is updated to 2022 (format %Y). The question asks about the period 2018 - 2022, therefore the time range is from 2018 to 2022.
[Final Answer]
Format: %Y
From: 2018
To: 2022
<<Example 1>>

<<Example 2>>
[Data Time Range]
'folder_date': 2024-04-01 - 2024-04-30

[Question]
Give me the daily budget in last week.

[Thought]
The data is updated to 2024-04-30 (format %Y-%m-%d). Therefore, the last week in this data is the week ending on 2024-04-30, which spans from 2024-04-24 to 2024-04-30.
[Final Answer]
Format: %Y-%m-%d
From: 2024-04-24
To: 2024-04-30
<<Example 2>>

<<Example 3>>
[Data Time Range]
'folder_date': 2024-04-01 - 2024-04-30

[Question]
Give me the daily budget in 2024-04-15.

[Thought]
The data is updated to 2024-04-30 (format %Y-%m-%d). The question only asks information for the date 2024-04-15, therefore, the expected time range is from 2024-04-15 to 2024-04-15.
[Final Answer]
Format: %Y-%m-%d
From: 2024-04-15
To: 2024-04-15
<<Example 3>>

<<Example 4>>
[Data Time Range]
'log_date': 2023/13/12 - 2024/13/12

[Question]
What is the average score in the last quarter?

[Thought]
The data is updated to 2024/13/12 (format %Y/%d/%m). Therefore, the last quarter is the final quarter of 2024, which spans from 2024/01/10 to 2024/13/12.
[Final Answer]
Format: %Y/%d/%m
From: 2024/01/10
To: 2024/13/12
<<Example 4>>

<<Example 5>>
[Data Time Range]
'date': 2024-01-04 - 2024-12-04

[Question]
Provide information about total profit.

[Thought]
The data is updated to 2024-12-04 (format %Y-%m-%d). However, the question does not mention any specific time range, so I will return full time range in the provided data.
[Final Answer]
Format: $Y-%m-%d
From: 2024-01-04
To: 2024-12-04
<<Example 5>>

Let's try it now.
[Data Time Range]
{time_range}

[Question]
{question}

[Thought]
"""
    prompt_to_use = PROMPT.format(time_range=time_range, question=question)
    response = get_response(prompt=prompt_to_use, model_name=model_name)

    time_format, from_time, to_time = response.split('[Final Answer]')[-1].strip().split('\n')
    time_format = time_format.replace('Format:', '').strip()
    from_time = from_time.replace('From:', '').strip()
    to_time = to_time.replace('To:', '').strip()
    try:
        return datetime.strptime(from_time, time_format), datetime.strptime(to_time, time_format)
    except:
        return None, None

def starts_with_number_text_pattern(string):
    pattern = r'^[\d|\*]+\. \w+'
    return bool(re.match(pattern, string))
def contains_analysis_part(string):
    analysis_part = ('Perform analysis')
    for part in analysis_part:
        if part in string:
            return True
    return False

def contains_visualization_part(string):
    visualization_part = ('visualization', 'visualize', 'draw')
    for part in visualization_part:
        if part in string.lower():
            return True
    return False

def plan_to_tasks(execution_plan: str):
    # Regular expression to split the execution plan
    pattern = r'\n(?=\d+\.\s)'  # Lookahead for a number followed by a dot and space

    # Splitting the execution plan using the regex pattern
    steps = re.split(pattern, execution_plan.strip())
    return steps

def generate_execution_plan(question: str, data_information: str, question_intent='Descriptive', example_plan:str=None,
                            model_name: str = 'gpt-3.5-turbo-1106', streaming_container=None):
    """Generate an execution plan to solve the question"""
    PROMPT = """
    [Actions]
    - Read data:
        - Description: Read data from specific tables, given the table names.
        - Goal: Get data from tables for further processing.
    - Grouping - Aggregation:
        - Description: Group the dataset by specified keys and perform an aggregation operation on a specified column.
        - Goal: Extract attributes of higher-level keys from lower-level data.
    - Filter data:
        - Description: Specify clearly the condition_field (column) and condition_values (value list or value ranges) to filter.
        - Goal: Only get the interested segments in data.
    - Get all features:
        - Description: Retrieve all features in the database given a key set.
        - Goal: Build a model or make a comparison on a specific target feature along with all features.
    - Creating new column:
        - Description: Generate a new column named <new_column_name> by applying a specified method to specified columns.
        - Goal: Create the target feature, or necessary information if it doesn't exist, always an attribute of data.
    - Merge datasets:
        - Description: Merge multiple datasets into one using specified keys, or merge target and factor features into one dataset. This step is mandatory for Diagnostic question.
        - Goal: Combine relevant information for analysis.
    - Perform analysis:
        - Description: Conduct any suitable analysis (build model, calculate correlation, get top result...) on target and factors to get answer.
        - Goal: Generate result and response for user using the prepared dataset. Simply say 'No analysis required' if the question only requires retrieving information.
    - Visualization:
        - Description: Draw a chart to illustrate the data.
        - Goal: Visualize the data as required.

    [Instruction]
    ---
    Given a question, type of question, and data information, your task is to determine which data should be prepared, create a list of actions using only the [Actions] suggested above to prepare the data for the answer, along with the required analysis to reach the response. Think twice before you provide your answer. Make sure your answer is valid, clear, and easy to understand. Keep the answer simple and remove any unnecessary steps.

    Your answer should follow this format:
    ```
    [Thought]
    <
    - List the provided tables along with their primary keys;
    - Determine the content of the data to prepare, and the desired data key;
    {logic_thought}
    - Determine whether the filter on datetime field is necessary.
    - If you need to convert a datetime type into a time unit such as hours or weekdays, you should add a step to convert the time to GMT+7.
    >

    [Output]
    <Step-by-step list>
    ```

    Let's think step by step.
    ---
    {example}

    Let's try it now.
    [Data Information]
    {data_information}

    [Question]:
    ---
    {question}
    ---
    """
    instruction_dict = {
        'Descriptive': dedent('''
    - Indicate the processing to reach each content:
        + If it exists in the dataset, simply mention the column name that presents that content.
        + Otherwise, suggest an approach to generate a suitable feature that aligns with the question using the provided data.
    '''),
        'Diagnostic': dedent('''
    - Determine the target feature of the question (the feature to explore or explain), how to create or retrieve the target column, and whether the table keys and the expected keys are the same. If they are not, perform grouping and aggregation on the expected keys.
    - Determine the factors to explain or to compare among target values, how to create / retrieve the factor columns, and whether the table keys and the expected keys are the same. If they are not, perform grouping and aggregation on the expected keys.
    - Determine the keys on which target and factors should be merged, always add a step to merge the target and factor feature.
    ''')
    }
    example_dict = {
        'Descriptive': dedent("""
    Here are some typical examples:
    ```
    <<Example 01>>
    [Data Information]
    You are provided 3 table(s):
    ---
    Table 'tinh_taikhoanquocgia'
    Primary Key:
        + 'nam', datatype 'int', values range: 2015 - 2023
        + 'tinh', datatype 'object, unique values: ['An Giang', 'Long An']
    Columns:
        + 'grdp_per_capita', datatype 'float', values range: 0 - 10
    ---
    Table 'tinh_nhankhauhoc'
    Primary Key:
        + 'nam', datatype 'int', values range: 2015 - 2023
        + 'tinh', datatype 'object, unique values: ['An Giang', 'Long An']
    Columns:
        + 'population', datatype 'float', values range: 0 - 10
        + 'average_lifespan', datatype 'float', values range: 1 - 100
    ---
    Table 'chi_ngan_sach'
    Primary Key:
        + 'nam', datatype 'int', values range: 2018 - 2022.
        + 'tinh', datatype 'object, unique values: ['An Giang', 'Long An']
        + 'loai_chi_ngan_sach_lv1': '''budget type, including:
        - chi_can_doi: balance expense
        - du_phong_ngan_sach: budget provision expense'''
        + 'loai_chi_ngan_sach_lv2': '''subtype of budget type at level 1, including:
        - chi_thuong_xuyen: frequent expenses.
        - chi_khac: other expenses.'''
        + 'loai_chi_ngan_sach_lv3': ''subtype of budget type at level 2, including:
        - chi_ho_tro: support expenses.
        - chi_cho_ho_tro: expenses for supporting
        - ho_tro_thuong_xuyen: frequent supporting expenses
        - ho_tro_khan_cap: urgent supporting expenses'''
    Columns:
        + 'tong_chi_ngan_sach_trieu_dong', datatype 'int', value range: 0 - 100
        + 'co_cau_chi_ngan_sach_%', datatype 'float', value range: 0 - 100
    ---

    [Question]
    ---
    Which province has the highest GRDP per capita and total budget expenses per capita in these categories: frequent expenses, aid expenses, and budget provisions, in the period 2018 - 2022?
    ---

    [Thought]
    I am provided 3 tables:
    - Table 'tinh_nhankhauhoc' with primary keys ['nam', 'tinh'], time range on 'nam': 2015 - 2023.
    - Table 'tinh_taikhoanquocgia' with primary keys ['nam', 'tinh'], time range on 'nam': 2015 - 2023.
    - Table 'chi_ngan_sach' with primary keys ['nam', 'tinh', 'loai_chi_ngan_sach_lv1', 'loai_chi_ngan_sach_lv2', 'loai_chi_ngan_sach_lv3'], time range on 'nam': 2018 - 2022.

    The prepared data should include information about GRDP per capita and total budget expenses per capita in frequent expenses, aid expenses, and budget provisions of each province. Since the provided tables do not contain information about budget per capita, this information can be indirectly obtained by dividing the total budget of each type by the population of each province. The primary keys for this dataset should be ('nam', 'tinh'). Then, use the GET_TOP analysis to identify the province with the highest value in each criterion.
    Besides, since the question mentions a time range 2018 - 2022, while the data time range is 2015 - 2023 in the first 2 tables, we need to filter time range 2018 - 2022 for the table 'tinh_taikhoanquocgia' and table 'tinh_nhankhauhoc', and no need apply this filter on table 'chi_ngan_sach.

    [Output]
    1. Read data: Read data from tables 'tinh_taikhoanquocgia' and 'chi_ngan_sach'.
    2. Filter Data:
        - Filter the 'nam' in value ranges [2018, 2022] on table 'tinh_taikhoanquocgia' only, since table 'chi_ngan_sach' has time range 2018 - 2022 already.
    3. Get the relevant attributes or required information of each province:
        - For information about GRDP per capita => this information can be obtained with: 
            + Get information GRDP per capita from table 'tinh'. Since this table's primary keys are ('nam', 'tinh'), no aggregation is necessary.
        - For information about total budget expenses, they are found in table 'thu_ngan_sach'. Since this table's keys are ['nam', 'tinh', 'loai_chi_ngan_sach_lv1', 'loai_chi_ngan_sach_lv2', 'loai_chi_ngan_sach_lv3'], it needs to be aggregated into ('nam', 'tinh'), as follows:
            3.1. For frequent expenses => this information can be obtained with:
                + Filter data: filter data with 'loai_chi_ngan_sach_lv2' in ['chi_thuong_xuyen'], and filter 'loai_chi_ngan_sach_lv3' (since 3 is the successor level of 2) is 'total' to get the total value of frequent expenses.
                + Grouping - Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'tong_chi_ngan_sach_trieu_dong'.
            3.2. For aid expenses => this information can be obtained with:
                + Filter data: filter data with 'loai_chi_ngan_sach_lv3' in ['chi_ho_tro', 'chi_cho_ho_tro', 'ho_tro_thuong_xuyen', 'ho_tro_khan_cap']. Since 3 is the last level category in this table, there's no need to filter the successor level as 'total'.
                + Grouping - Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'tong_chi_ngan_sach_trieu_dong'.
            3.3. For budget provision expenses => This information can be obtained with:
                + Filter data: filter data with 'loai_chi_ngan_sach_lv1' as 'du_phong_ngan_sach', and filter 'loai_chi_ngan_sach_lv2' (since 2 is the successor level of 1) is 'total' to get the total value of budget provision expenses.
                + Grouping Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'tong_chi_ngan_sach_trieu_dong'.
    4. Merge datasets: Merge all the relevant data from different tables to obtain overall information for each province in each year, on keys ['nam', 'tinh'].
    5. Create new column: Divide each type of expenses (frequent expenses, aid expenses, and budget provision expenses) by population to get budget expenses per capita.
    6. Perform analysis: Group the data province and calculate the average, as well as sort the values in descending order to find the province having:
        - Highest GRDP per capita.
        - Highest total budget expenses in frequent expenses.
        - Highest total budget expenses in aid expenses.
        - Highest total budget expenses in budget provisions.
    7. Visualization: No visualization required.
    <<Example 01>>
    ```"""),
        'Diagnostic': dedent("""
    Here are some typical examples:
    ```
    <<Example 01>>
    ```
    [Data Information]
    You are provided these tables:
    TARGET TABLE:
    ---
    Table 'tinh'
    Primary Key:
        + 'nam', datatype 'int', values range: 2015 - 2023
        + 'tinh', datatype 'object', unique values: ['An Giang', 'Long An']
    Columns:
        + 'GDP', datatype 'float', values range: 0 - 10
    ---

    FACTOR TABLES:
    ---
    Table 'nganh_kinh_te'
    Primary Key:
        + 'nam', datatype 'int', values range: 2018 - 2022
        + 'tinh', datatype 'object, unique values: ['An Giang', 'Long An']
        + 'nganh_kinh_te_lv1', dataype 'object', unique values: ['agriculture', 'local services', 'wholesales']
        + 'nganh_kinh_te_lv2', dataype 'object', unique values: ['state_education', 'non_state_education', 'total', 'wholesales_small', 'whole_sales_large']
    Columns:
        + 'GDP', datatype 'float', values range: 0 - 100
        + 'average_income', datatype 'float', values range: 1 - 3
        + 'labor_size', datatype 'float', values range: 5 - 8
    ---

    [Question]
    ---
    Which features related to economic activities Wholesales, and Education, have the greatest impact on GDP growth in An Giang in 2018 - 2022 ?
    ---

    [Thought]
    I am provided these tables:
    - TARGET: Table 'tinh' with primary keys ['nam', 'tinh'], time range on 'nam': 2015 - 2023.
    - FACTOR:
        + Table 'nganh_kinh_te' with primary keys ['nam', 'tinh', 'nganh_kinh_te_lv1', 'nganh_kinh_te_lv2'], time range on 'nam': 2018 - 2022.

    The prepared dataset should contain:
    - Target Information: the GDP growth of province.
    - Factor Information: the features related to economic activities Wholesales and Education of each province.

    The desired keys of this dataset is ['nam', 'tinh'].
    - For target information, the provided data does not contain the GDP growth of province, but it can be retrieved from the column 'GDP' in table 'tinh' by calculating the growth rate over years. Since the table 'tinh' has primary keys ('nam', 'tinh') and the expected keys are ('nam', 'tinh') (they are identical), no aggregation is needed.
    - For factor information, the question wants to explore the features related to economic activities Wholesales and Education, I need to filter these activities from table 'nganh_kinh_te'. Since the table 'nganh_kinh_te' has primary keys ('nam', 'tinh', 'nganh_kinh_te_lv1', 'nganh_kinh_te_lv2'), and the expected keys are ('nam', 'tinh') (they are different), it is necessary to grouping and aggregation on keys ('nam', 'tinh').
    - Merge the target and the factors, then perform BUILD_MODEL to explain the budget deficit status using all factors.

    [Output]
    1. Read data: Read data from tables 'tinh' and 'nganh_kinh_te'.
    2. Filter data:
        - Filter column 'tinh' is An Giang for both tables.
        - Filter the 'nam' column to include the years 2018-2022 for the 'tinh' table only, as the 'nganh_kinh_te' table already covers the period 2018-2022.
    3. Retrieve target and factors information:
        - For target information about GDP growth, get the data from table 'tinh'. Since this table's primary keys are ('nam', 'tinh') and the expected keys are also ('nam', 'tinh'), no aggregation is necessary. Since the provided table does not contain information about GDP growth, it can be obtained as follows:
            3.1. Create new column: sort the table 'tinh' on year, and use pct_change to calculate the growth of GDP.
        - For information about features related to economic activities, they are found in table 'thu_ngan_sach'. Since this table's keys are ['nam', 'tinh', 'nganh_kinh_te_lv1', 'nganh_kinh_te_lv2'] but the expected keys are ('nam', 'tinh') (they are different), it needs to be aggregated into ('nam', 'tinh'), as follows:
            3.2. For information about economic activity Wholesales => this information can be obtained with:
                + Filter data: filter data with 'nganh_kinh_te_lv1' in ['wholesales'], and filter 'nganh_kinh_te_lv2' (since this column exists) is 'total' to get the total/general value of activity Wholesales.
                + Grouping - Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'GDP', the sum of 'average_income' and the sum of 'labor_size'.
            3.3 For information about economic activity Education => this information can be obtained with:
                + Filter data: filter data with 'nganh_kinh_te_lv2' in ['state_education', 'non_state_education']. Since 2 is the last level category in this table, there's no need to filter the successor level as 'total'.
                + Grouping - Aggregation: since table 'thu_ngan_sach' ('nam', 'tinh'), grouping data on ('nam', 'tinh'), calculate the sum of 'GDP', the sum of 'average_income' and the sum of 'labor_size'.
            3.4. Merge datasets: Merge all the relevant data from different tables to obtain overall information for each province in each year, on keys ['nam', 'tinh'].
    4. Perform analysis: Build a model to identify the most impact features on GDP growth.
    ```
    <<Example 01>>

    <<Example 02>>
    ```
    [Data Information]
    You are provided these tables:
    TARGET TABLE:
    ---
    Table 'tinh'
    Primary Key:
        + 'nam', datatype 'int', values range: 2015 - 2023
        + 'tinh', datatype 'object', unique values: ['An Giang', 'Long An']
        + 'nganh_kinh_te_lv1', datatype 'object', unique values: ['cong nghiep', 'thuc pham']
    Columns:
        + 'budget_expenses', datatype 'float', values range: 0 - 100.
        + 'budget_revenue', datatype 'float', values range: 0 - 80.
    ---

    [Question]
    ---
    What are the differences between provinces with a budget deficit and those without?
    ---

    [Thought]
    I am provided only TARGET table:
    - TARGET: Table 'tinh' with primary keys ['nam', 'tinh'], time range on 'nam': 2015 - 2023.

    The prepared data should contain:
    + Target Information: the budget deficit status of each province.
    + Factor Information: all the features to explain the budget deficit status.

    The desired keys are ['nam', 'tinh'].
    - For the target information, since the budget deficit status is not presented in the provided information, I need to create new column 'has_budget_deficit' by checking if a province has budget expenses more than budget revenue. This information can be obtained by calculating the total budget expenses and total budget revenue of each province, then get the distance between them, and finally checking if the total budget expenses is more than the total budget revenue.
    - For the factors information, since FACTOR tables are not specified, the default option is to obtain all the features of the province (corresponding to keys ['nam', 'tinh']).
    - Merge the target and all features, then perform BUILD_MODEL to explain the budget deficit status using all features.

    [Output]
    1. Read data: Read data from table 'tinh'.
    2. Retrieve target and factors information:
        + For the target information about budget deficit status, it can be retrieved from table 'tinh'. Since this table has primary keys ('nam', 'tinh', 'nganh_kinh_te') but the expected keys are ('nam', 'tinh') (they are different), it is necessary to grouping and aggregation on ('nam', 'tinh'), as follow:
            2.1. Grouping Aggregation: group by the table 'tinh' on ('nam', 'tinh') and calculate the sum of 'budget_expenses', and sum of 'budget_revenue'.
            2.2. Create new column 'has_budget_deficit' by checking if ('sum_budget_expenses' - 'sum_budget_revenue') > 0.
        + For factor information, since there is no feature specified, the default option is to get all features of province with the expected keys, using:
            2.2. Get all features: get all the features of province to explain the budget deficit status 'has_budget_deficit' of province, using expected keys ('nam', 'tinh').
        + Merge target and all features: even though there is only one table, there are 2 datasets containing target and factors => Merge the target with all the features to get enough content to explain the target.
            2.3. Merge datasets: merge the target and all features on key ('nam', 'tinh').
    3. Perform analysis: Build a model to explain the budget deficit status.
    4. Visualization: No visualization required.
    ```
    <<Example 02>>""")
    }

    example = example_dict.get(question_intent, "")
    if example_plan is not None:
        example += f"""
        <<Specific Example>>
        ```
        {example_plan}
        ```
        <<Specific Example>>
        """

    prompt_to_use = PROMPT.format(question=question, data_information=data_information, example=example, logic_thought=instruction_dict.get(question_intent, ""))
    llm_common.model_name = model_name
    response = ""

    for chunk in llm_common.stream(prompt_to_use):
        response += chunk
        if streaming_container is not None:
            with streaming_container.container(border=True): st.text(response)

    if streaming_container is not None:
        streaming_container.empty()

    thought, execution_plan = response.replace("```", "").split('[Output]')
    execution_plan = execution_plan.strip()
    llm_common.model_name = base_model
    return thought, execution_plan

def refine_execution_plan(execution_plan: str, feedback: str, model_name='gpt-3.5-turbo-1106'):
    PROMPT = """
[Instruction]
---
Given a question, an execution plan to solve the question, and feedback, your task is to modify the execution plan as required. You should return all the steps, including both the unchanged and changed steps (except the removed steps), in your modified execution plan.

Your answer should follow this format:
```
[Output]
<Step-by-step list of modified plan, if there is no modification needed, simply return the original plan>
```

Let's try it now.

[Execution Plan]
{execution_plan}

[Feedback]
{feedback}

[Output]
"""
    prompt_to_use = PROMPT.format(execution_plan=execution_plan, feedback=feedback)
    response = combine_result(previous_result=execution_plan, refined_result=get_response(prompt=prompt_to_use, model_name=model_name))
    return response.split('[Output]')[-1].replace("```", "").strip()

@st.cache_data(ttl=60*60)
def split_execution_plan(execution_plan: str):
    execution_lines = execution_plan.split('\n')
    execution_parts = dict()

    curr_part = "data_agent"
    start_index = 0
    while not starts_with_number_text_pattern(execution_lines[start_index]):
        start_index += 1

    for line in execution_lines[start_index:]:
        if starts_with_number_text_pattern(line):
            if not contains_visualization_part(line) and not contains_analysis_part(line):
                curr_part = "data_agent"
            elif "Perform analysis" in line:
                if "model" in line.lower():
                    curr_part = "model_agent"
                else:
                    curr_part = "analysis_agent"
            elif contains_visualization_part(line):
                curr_part = "vis_agent"
        execution_parts[curr_part] = execution_parts.get(curr_part, '') + f'\n{line}'

    return execution_parts

@st.cache_data(ttl=60*60)
def generate_code(data_information: str, request: str, code_utils:str="", question_intent:str="Descriptive", model_name='gpt-3.5-turbo-1106'):
    """Generate and execute a Python code to process dataset based on a specific request"""
    PROMPT = """
    {code_utils}
    << TASK DESCRIPTION >>
    Given a request, your task is to write a Python code that solves the request, and assign the final result to 'result'. You should prioritize the utils in list shown above in your code. Do not generate or define any function.
    
    Your answer should follow this format:
    '''
    [FINAL ANSWER]
    ```python
    <your python code which solves the request and returns 'result' as the final result in the code>
    ```
    '''
    
    {example}
    
    Let's try it now.
    
    [DATA INFORMATION]
    {data_information}
    
    [REQUEST]
    {request}
    
    [FINAL ANSWER]
    """
    example = dedent('''
    <<Example>>
    [DATA INFORMATION]
    You are provided 1 table(s).
    ---
    Table 'tinh'
    Primary Key:
        + 'nam', datatype 'int', values range: 2015 - 2023
        + 'tinh', datatype 'object', unique values: ['An Giang', 'Long An']
    Columns:
        + 'grdp_per_capita', datatype 'float', values range: 0 - 10
    ---
    Table 'thu_ngan_sach'
    Primary Key:
        + 'nam', datatype 'int', values range: 2015 - 2023
        + 'tinh', datatype 'object', unique values: ['An Giang', 'Long An']
        + 'loai_chi_ngan_sach_lv1', datatype 'object', unique values: ['chi_can_doi', 'du_phong_ngan_sach']
        + 'loai_chi_ngan_sach_lv2', datatype 'object', unique values: ['total', 'chi_thuong_xuyen', 'chi_khac']
        + 'loai_chi_ngan_sach_lv3', datatype 'object', unique values: ['total', 'chi_ho_tro', 'chi_cho_ho_tro', 'ho_tro_thuong_xuyen', 'ho_tro_khan_cap']
    Columns:
        + 'tong_chi_ngan_sach_trieu_dong', datatype 'int', value range: 0 - 100
        + 'co_cau_chi_ngan_sach_%', datatype 'float', value range: 0 - 100
    ---

    [REQUEST]
    ###
    1. Read data: Read data from tables 'tinh' and 'chi_ngan_sach'.
    2. Filter data:
        - Filter the column 'năm' with value ranges [2018, 2022], using function 'between', for both table 'tinh' and 'chi_ngan_sach'.
    3. Get the relevant attributes of each province:
        - For information about GRDP per capita => Keep Original Data: it is located in table 'tinh' with keys ('nam', 'tinh'), therefore no aggregation is necessary.
        - For information about total budget expenses, they are found in table 'thu_ngan_sach'. Since this table's keys are ['nam', 'tinh', 'loai_chi_ngan_sach_lv1', 'loai_chi_ngan_sach_lv2', 'loai_chi_ngan_sach_lv3'], it needs to be aggregated into ('nam', 'tinh'), as follows:
            3.1. For frequent expenses => Filter data: filter data with 'loai_chi_ngan_sach_lv2' in ['chi_thuong_xuyen'], and Grouping - Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'tong_chi_ngan_sach_trieu_dong'.
            3.2. For aid expenses => Filter data: filter data with 'loai_chi_ngan_sach_lv3' in ['chi_ho_tro', 'chi_cho_ho_tro', 'ho_tro_thuong_xuyen', 'ho_tro_khan_cap'], and Grouping - Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'tong_chi_ngan_sach_trieu_dong'.
            3.3. For budget provision expenses => Filter data: filter data with 'loai_chi_ngan_sach_lv1' as 'du_phong_ngan_sach', and Grouping Aggregation: grouping data on ('nam', 'tinh'), calculate the sum of 'tong_chi_ngan_sach_trieu_dong'.
        - Merge datasets: Merge all the above data to obtain overall information for each province in each year.
    ###

    [FINAL ANSWER]
    ```python
    # 1. Read data
    tinh = read_table('tinh')
    chi_ngan_sach = read_table('chi_ngan_sach')

    # 2. Filter data
    tinh_filtered = tinh[(tinh['nam'].between(2018, 2022))]
    chi_ngan_sach_filtered = chi_ngan_sach[(chi_ngan_sach['nam'].between(2018, 2022))]

    # 3. Get the relevant attributes of each province
    # 3.1. For frequent expenses
    frequent_expenses = chi_ngan_sach_filtered[(chi_ngan_sach_filtered['loai_chi_ngan_sach_lv2'] == 'chi_thuong_xuyen')&(chi_ngan_sach_filtered['loai_chi_ngan_sach_lv3'] == 'total')].groupby(['nam', 'tinh']).agg({'tong_chi_ngan_sach_trieu_dong': 'sum'}).reset_index()

    # 3.2. For aid expenses
    aid_expenses = chi_ngan_sach_filtered[(chi_ngan_sach_filtered['loai_chi_ngan_sach_lv3'].isin(['chi_ho_tro', 'chi_cho_ho_tro', 'ho_tro_thuong_xuyen', 'ho_tro_khan_cap'])))].groupby(['nam', 'tinh']).agg({'tong_chi_ngan_sach_trieu_dong': 'sum'}).reset_index()

    # 3.3. For budget provision expenses
    budget_provision_expenses = chi_ngan_sach_filtered[(chi_ngan_sach_filtered['loai_chi_ngan_sach_lv1'] == 'du_phong_ngan_sach')&(chi_ngan_sach_filtered['loai_chi_ngan_sach_lv2'] == 'total')].groupby(['nam', 'tinh']).agg({'tong_chi_ngan_sach_trieu_dong': 'sum'}).reset_index()

    # 5. Merge datasets
    result = merge_tables([tinh_filtered, frequent_expenses, aid_expenses, budget_provision_expenses], keys=['nam', 'tinh'], how='left')
    ```
    <<Example>>
    ''')
    prompt_to_use = dedent(PROMPT).format(code_utils=code_utils, example=example, data_information=data_information, request=request)
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    python_code_extracted = extract_python_code(response.split('[FINAL ANSWER]')[-1])
    return python_code_extracted

@st.cache_data(ttl=60*60)
def get_response_type(message: str, model_name:str=base_model):
    """Detect the type of user message: including 'Agree', 'Disagree', 'Question'"""
    PROMPT = """
[Instruction]
---
Given a user message, your task is to classify the message into one of these three categories:
- 'agree': a reponse accepting a solution.
- 'give_feedback': a response denying, or giving some feedbacks, or suggesting another approaches, to modify the solution.
- 'ask_information': a response requesting more information to make decision.

Your answer should follow this format:
```
[Output]
<response category>
```

Here are some examples:
1.
```
[Message]
Yes, it's OK.

[Output]
agree
```

2.
```
[Message]
Generate a step filtering date.

[Output]
give_feedback
```

3.
```
[Message]
Tell me the formula to get the CR.

[Output]
ask_information
```

4.
```
[Message]
Could you change the name of the column with the prefix 'score_'?

[Output]
give_feedback
```

Let's try it now.
[Message]
{message}

[Output]
"""
    prompt_to_use = PROMPT.format(message=message)
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    return response.split('[Output]')[-1].replace('```', '').strip()

def generate_next_message(history: List[dict]=None, stop: List[str]=None):
    """Call LLM to produce a response."""
    full_response = ""
    if len(history) > 5:
        history.pop(1)
    for response in openai.ChatCompletion.create(
            model=base_model,
            messages=history,
            temperature=0,
            max_tokens=2048,
            stop=stop, stream=True,
            presence_penalty=0.8,
            seed=212,
            headers={
                "Helicone-Auth": f"Bearer {os.environ['HELICONE_API_KEY']}",
            },
            top_p=0.0001
    ):
        new_content = response.choices[0].delta.get("content", "")
        full_response += new_content
        with st.session_state.global_container(border=True): st.text(new_content)

    st.session_state.global_container
    return full_response

@st.cache_data(ttl=60*60)
def generate_chart_code(question: str, data_information: str, model_name: str = 'gpt-4-1106-preview'):
    PROMPT = """
[Instruction]
Given a question and a dataframe (sometimes the processing logic will also be provided to guide your solution), your task is to write Python code to visualize the data as requested. The chart must be a Plotly graph object, and it must be assigned to the variable 'result' at the end.

IMPORTANT:
- Only generate a code to draw the chart, do not apply any processing on the data, since the provided data has been preprocessed.
- If the [Processing Logic] is provided, consider it very carefully while generating the code for visualization.
- Assume that the variable 'df' exists and use this variable to draw chart in the code.
- Assign the final result to the variable 'result'.

Your answer should follow this format:
[Output]
```python
<your plotly python code>
```

[Example]
Here is an example:
<<Example 1>>
[Dataset]
The dataset 'df' has 100 rows and 2 columns.
Columns information:
    - Column 'class', datatype 'categorical'
        - Unique values: ['A', 'B', 'C']
    - Column 'average score', datatype 'numeric'
        - Value ranges: 0 - 10

[Question]
Draw a bar chart to visualize the average score of each class.

[Output]
```python
import pandas as pd
import plotly.graph_objects as go

# Create bar chart
fig = go.Figure(data=go.Bar(
    x=df['class'],
    y=df['average_score'],
    marker_color='skyblue'
))

# Update chart layout
fig.update_layout(
    title='Average Score by Class',
    xaxis=dict(title='Class'),
    yaxis=dict(title='Average Score')
)

result = fig
```
<<Example 1>>

<<Example 2>>
[Dataset]
The dataset 'df' has 100 rows and 2 columns.
Columns information:
    - Column 'source', datatype 'categorical'
        - Unique values: ['A', 'B', 'C']
    - Column 'action', datatype 'categorical'
        - Unique values: ['get_home', 'preview', 'subscribe']
    - Column 'user number', datatype 'numeric'
        - Value ranges: 0 - 10

[Logic Processing]
Funnel is following the steps: 'get_home', 'preview', 'subscribe'.

[Question]
Draw a funnel chart for each 'source'.

[Output]
```python
import pandas as pd
import plotly.graph_objects as go

# Filter the dataframe for the specific actions in the funnel
funnel_steps = ['get_home', 'preview', 'subscribe']
df_funnel = df[df['action'].isin(funnel_steps)]

# Pivot the dataframe to get the count of unique users for each action and utm_source
df_pivot = df_funnel.pivot_table(index='source', columns='action', values='user number', aggfunc='sum').fillna(0)

# Create the funnel chart for each utm_source
fig = go.Figure()

for source in df_pivot.index:
    values = df_pivot.loc[source, funnel_steps].tolist()
    fig.add_trace(go.Funnel(
        name=source,
        y=funnel_steps,
        x=values,
        textinfo="value+percent previous"
    ))

# Update chart layout
fig.update_layout(
    title='Funnel Chart by UTM Source',
    xaxis=dict(title='Number of Unique Users'),
    yaxis=dict(title='Funnel Steps')
)

result = fig
```
<<Example 2>>
---
Let's try it now.
[Dataset]
{data_information}

[Question]
{question}

[Output]
"""
    prompt_to_use = PROMPT.format(question=question, data_information=data_information)
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    chart_code = extract_python_code(response)
    return chart_code

@st.cache_data(ttl=60*60)
def generate_model_code(data_information:str, question: str, model_name: str = 'gpt-4o'):
    PROMPT = """
    [Instruction]
    Given a question and a dataframe (sometimes the processing logic will also be provided to guide your solution), your task is to write Python code to build model as requested. The model must be Scikit-learn object, and it must be assigned to the variable 'result' at the end.

    IMPORTANT:
    - If the [Processing Logic] is provided, consider it very carefully while generating the code for visualization.
    - Assume that the variable 'df' exists and use this variable to build model in the code.
    - Assign the model to the variable 'result'.

    Your answer should follow this format:
    [Output]
    ```python
    <your Scikit-learn python code>
    ```

    Let's try it now.
    [Dataset]
    {data_information}

    [Question]
    {question}

    [Output]
    """
    prompt_to_use = dedent(PROMPT).format(question=question, data_information=data_information)
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    model_code = extract_python_code(response)
    return model_code

@st.cache_data(ttl=60*60)
def modify_code(feedback: str, data_information: str, previous_code: str, code_utils:str="", model_name="gpt-3.5-turbo-1106"):
    """This function is used for modifying/refining a specific code snippet given a feedback"""
    PROMPT = """
    {code_utils}
    [Instruction]
    Given specific data information, feedback, and a previous Python code snippet, your task is to generate new Python code based on the previous code with the specified modifications provided in the feedback. Avoid generate any function in the code.
    
    Your answer should follow this format:
    [Output]
    ```python
    <your full Python code with modifications>
    ```
    
    Let's try it now.
    [Data information]
    {data_information}
    
    [Previous code]
    {previous_code}
    
    [Feedback]
    {feedback}
    
    [Output]
    """
    prompt_to_use = dedent(PROMPT).format(code_utils=code_utils, data_information=data_information, previous_code=previous_code, feedback=feedback)
    response = combine_result(previous_result=previous_code, refined_result=get_response(prompt=prompt_to_use, model_name=model_name), is_code=True)
    code = extract_python_code(response)
    if code is not None:
        return code
    st.warning(response)
    return dedent(response)

def is_request_needed(request:str, model_name:str='gpt-3.5-turbo-1106'):
    PROMPT = """
    [Instruction]
    Given a request, your task is to classify whether the request is necessary to process (1) or not (0).
    
    For example:
    1. 
    Request: "Visualization: The analysis is not needed."
    Output: 0
    
    2.
    Request: "Perform sort values to get top 5."
    Output: 1
    
    Let's try it now.
    Request: "{request}"
    Output:
    """
    prompt_to_use = dedent(PROMPT).format(request=request)
    pattern = r'\d+'
    response = get_response(prompt=prompt_to_use, model_name=model_name)
    response = [int(x) for x in re.findall(pattern, response)]
    if len(response) > 0:
        return response[0]
    else:
        response = get_response(prompt=prompt_to_use, model_name=model_name)
        response = [int(x) for x in re.findall(pattern, response)]
        return response[0]

def get_openai_embeddings(documents:List[str]):
    """Get embeddings for a list of texts using OpenAI's API."""
    embeddings = openai.Embedding.create(
        input=documents,
        engine="text-embedding-3-large"  # Choose an appropriate engine for your task
    )
    # Extract the embeddings from the response
    return np.array([embedding['embedding'] for embedding in embeddings['data']])

def retrieve_similar_example(question: str, examples: dict, nb_example:int=1):
    if examples is None:
        return None

    def normalize_text(text):
        text = text.strip().lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    q_norm = normalize_text(question)
    documents = list(examples.keys())
    try:
        question_embedding = get_openai_embeddings([q_norm])
        example_embeddings = get_openai_embeddings([normalize_text(sent) for sent in documents])
    except:
        time.sleep(3)
        question_embedding = get_openai_embeddings([q_norm])
        example_embeddings = get_openai_embeddings([normalize_text(sent) for sent in documents])

    dimension = example_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(example_embeddings)

    scores, indices = index.search(question_embedding, k=nb_example)
    if scores[0][0] > 0.75:
        example_sentence = documents[indices[0][0]]
        return examples[example_sentence]
    return None