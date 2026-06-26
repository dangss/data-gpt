from typing import Iterable, Union, List, Tuple, Any, Optional
from .chat_message import FullMessage, Message
from .agent_utils import *
from common.app_utils import table_none, table_not_exist, read_path

class Planner:
    def __init__(self, database) -> None:
        self.processing_notes = database.get("processing_notes", "")
        self.tables = database["tables"]
        self.sample_conversation = database.get("sample_conversation", None)
        self.system_message = "You are a professional consultant who answers user questions and suggests solutions."
        self.history = [] # list of Message(type, content)
        self.messages = [
            {
                "role": "system",
                "content": self.system_message
            }]
        self.current_solution = None
        self.data_information = None

    def reset_history(self):
        self.messages = [
            {"role": "system", "content": self.system_message}
        ]

    def update_plan(self, execution_plan:str, thought:str=""):
        if execution_plan != self.current_solution:
            self.current_solution = execution_plan
            instruction = "Here is my proposed execution plan. Does this plan meet your requirement?" if thought != "" else "Here is my refined plan. Does it meet your requirement?"
            messages = [
                Message(msg_type="text", content=thought),
                Message(msg_type="solution", content=execution_plan),
                Message(msg_type="text", content=instruction)
            ]
            self.history += messages
            return messages
        return None

    def run(self, message: str, message_type:str=None, relevant_tables:dict=None) -> FullMessage:
        llm_output = FullMessage(role="assistant", agent="planner")
        def get_full_tables(return_columns:bool=False)->Union[dict, list]:
            """Get all the chosen tables and columns (including 'target' and 'factor' table), along with duration of table"""
            all_tables = dict()
            for feature_type in relevant_tables.keys():
                for table in relevant_tables[feature_type].keys():
                    if table not in all_tables.keys():
                        all_tables[table] = dict()
                    if feature_type != "time_condition":
                        current_columns = all_tables.get(table, dict()).get("columns", [])
                        all_tables[table]["columns"] = current_columns + [c for c in relevant_tables[feature_type].get(table, []) if c not in current_columns]
                    else:
                        all_tables[table]["time_condition"] = relevant_tables[feature_type][table]

            if not return_columns:
                return list(all_tables.keys())
            return all_tables
        def read_table(table_name:str = None, filter_relevant_columns=True):
            if table_name is None:
                raise table_none

            if table_name not in self.tables.keys():
                raise table_not_exist(table_name=table_name, all_tables=list(self.tables.keys()))

            relevant_table_dict = get_full_tables(return_columns=True)
            if "data" not in self.tables[table_name].keys():
                duration = None
                from_date = None
                to_date = None
                if "time_condition" in relevant_table_dict[table_name].keys():
                    duration = relevant_table_dict[table_name].get("time_condition", dict()).get("duration", list(self.tables[table_name]["path"].keys())[0])
                    from_date = relevant_table_dict[table_name].get("time_condition", dict()).get("from", None)
                    to_date = relevant_table_dict[table_name].get("time_condition", dict()).get("to", None)

                data_path = self.tables[table_name]["path"] if isinstance(self.tables[table_name]["path"], str) else self.tables[table_name]["path"][duration]
                self.tables[table_name]["data"] = list(read_path(data_path, from_date=from_date, to_date=to_date).values())[0]
                for column in self.tables[table_name]["data"].columns:
                    if get_dtype_feature(self.tables[table_name]["data"][column]) == "datetime":
                        self.tables[table_name]["data"][column] = pd.to_datetime(self.tables[table_name]["data"][column])

            if filter_relevant_columns and relevant_table_dict[table_name] is not None:
                # only choose relevant columns
                return self.tables[table_name]["data"][relevant_table_dict[table_name]["columns"]]

            return self.tables[table_name]["data"]
        def get_table_schemas(table_names: List[str], include_processing_notes:bool=False) -> str:
            if len(table_names) == 0:
                return ""

            metadata_output = []
            for table_name in table_names:
                # table_duration = relevant_tables.get("time_condition", dict()).get(table_name, dict()).get("duration", "")
                # if table_duration != "":
                #     duration_str = f'\n----\nDuration: {table_duration}'
                # else:
                duration_str = ""

                processing_note_str = ""
                if include_processing_notes and self.tables[table_name].get("processing_logic", False):
                    logic_table = "\n- ".join(self.tables[table_name]["processing_logic"])
                    processing_note_str = f"\n----\n[Processing Logic]:\n{logic_table}\n----\n"
                df = read_table(table_name=table_name, filter_relevant_columns=True)
                metadata = f"Table '{table_name}'{duration_str}\n----\nPrimary keys:\n"
                for key in self.tables[table_name]["key"].keys():
                    dtype = get_dtype_feature(df[key])
                    metadata += f"\t- '{key}' (datatype: {dtype}): '{self.tables[table_name]['key'][key]}'\n"
                    metadata += f"\t\t{get_column_values(df[key], column_dtype=dtype)}\n"
                metadata += "\n----\nColumns:\n"
                for column in self.tables[table_name]["column_description"].keys():
                    if column not in df.columns: continue
                    dtype = get_dtype_feature(df[column])
                    metadata += f"\t- '{column}' (datatype: {dtype}): '{self.tables[table_name]['column_description'][column]}'\n"
                    metadata += f"\t\t{get_column_values(df[column], column_dtype=dtype)}\n"
                metadata += '\n----\n'
                metadata += f"Here are first 2 rows of data:\n{df.head(2)}\n----\n"
                metadata += f"{processing_note_str}"
                metadata_output.append(metadata)

            return "\n".join(metadata_output)

        planner_container = st.empty()
        example_plan = retrieve_similar_example(question=message, examples=self.sample_conversation)

        if example_plan is not None: example_plan = example_plan.get("execution_plan", "")
        assistant_msg = []
        if message_type in VALID_INTENTS: # user question
            if message_type == "Descriptive":
                data_information = f"You are provided {len(relevant_tables['target'].keys())} table(s).\n"
                data_information += f"\n{get_table_schemas(relevant_tables['target'].keys(), include_processing_notes=True)}"
            else: # Diagnostic; Predictive
                data_information = f"You are provided these tables:\nTARGET TABLE:\n"
                data_information += f"\n{get_table_schemas(relevant_tables['target'].keys(), include_processing_notes=True)}"

                if len(relevant_tables["factor"].items()) > 0:
                    data_information += f"\nFACTOR TABLES:\n{get_table_schemas(relevant_tables['factor'].keys(), include_processing_notes=True)}"

            data_information += "\n---\n"
            self.data_information = data_information
            sent_question = message + f"\n({self.processing_notes})"
            self.messages.append({"role": "user", "content": f'DATA INFORMATION:\n{data_information}\n\nQuestion:\n{sent_question}'}) # append to history
            execution_thought, execution_plan = generate_execution_plan(question=sent_question, data_information=data_information, question_intent=message_type,
                                                               example_plan=example_plan, model_name='gpt-4o', streaming_container=planner_container)
            plan_text = f'{execution_thought}\n\nEXECUTION PLAN:\n{execution_plan}'
            self.messages = add_assistant_msg(self.messages, new_input=f'{plan_text}\n\nDoes this execution plan meet your requirement?')
            assistant_msg = self.update_plan(execution_plan=execution_plan, thought=execution_thought)

        elif "give_feedback" in message_type: # message from user, usually feedbacks
            self.messages.append({"role": "user", "content": message})  # append to history
            execution_plan = refine_execution_plan(execution_plan=self.current_solution, feedback=message)
            plan_text = f"REFINED EXECUTION PLAN:\n{self.current_solution}"
            self.messages = add_assistant_msg(self.messages, new_input=f"{plan_text}\n\nDoes this execution plan meet your requirement?")
            assistant_msg = self.update_plan(execution_plan=execution_plan)

        elif "ask_information" in message_type:
            self.messages.append({"role": "user", "content": message})
            new_response = generate_next_message(history=self.messages)
            if isinstance(new_response, str):
                self.messages = add_assistant_msg(self.messages, new_input=new_response)
                assistant_msg = [
                    Message(msg_type="text", content=new_response)
                ]
        else:
            self.messages.append({"role": "assistant", "content": f"Final Answer:\n{self.current_solution}"})
            execution_parts = split_execution_plan(self.current_solution)
            self.current_solution = execution_parts
            assistant_msg = [
                Message(msg_type="text", content=f"Final Answer: Plan has been generated. Here is the execution plan:\n", object=self.current_solution)
            ]

        llm_output.messages = assistant_msg.copy()
        return llm_output