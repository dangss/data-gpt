import pandas as pd
import io
import sys
from functools import partial

from langchain_core.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .chat_message import FullMessage, Message
from .agent_utils import *
from common.app_utils import table_none, table_not_exist, read_path
from common.simple_model_utils import lgbm_model, statistical_model, STRONG_PREDICTOR_THRESHOLD

class ExecutionAgent:
    """This is an abstract class for agents whose tasks involve generating and executing code to obtain a specific result."""
    agent_system_message = {
        "data_agent": "You are a senior Data Engineer who works efficiently with Python to prepare useful datasets given an execution plan.",
        "vis_agent": "You are a senior Data Visualizer who works efficiently with the Python Plotly package to generate useful and beautiful charts and tables for visualization purposes.",
        "analysis_agent": "You are a senior Data Analysis who works efficiently with Python to conduct the useful analysis for a specific request",
        "model_agent": "You are a senior Data Scientist who works with the Python Scikit-learn package to build useful machine learning models for predicting and forecasting specific features."
    }
    agent_object = {
        "data_agent": "dataset",
        "vis_agent": "chart",
        "analysis_agent": "analysis",
        "model_agent": "model"
    }

    agent_code_gen_fn = {
        "vis_agent": generate_chart_code,
        "analysis_agent": lambda question, data_information, model_name: generate_code(data_information=data_information, request=question, model_name=model_name),
        "model_agent": generate_model_code
    }

    def __init__(self, agent_name:str, database:dict):
        if agent_name not in self.agent_system_message.keys():
            raise ValueError(f"Agent name '{agent_name}' is not valid. The available options are: {list(self.agent_system_message.keys())}")

        # agent and system message initialization
        self.agent_name = agent_name
        self.history = [] # list of Messages(type, content, object)
        self.messages = [{"role": "system", "content": self.agent_system_message[self.agent_name]}]

        # data information
        self.selected_tables = None
        self.tables = database['tables']

        # code and result
        self.current_solution = None
        self.current_result = None

        # util_functions
        self.util_functions = ""
        self.env_locals = None

        # Tools to select
        self.tool_agent = None
        self.function_map = None

    def reset_history(self):
        self.history = []
        self.messages = [{"role": "system", "content": self.agent_system_message[self.agent_name]}]

    def select_tool(self, data_information:str, request:str):
        if self.tool_agent is None: return None
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", dedent(f"Given this data: {data_information}\nWhich functions and columns the most appropriate to the query?")),
                ("human", "{input}")
            ]
        ).format(input=request)
        return self.tool_agent.invoke(prompt).additional_kwargs.get("function_call", None)

    def read_table(self, table_name: str = None, filter_relevant_columns=True):
        if table_name is None:
            raise table_none

        if table_name not in self.tables.keys():
            raise table_not_exist(table_name=table_name, all_tables=list(self.tables.keys()))

        if 'data' not in self.tables[table_name].keys():
            duration = None
            from_date = None
            to_date = None
            if table_name in self.selected_tables and 'time_condition' in self.selected_tables[table_name].keys():
                duration = self.selected_tables[table_name].get('time_condition', dict()).get('duration', list(self.tables[table_name]['path'].keys())[0])
                from_date = self.selected_tables[table_name].get('time_condition', dict()).get('from', None)
                to_date = self.selected_tables[table_name].get('time_condition', dict()).get('to', None)

            data_path = self.tables[table_name]['path'] if isinstance(self.tables[table_name]['path'], str) else self.tables[table_name]['path'][duration]
            self.tables[table_name]['data'] = list(read_path(data_path, from_date=from_date, to_date=to_date).values())[0]
            for column in self.tables[table_name]['data'].columns:
                if get_dtype_feature(self.tables[table_name]['data'][column]) == 'datetime':
                    self.tables[table_name]['data'][column] = pd.to_datetime(self.tables[table_name]['data'][column])

        if filter_relevant_columns and self.selected_tables[table_name] is not None:
            # only choose relevant columns
            return self.tables[table_name]['data'][self.selected_tables[table_name]['columns']]

        return self.tables[table_name]['data']

    def get_table_schemas(self, table_names: List[str], include_processing_notes: bool = False) -> str:
        if len(table_names) == 0:
            return ""

        metadata_output = []
        for table_name in table_names:
            # table_duration = self.selected_tables.get("time_condition", dict()).get(table_name, dict()).get("duration", "")
            # if table_duration != "":
            #     duration_str = f'\n----\nDuration: {table_duration}'
            # else:
            duration_str = ""

            processing_note_str = ""
            if include_processing_notes and self.tables[table_name].get('processing_logic', False):
                logic_table = '\n- '.join(self.tables[table_name]["processing_logic"])
                processing_note_str = f'\n----\n[Processing Logic]:\n{logic_table}\n----\n'
            df = self.read_table(table_name=table_name, filter_relevant_columns=True)
            metadata = f'Table \'{table_name}\'\n{duration_str}\n----\nPrimary keys:\n'
            for key in self.tables[table_name]["key"].keys():
                dtype = get_dtype_feature(df[key])
                metadata += f'\t- "{key}" (datatype: {dtype}): "{self.tables[table_name]["key"][key]}"\n'
                metadata += f"\t\t{get_column_values(df[key], column_dtype=dtype)}\n"
            metadata += '\n----\nColumns:\n'
            for column in self.tables[table_name]["column_description"].keys():
                if column not in df.columns: continue
                dtype = get_dtype_feature(df[column])
                metadata += f'\t- "{column}" (datatype: {dtype}): "{self.tables[table_name]["column_description"][column]}"\n'
                metadata += f"\t\t{get_column_values(df[column], column_dtype=dtype)}\n"
            metadata += '\n----\n'
            metadata += f"Here are first 2 rows of data:\n{df.head(2)}\n----\n"
            metadata += f"{processing_note_str}"
            metadata_output.append(metadata)

        return "\n".join(metadata_output)

    def get_data_information(self, data: Union[pd.DataFrame, dict]):
        if isinstance(data, pd.DataFrame):
            data_information = get_column_info(df=data).replace('Dataset has', 'Dataset "df" has')
            processing_logic = ""
            for table in self.selected_tables:
                processing_logic += '\n- '.join(self.tables[table].get('processing_logic', []))
            processing_logic = processing_logic.strip()
            if processing_logic != '': processing_logic = f"\n[Processing Logic]\n{processing_logic}\n"
            data_information += processing_logic
        else:
            # table information
            data_information = "You are provided these tables:\n"
            for key, tables in data.items():
                if len(tables.items()) > 0:
                    data_information += f"{key.upper()} TABLES:\n"
                    data_information += f"{self.get_table_schemas(table_names=list(tables.keys()))}"
        return data_information

    def get_all_features(self, keys: list = None):
        """Get all factors of a specific keys"""
        if keys is None:
            raise ValueError("Keys cannot be empty. Please specifieid keys.")

        provided_key = set(keys)
        table_to_merge = []
        for table in self.tables.keys():
            if table not in table_to_merge and set(self.tables[table]['key'].keys()) == provided_key:
                table_to_merge.append(table)
        if len(table_to_merge) > 0:
            merged_table = merge_tables([self.read_table(tbl, filter_relevant_columns=False) for tbl in table_to_merge], keys=keys)
            return merged_table

        return None

    def get_last_result(self):
        if len(self.history) > 0:
            for idx in range(len(self.history)-1, -1, -1):
                if self.history[idx].object is not None:
                    return self.history[idx].object
        return None

    def run_code(self, execution_code:str, data_information:str=None, original_dataset:pd.DataFrame=None, success_code:str=None):
        """
        Run code without changing success_code
        """
        if len(self.history) > 0 and execution_code == self.history[-1].content:
            return None

        code_messages = [Message(msg_type="solution", content=execution_code)]
        run_ok = False
        err_count = 0

        if original_dataset is not None:
            self.env_locals['df'] = original_dataset

        if data_information is None:
            data_information = self.get_data_information(data=original_dataset)

        while not run_ok and err_count < 5:
            try:
                exec(f'import pandas as pd\n{dedent(execution_code)}', self.env_locals)
                run_ok = True
                code_messages[-1].object = copy.deepcopy(self.env_locals['result'])
            except:
                error_message = format_exc()
                code_messages[-1].object = error_message
                with st.spinner('There is an error occurred. Try to fix.'):
                    success_part=""
                    if success_code is not None:
                        success_part = f"""This part of code has been run successfully, do not modify:\n```python\n{success_code}\n```"""
                    print("DATA INFORMATION TO FIX")
                    print(data_information)

                    print("PREVIOUS CODE")
                    print(execution_code)

                    feedback = f'The execution of the above code returned this error: \n{error_message}\n--\n{success_part}\nFix the error.\n'
                    print(feedback)
                    execution_code = modify_code(
                        feedback=feedback,
                        data_information=data_information, previous_code=execution_code, code_utils=self.util_functions)

                code_messages += [Message(msg_type="text", content="Modified code:"), Message(msg_type="solution", content=execution_code)]
                run_ok = False
                err_count += 1

        if not run_ok:
            code_messages += [Message(msg_type="text", content=f"Failed to create {self.agent_object[self.agent_name]} due to some errors. Could you help me to investigate the error? Otherwise, you can try new question.")]
        else:
            code_messages += [Message(msg_type="text", content=f"Does this result meet your requirement?")]

        self.history += code_messages
        for message in code_messages:
            message.show(editable=False)

        self.env_locals.update(locals())

        return code_messages

    def run(self, message: str, message_type:str, selected_tables:dict=None, dataset:pd.DataFrame=None) -> FullMessage:
        final_message = FullMessage(role="assistant", agent=self.agent_name)

        if self.env_locals is None:
            self.env_locals = locals()

        # define function
        self.env_locals["read_table"] = self.read_table
        self.env_locals["get_all_features"] = self.get_all_features
        self.env_locals["merge_tables"] = merge_tables

        if self.agent_name in self.agent_code_gen_fn and self.agent_code_gen_fn[self.agent_name] is None:
            final_message.messages = [Message(msg_type="text", content="Final Answer: This agent is not ready for execution.")]
            return final_message

        if selected_tables is not None and self.selected_tables is None:
            self.selected_tables = get_selected_tables(selected_tables=selected_tables, return_columns=True)

        if dataset is not None:
            data_information = self.get_data_information(data=dataset)
        else:
            data_information = self.get_data_information(data=selected_tables)

        sent_msg = message
        messages = []
        if message_type == "agree" and len(self.history) == 0 :
            messages += [Message(msg_type="text", content=f"Final Answer: No request to process.")]
            return final_message

        if is_request_needed(request=sent_msg) == 0 and self.get_last_result() is None:
            final_message.messages = [Message(msg_type="text", content="Currently, there is no specific request to process. What would you like to do on this dataset?", object=dataset)]
            return final_message

        if dataset is None and self.agent_name != "data_agent":
            final_message.messages = [Message(msg_type="text", content="Final Answer: No data to process.")]
            return final_message

        if message_type != "agree": # need to create / modify code => execution
            code_messages = []
            if message_type in VALID_INTENTS or len(self.history) == 0: # first request
                if len(self.history) == 0:
                    self.history.append(Message(msg_type="text", content="Initial request:", object=message))

                tasks = plan_to_tasks(execution_plan=message)
                execution_code = ""
                for idx, task in enumerate(tasks):
                    self.history += [Message(msg_type="text", content=f"<b>{task}</b>")]
                    task_words = task.replace("\n", "<br>")
                    st.markdown(f"<p><b>{task_words}</b></p>", unsafe_allow_html=True)
                    if self.agent_name == "data_agent":
                        if idx == 0:
                            execution_code += generate_code(data_information=data_information, request=f"Generate code to solve this task:\n'''\n{task}\n'''\n* Assign the result to variable 'result'.\n", code_utils=self.util_functions) + "\n"
                        else:
                            past_steps = "\n".join(tasks[:idx])
                            request = f"The data information above is generated through these steps:\n'''{past_steps}\n'''\nHere is the current code:\n```python\n{self.history[-1].content}\n```\nPlease follow these instructions:\n- Continue writing code to solve the step {idx+1}: '''{task}'''\n- If the step is not required, or it has been processed in the previous code, simply return previous code;\n- Assign the result to variable 'result'.\n"
                            execution_code += generate_code(data_information=data_information, request=request, code_utils=self.util_functions, model_name="gpt-4o") + "\n"
                    else:
                        # Agent that is not data_agent always has data 'df' to process.
                        sent_task = f"{task} (Assume that 'df' is pre-defined).\n* Assign the result to variable 'result'.\n"
                        selected_tool = self.select_tool(data_information=data_information, request=sent_task)
                        if (selected_tool is None) or (selected_tool["name"] not in self.function_map):
                            if self.agent_name != "model_agent":
                                execution_code = self.agent_code_gen_fn[self.agent_name](data_information=data_information, question=sent_task, model_name="gpt-4-1106-preview")
                            else:
                                execution_code = ""
                        else:
                            function_name = selected_tool["name"]
                            function_args = json.loads(selected_tool["arguments"])
                            try:
                                result = self.function_map[function_name](df=dataset, **function_args)
                                code_messages = [Message(msg_type="solution", content=f"{selected_tool}", object=result)]
                                self.history += code_messages
                            except:
                                error_msg = format_exc()
                                print(error_msg)

                    success_code = None
                    if idx > 0:
                        success_code = self.history[-1].content

                    if execution_code != "":
                        code_messages = self.run_code(execution_code=execution_code, data_information=data_information, original_dataset=dataset, success_code=success_code)
                        buffer = io.StringIO()
                        sys.stdout = buffer
                        if isinstance(self.env_locals["result"], pd.DataFrame):
                            print("Current data:")
                            print(self.get_data_information(data=self.env_locals["result"]))
                        else:
                            print("Current result")
                            print(self.env_locals["result"])
                        data_information = buffer.getvalue()
                        sys.stdout = sys.__stdout__
            else:
                sent_msg = f"{message}\n* Assign the result to variable 'result'.\n"
                execution_code = modify_code(feedback=sent_msg, data_information=data_information, previous_code=self.history[-1].content, code_utils=self.util_functions, model_name='gpt-4-1106-preview')
                if self.agent_name == "data_agent":
                    current_dataset = self.history[-1].object
                else:
                    current_dataset = dataset.copy()
                code_messages = self.run_code(execution_code=execution_code, data_information=data_information, original_dataset=current_dataset)

            messages += code_messages
        else:
            messages += [Message(msg_type="text", content=f"Final Answer: I have created {self.agent_object[self.agent_name]} at the request. Here is the result:", object=self.get_last_result())]

        final_message.messages = messages.copy()
        return final_message

class DataAgent(ExecutionAgent):
    def __init__(self, database:dict) -> None:
        super(DataAgent, self).__init__(agent_name="data_agent", database=database)
        self.util_functions = dedent("""
        << UTILS >>
        1. read_table(table_name:str): read the table given its name.
        2. get_all_features(keys:list): get all the features in the database given a key set.
        3. merge_tables(tables:List[pd.DataFrame], keys:List[str]): merge the datasets on specific keys.
        """)

class AnalysisAgent(ExecutionAgent):
    def __init__(self, database:dict) -> None:
        super(AnalysisAgent, self).__init__(agent_name="analysis_agent", database=database)

class VisAgent(ExecutionAgent):
    def __init__(self, database:dict) -> None:
        super(VisAgent, self).__init__(agent_name="vis_agent", database=database)

class ModelAgent(ExecutionAgent):
    def __init__(self, database:dict) -> None:
        super(ModelAgent, self).__init__(agent_name="model_agent", database=database)

        @tool
        def build_explain_model(target_feature: str):
            """Build a LightGBM model to explain the 'target_feature' (default option for explanation)"""
            pass

        @tool
        def build_predict_model(target_feature: str):
            """Build a LightGBM model to predict the 'target_feature' (default option for prediction)"""
            pass

        @tool
        def build_custom_model(request: str):
            """Build a model given a specific request"""
            pass

        tools = [build_explain_model, build_predict_model, build_custom_model]
        self.function_map = {
            "build_explain_model": partial(lgbm_model, is_explain=True),
            "build_predict_model": partial(lgbm_model, is_explain=False),
            "build_custom_model": None
        }
        self.tool_agent = ChatOpenAI(model="gpt-4o-mini").bind_functions(functions=[
            format_tool_to_openai_function(t) for t in tools
        ])