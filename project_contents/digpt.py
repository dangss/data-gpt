from copy import deepcopy
from dotenv import load_dotenv
from streamlit_tree_select import tree_select
import os
from common.di_config import INTERNAL_PROXY

load_dotenv("env")
# os.environ["http_proxy"]=INTERNAL_PROXY
# os.environ["https_proxy"]=INTERNAL_PROXY

from common.app_utils import *
from agents.chat_message import FullMessage, Message, init_msg, not_supported_msg, display_all_msg
from common.display_utils import show_dictionary, show_model, show_tables, show_data, show_analysis
from common.chart_utils import *
from agents.agent_flow import *
import agents.state as agent_state
from agents.agent_utils import get_question_intent, get_response_type, suggest_time_range, VALID_INTENTS

# SPARK #
from pyspark.sql import SparkSession

# Open AI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

page_prepare("Talk to your data")
AGENTS = ["planner", "data_agent", "vis_agent", "analysis_agent", "model_agent"]
EXECUTION_PROCESS = [data_visualization, data_analysis, data_modeling]
state_show_object = {
    data_preparation: show_data,
    data_analysis: show_analysis,
    data_visualization: lambda result: st.plotly_chart(result),
    data_modeling: show_model
}

## initialize database
if "full_databases" not in st.session_state:
    st.session_state.full_databases = AppData()

## initialize spark
if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder.appName("DI-GPT") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

    # st.session_state.spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

## initialize messages
if "continue_conversation" not in st.session_state:
    st.session_state.continue_conversation = False

if "messages" not in st.session_state:
    # reset conversation #
    st.session_state.messages = {"INTRO": {"state_history": [init_msg]}}
    agent_state.init_states()

# initial other variables
if "user_question" not in st.session_state:
    # Current User Question in Each Conversation: Distinguish from Other User Instructions / Feedbacks. #
    st.session_state.user_question = {
        "content": None,
        "question_intent": None,
        "question_features": dict(), # features of question (chosen by user)
        "full_process": PROCESS(), # process to solve the question
    }

if "process_shown" not in st.session_state:
    st.session_state["process_shown"] = False

if "global_container" not in st.session_state:
    st.session_state.global_container = st.empty()

## Chat UI ##
def renew_conversation(new_question:str=None):
    del st.session_state["messages"]
    for agent in AGENTS:
        del st.session_state[agent]

    st.session_state.user_question = {
        "content": new_question,
        "question_intent": None,
        "question_features": dict(), # features of question (chosen by user
        "full_process": PROCESS(),  # process to solve the question
    }

    agent_state.init_states()
    st.session_state.continue_conversation = new_question is not None
    if st.session_state.continue_conversation:
        st.rerun()

def show_suggestion():
    with st.container(border=True):
        st.caption("Suggestion")
        suggestion_cols = st.columns(len(agent_state.SUGGEST_RESPONSE.items()))
        for idx, (response_key, response_value) in enumerate(agent_state.SUGGEST_RESPONSE.items()):
            with suggestion_cols[idx]:
                st.button(response_value, key=response_key) # simply assign key
        st.caption('Or')
        st.button(agent_state.END_PROCESS_STATE["value"], help=agent_state.END_PROCESS_STATE["description"], key=agent_state.END_PROCESS_STATE["key"])

def show_history(process_state, state_history):
    toggle_history = st.toggle(label=f"Show history of {process_state}", value=False,
                               key=f"history_state_{process_state}")
    if toggle_history:
        if len(state_history) > 0:
            with st.container(border=True):
                display_all_msg(msg_list=state_history)

def show_next_process():
    with st.container(border=True):
        st.caption("Which process do you want to continue?")
        process_cols = st.columns(len(EXECUTION_PROCESS))
        for idx, step_name in enumerate(EXECUTION_PROCESS):
            if step_name == data_modeling and st.session_state.user_question["question_intent"]["category"] == "Descriptive": continue
            step = st.session_state.user_question["full_process"].get_step_by_name(step_name=step_name)
            with process_cols[idx]:
                step_code = step_name.lower().replace(" ", "_")
                if step is not None and step.is_done:
                    st.button(step_name, key=step_code, disabled=True)
                else:
                    st.button(step_name, key=step_code, disabled=False)
        st.caption('Or')
        st.button(agent_state.END_PROCESS_STATE["value"], help=agent_state.END_PROCESS_STATE["description"], key=agent_state.END_PROCESS_STATE["key"])
@st.experimental_fragment
def show_state_messages(state_messages):
    for idx, process_state in enumerate(state_messages.keys()):
        if process_state == "INTRO": continue
        state_result = state_messages[process_state]["state_result"]
        state_history = state_messages[process_state]["state_history"]

        process_state_key = process_state.replace(" ", "-").lower()
        st.markdown(f"<h3 id='{process_state_key}'>{process_state}</h3>", unsafe_allow_html=True)
        if process_state not in [table_selection, time_filtering]:
            show_history(process_state=process_state, state_history=state_history)

        if state_result is not None:
            st.write(f"✅ Current result of state {process_state}")
            if process_state == table_selection:
                for feature_type in state_result.keys():
                    st.subheader(feature_type.upper())
                    show_tables(state_result[feature_type])
            elif process_state == time_filtering:
                for table in state_result.keys():
                    with st.expander(f"Table {table}", expanded=False):
                        st.markdown(f"<b>Duration:</b> {state_result[table]['duration']}", unsafe_allow_html=True)
                        st.markdown(f"<b>Time range:</b> {state_result[table]['from'].strftime('%Y/%m/%d')} => {state_result[table]['to'].strftime('%Y/%m/%d')}", unsafe_allow_html=True)
            elif process_state == plan_generation:
                show_dictionary(result=state_result["result"], title="Execution Plan")
            elif process_state in state_show_object.keys():
                if "text" in state_result.keys():
                    st.success(state_result["text"])
                if state_result.get("result", None) is not None:
                    state_show_object[process_state](state_result["result"])
        st.divider()

def handle_end_process():
    if "end_process" in st.session_state and st.session_state.end_process:
        renew_conversation(new_question=None)
        del st.session_state.end_process
        st.rerun()

def use_suggest_response():
    handle_end_process()
    for response_key, response_value in agent_state.SUGGEST_RESPONSE.items():
        if response_key in st.session_state and st.session_state[response_key]:
            del st.session_state[response_key]
            return response_value
    return False

def complete_current_step(next_step:str=None):
    print(f"{st.session_state.user_question['full_process'].current_step.step_name} done.")
    st.session_state.user_question["full_process"].current_step.is_done = True
    st.session_state.continue_conversation = True

    # show result
    if current_step.step_name in state_show_object.keys():
        state_show_object[current_step.step_name](task_to_agent[current_step.step_name].get_last_result())

    if next_step is None:
        st.session_state.user_question["full_process"].set_current_step(None)
    else:
        next_step = st.session_state.user_question["full_process"].get_step_by_name(step_name=next_step)
        st.session_state.user_question["full_process"].set_current_step(next_step)
        st.session_state.user_question["full_process"].current_step.is_done = False

    st.session_state["process_shown"] = False
    st.rerun()

def update_current_step():
    handle_end_process()
    # for step in st.session_state.user_question['full_process'].process_steps:
    for step_name in EXECUTION_PROCESS:
        step_code = step_name.lower().replace(" ", "_")
        if step_code in st.session_state and st.session_state[step_code]:
            curr_step = st.session_state.user_question["full_process"].get_step_by_name(step_name=step_name)
            if curr_step is None:
                st.warning("This step is not included in execution plan.")
                with st.spinner("Initializing new step"):
                    curr_step = PROCESS_STEP(step_name=step_name)
                    st.session_state.user_question["full_process"].add_step(step=curr_step)
                    time.sleep(2)

            del st.session_state[step_code]
            curr_step.is_done = False
            st.session_state.user_question["full_process"].set_current_step(curr_step)
            return curr_step.step_name

    return False
@st.experimental_fragment
def show_new_messages(current_step:PROCESS_STEP):
    """show new generated messages with editable mode"""
    used_agent = task_to_agent.get(current_step.step_name, None)
    if used_agent is None: return

    if current_step.step_name != plan_generation: # execution agent => run code
        if "newest_code" in st.session_state and st.session_state["newest_code"] is not None:
            new_messages = used_agent.run_code(execution_code=st.session_state["newest_code"]["text"],
                                               original_dataset=st.session_state.data_agent.get_last_result())
            if new_messages is not None: # code is executed
                new_messages = [Message(msg_type="text", content="Modify code")] + new_messages.copy()
                st.session_state.messages[current_step.step_name]["state_history"][-1].messages += new_messages.copy()
                st.rerun()
    else:
        if "newest_plan" in st.session_state and st.session_state["newest_plan"] is not None:
            new_messages = used_agent.update_plan(execution_plan = st.session_state["newest_plan"]["text"])
            if new_messages is not None:
                new_messages[-2].is_modified = True
                new_messages = [Message(msg_type="text", content="Modify plan")] + new_messages.copy()
                st.session_state.messages[current_step.step_name]["state_history"][-1].messages += new_messages.copy()
                st.rerun()

    messages = st.session_state.messages[current_step.step_name]["state_history"][-1].messages[-2:]
    for idx, msg in enumerate(messages):
        if msg.msg_type == "solution":
            if msg.content == used_agent.history[-2].content: # last solution
                msg.show(editable=True, language="text" if current_step.step_name == plan_generation else "python")
            else:
                msg.show()
        else:
            msg.show()

# show current state messages
display_all_msg(st.session_state.messages["INTRO"]["state_history"])

if "user_question" in st.session_state and st.session_state.user_question["question_intent"] is not None:
    with st.expander(f"Your question is classified as {st.session_state.user_question['question_intent']['category']} question. Click to expand and view the details.", expanded=False):
        st.write(st.session_state.user_question["question_intent"]["explanation"])

show_state_messages(st.session_state.messages)

## app flow
full_databases = st.session_state.full_databases
db_name = st.sidebar.selectbox("Choose a database:", options=full_databases.database_options, index=0)
reinit_agent = False
if db_name is None:
    st_warning("Choose a database to start your analysis")
else:
    if "db_name" not in st.session_state or db_name != st.session_state.db_name:
        st.session_state["db_name"] = db_name
        full_databases.load_database(db_name=db_name)
        reinit_agent = True

    ## initialize agents
    for agent in AGENTS:
        if (agent not in st.session_state) or reinit_agent:
            st.session_state[agent] = init_agent(agent_type=agent, database=full_databases.current_database)
    reinit_agent = False
    ## planner ##
    planner = st.session_state.planner

    ## Execution Agents ##
    data_agent = st.session_state.data_agent
    vis_agent = st.session_state.vis_agent
    analysis_agent = st.session_state.analysis_agent
    model_agent = st.session_state.model_agent

    # mapping task to agent
    task_to_agent = {
        plan_generation: planner,
        data_preparation: data_agent,
        data_visualization: vis_agent,
        data_analysis: analysis_agent,
        data_modeling: model_agent
    }

    ## tables ##
    tables = full_databases.current_database["tables"]
    st.sidebar.link_button("View metadata catalog", "/Metadata")
    if st.session_state["user_question"]["content"] is None:  # init the conversation with user having no idea to ask
        if db_name == "00_Mekong_Delta":
            ideas = [
                "Can you provide the GRDP for all Mekong Delta provinces over the years from 2018 to 2022?",
                # "Can you provide the GRDP breakdown by economic zone for all Mekong Delta provinces over the years?",
                "Could you give me the unemployment rate of each province from 2018 to 2022?", # Describe
                # "Is there any relationship between average income and the population size of each province?", # Relationship
                "What are the most important factors that can help increase GRDP of a province?",
                # "Can marital-related factors explain average income of each province for the period 2017 to 2022?"
            ]
        elif db_name == "01_RBT":
            ideas = [
                "Visualize the user funnel of each source in April 10",
                "Provide information about total revenue of each package, breaking down by source, from April 23 to April 30",
                "In the last week of April, how many new and renew user in each day?",
                # "Provide the daily CR of each source",
                # 'Are there any differences between users from iOS and Android?'
            ]
        else:
            ideas = []

        with st.container(border=True):
            st.write("Suggestion")
            for idx, idea in enumerate(ideas):
                if idx%2 == 0:
                    new_row = st.columns(2)

                with new_row[idx%2]:
                    if st.button(idea):
                        renew_conversation(new_question=idea)
    if update_current_step():
        st.session_state.continue_conversation = True

    current_step = st.session_state.user_question["full_process"].current_step
    if current_step is not None and current_step.step_name in task_to_agent.keys() and current_step.step_name in st.session_state.messages.keys():
        st.session_state.user_question["full_process"].show_process()
        st.session_state['process_shown'] = True
        if 'ok' not in st.session_state or not st.session_state['ok']:
            show_new_messages(current_step=current_step)
            show_suggestion()

    if user_prompt := st.chat_input("Ask about the data") or st.session_state.continue_conversation or use_suggest_response():
        if isinstance(user_prompt, str) or (st.session_state.continue_conversation and (st.session_state.user_question["question_intent"] is None or st.session_state.user_question["full_process"].current_step is None)):
            if isinstance(user_prompt, str):
                user_prompt = user_prompt.strip()

            if not isinstance(user_prompt, str):
                user_prompt = st.session_state.user_question["content"] # from the previous session

            user_msg = FullMessage(role="user", messages=[Message(msg_type="text", content=user_prompt)])
            # add user message to the latest msg
            last_process_state = list(st.session_state.messages.keys())[-1]
            if st.session_state.user_question["question_intent"] is None or st.session_state.user_question["content"] != user_prompt:
                st.session_state.messages[last_process_state]["state_history"].append(user_msg)
                with st.chat_message("user"):
                    usr_container = st.empty()
                    user_msg.show(usr_container)

            if st.session_state.user_question["content"] is None:
                st.session_state.user_question["content"] = user_prompt

        if st.session_state.user_question["question_intent"] is None:  # get information of user question
            question_intent_result = get_question_intent(st.session_state.user_question["content"])
            if question_intent_result[1] in VALID_INTENTS:
                st.session_state.user_question["question_intent"] = {"category": question_intent_result[1], "explanation": question_intent_result[0]}
                question_intent = st.session_state.user_question["question_intent"]["category"]
                st.session_state.user_question["full_process"].init_selector_steps(include_time_filtering=st.session_state.db_name=="01_RBT")
                st.session_state.user_question["full_process"].set_current_step(st.session_state.user_question["full_process"].get_step_by_name(step_name=table_selection))
                if len(st.session_state.user_question["full_process"].process_steps) > 0:
                    with st.expander(
                            f"Your question is classified as {st.session_state.user_question['question_intent']['category']} question. Click to expand and view the details.",
                            expanded=False):
                        st.write(st.session_state.user_question["question_intent"]["explanation"])

        question_intent = st.session_state.user_question["question_intent"]
        question_content = st.session_state.user_question["content"]

        if (question_intent is not None and question_intent["category"] not in ["Predictive", "Others"]):
            current_step = st.session_state.user_question["full_process"].current_step
            if not st.session_state["process_shown"]:
                st.session_state.user_question["full_process"].show_process()

            if current_step is None:
                show_next_process()
            else:
                st.session_state.continue_conversation = False

                if current_step.step_name not in st.session_state.messages:
                    st.session_state.messages[current_step.step_name] = {"state_history": [], "state_result": None}
                    sent_msg = question_content
                else:
                    sent_msg = user_prompt

                with st.chat_message("assistant"):
                    msg_type = None
                    if sent_msg != st.session_state.user_question["content"]:
                        msg_type = get_response_type(message=sent_msg)
                    else:
                        msg_type = question_intent["category"]

                    st.subheader(current_step.step_name)
                    if current_step.step_name == table_selection:
                        relevant_tables = full_databases.search_question(question=sent_msg, split_info=st.session_state.user_question["question_intent"]["category"] == "Diagnostic")
                        if len(relevant_tables) > 0:
                            shown_tables = {}
                            for information_type in ["target", "factor"]:
                                shown_tables[information_type] = {}
                                segment_relevant_tables = relevant_tables[relevant_tables["information_type"] == information_type]
                                for info in segment_relevant_tables["information"].unique():
                                    filter_relevant_tables = segment_relevant_tables[segment_relevant_tables["information"] == info]
                                    shown_tables[information_type][info] = {
                                            row["table"]: {
                                            "high_similar_columns": [col for col in row["high_similar_columns"] if col != "table_description"],
                                            "medium_similar_columns": [col for col in row["relevant_information"] if col not in row["high_similar_columns"] and col != "table_description"],
                                            "others": sorted([col for col in full_databases.current_database["tables"][row["table"]]["column_description"].keys() if col not in row["relevant_information"]])
                                        } for idx, row in filter_relevant_tables.iterrows()
                                    }
                            with st.form("table_selection_form"):
                                st.success("These tables appear to be useful to your question. Please select the tables and columns you expect to use to continue the analysis. Expand the table to view its columns.")
                                for feature_type in ["target", "factor"]:
                                    data_tree = [
                                        {
                                            "value": info,
                                            "label": f'{info}',
                                            "children": [{
                                                    "value": f"{info}|{table}",
                                                    "label": f"Table {table}; primary keys (pre-selected columns): {list(full_databases.current_database['tables'][table]['key'].keys())}",
                                                    "children": ([
                                                        {
                                                            "value": f"{info}|{table}|high_similar_columns",
                                                            "label": f"High Similar Columns ({len(shown_tables[feature_type][info][table]['high_similar_columns'])} features)",
                                                            "children": [
                                                                {
                                                                    "value": f"{info}|{table}|{column}",
                                                                    "label": column
                                                                }
                                                                for column in shown_tables[feature_type][info][table]["high_similar_columns"]
                                                            ]
                                                        }] if len(shown_tables[feature_type][info][table]["high_similar_columns"]) > 0 else []) + \
                                                        ([{
                                                            "value": f"{info}|{table}|medium_similar_columns",
                                                            "label": f"Medium Similar Columns ({len(shown_tables[feature_type][info][table]['medium_similar_columns'])} features)",
                                                            "children": [
                                                                {
                                                                    "value": f"{info}|{table}|{column}",
                                                                    "label": column
                                                                }
                                                                for column in shown_tables[feature_type][info][table]["medium_similar_columns"]
                                                            ]
                                                        },
                                                    ] if len(shown_tables[feature_type][info][table]["medium_similar_columns"]) > 0 else []) + \
                                                        ([{
                                                            "value": f"{info}|{table}|others",
                                                            "label": f"Other Columns ({len(shown_tables[feature_type][info][table]['others'])} features)",
                                                            "children": [
                                                                {
                                                                    "value": f"{info}|{table}|{column}",
                                                                    "label": column
                                                                }
                                                                for column in shown_tables[feature_type][info][table]["others"]
                                                            ]
                                                        },
                                                    ] if len(shown_tables[feature_type][info][table]["others"]) > 0 else [])
                                                } for table in shown_tables[feature_type][info].keys()
                                            ]
                                        } for info in shown_tables[feature_type].keys()
                                    ]
                                    st.session_state[f"{feature_type}_features"] = {"checked": []}
                                    if len(data_tree) > 0:
                                        st.subheader(f"Choosing data for {feature_type.upper()}")
                                        feature_note = "the features" if st.session_state.user_question["question_intent"]["category"] == "Descriptive" or feature_type == "factor" else "ONE feature"
                                        st.caption(f"You should choose {feature_note} about: \"{'; '.join(list(shown_tables[feature_type].keys()))}\"")
                                        # info_to_pre_check = list(shown_tables[feature_type].keys())
                                        # for info in info_to_pre_check:
                                        #     first_table = list(shown_tables[feature_type][info].keys())[0]
                                        #     last_index = 10 if feature_type == 'factor' else 1
                                        #     st.session_state[f'{feature_type}_features']['checked'] += [f'{info}|{first_table}|{column}' for column in shown_tables[feature_type][info][first_table]['high_similar_columns'][:last_index]]
                                        tree_select(
                                            data_tree[0]["children"] if len(data_tree) <= 1 else data_tree,
                                            # checked=st.session_state[f'{feature_type}_features']['checked'],
                                            expanded=list(shown_tables[feature_type].keys()) if len(data_tree) <= 1 else None,
                                            key=f"{feature_type}_features"
                                        )
                                        st.divider()

                                def complete_table_selection():
                                    chosen_tables = {"target": dict(), "factor": dict()}
                                    for feature_type in chosen_tables.keys():
                                        for item in st.session_state[f"{feature_type}_features"]["checked"]:
                                            components = item.split("|")
                                            if len(components) == 3 and components[-1] not in ["high_similar_columns", "medium_similar_columns", "others"]: # format info|table|<column_name>
                                                table_columns = chosen_tables[feature_type].get(components[1], [])
                                                chosen_tables[feature_type][components[1]] = table_columns + ([components[-1]] if components[-1] not in table_columns else [])

                                        # add keys
                                        for table in chosen_tables[feature_type].keys():
                                            chosen_tables[feature_type][table] = list(full_databases.current_database["tables"][table]["key"].keys()) + chosen_tables[feature_type][table]

                                    agent_state.set_local_tables("current_tables", chosen_tables)
                                    st.session_state.messages[current_step.step_name]["state_result"] = deepcopy(agent_state.get_local_tables("current_tables"))
                                    current_step.is_done = True
                                    next_step = st.session_state.user_question["full_process"].get_step_by_name(step_name=time_filtering)
                                    if next_step is None:
                                        next_step = st.session_state.user_question["full_process"].get_step_by_name(step_name=plan_generation)
                                    st.session_state.user_question["full_process"].set_current_step(step=next_step)
                                    st.session_state.continue_conversation = True  # auto go to next step when current step is done
                                st.form_submit_button("Submit", help="Click this button after obtaining the expected tables.", on_click=complete_table_selection)
                                st.form_submit_button(agent_state.END_PROCESS_STATE["value"], help=agent_state.END_PROCESS_STATE["description"], on_click=renew_conversation)
                        else:
                            st.warning("No relevant information.")
                            renew_conversation()
                    if current_step.step_name == time_filtering:
                        chosen_tables = agent_state.get_local_tables("current_tables")
                        all_table_names = []
                        for feature_type in chosen_tables.keys():
                            all_table_names += [table for table in chosen_tables[feature_type].keys() if table not in all_table_names]

                        def add_time_filter(table_list:list):
                            chosen_tables["time_condition"] = dict()
                            for table in table_list:
                                chosen_tables["time_condition"][table] = {
                                    "duration": st.session_state[f"{table}_duration"],
                                    "from": st.session_state[f"{table}_from"],
                                    "to": st.session_state[f"{table}_to"]
                                }
                            agent_state.set_local_tables("current_tables", chosen_tables)
                            st.session_state.messages[current_step.step_name]["state_result"] = deepcopy(chosen_tables["time_condition"])
                            current_step.is_done = True
                            st.session_state.user_question["full_process"].set_current_step(step=st.session_state.user_question["full_process"].get_step_by_name(step_name=plan_generation))
                            st.session_state.continue_conversation = True

                        with st.form("Time Range Filtering"):
                            time_range = "'folder_date': 2024-04-01 - 2024-04-30"
                            from_time, to_time = suggest_time_range(question=st.session_state.user_question["content"], time_range=time_range)
                            st.info("For each table below, please select your expected time range and duration.")
                            for idx, table in enumerate(all_table_names):
                                st.markdown(f"<b>{idx+1}. {table}</b>", unsafe_allow_html=True)
                                st.session_state[f"{table}_duration"] = list(full_databases.current_database["tables"][table]["path"].keys())[0]
                                # st.radio("Choose duration:", options=list(), key=f'{table}_duration')
                                st.write("Choose time range:")

                                min_date = datetime(2024,4,1)
                                max_date = datetime(2024,4,30)

                                if from_time is None and to_time is None:
                                    from_time = min_date
                                    to_time = max_date
                                    st.warning("Cannot detect suitable time range from your question. Please select your expected time range.")
                                st.date_input("From", value=min(max(from_time, min_date), max_date), min_value=min_date, max_value=max_date, format="YYYY/MM/DD", key=f"{table}_from")
                                st.date_input("To", value=max(min_date, min(to_time, max_date)), min_value=st.session_state[f"{table}_from"], max_value=max_date, format="YYYY/MM/DD", key=f"{table}_to")
                                st.divider()
                            st.form_submit_button("Submit", help="Click this button after choosing expected time range.",
                                                  on_click=add_time_filter, args=(all_table_names,))
                            st.form_submit_button(agent_state.END_PROCESS_STATE["value"], help=agent_state.END_PROCESS_STATE["description"], on_click=renew_conversation)
                    if current_step.step_name == plan_generation:
                        if agent_state.get_local_tables("current_tables") is None:
                            st.warning("An error occurred while getting relevant tables. Please reload and try this question again.")
                            renew_conversation()

                        with st.spinner("Generating response"):
                            planner_full_msg = planner.run(sent_msg, message_type = msg_type,
                                                                     relevant_tables=agent_state.get_local_tables("current_tables"))

                        st.session_state.messages[current_step.step_name]["state_history"].append(planner_full_msg)
                        if "Final Answer" in str(planner_full_msg):
                            st.session_state.messages[current_step.step_name]["state_result"] = {"result": planner.current_solution}
                            processing_steps = st.session_state.user_question["full_process"].add_steps_from_plan(execution_plan=planner.current_solution)
                            current_step.is_done = True
                            st.session_state.user_question["full_process"].set_current_step(step=st.session_state.user_question["full_process"].get_step_by_name(step_name=data_preparation))
                            with st.spinner("Splitting the plan into execution steps."):
                                time.sleep(2)
                            st.session_state.continue_conversation = True
                            complete_current_step(next_step=data_preparation)
                        else:
                            st.rerun()

                    if current_step.step_name == data_preparation:
                        # there is an error while saving data
                        if planner.current_solution is None:
                            st.warning("An error occurred in generating execution plan.")
                            renew_conversation()
                        with st.spinner("Generating response"):
                            if sent_msg == st.session_state.user_question["content"]:
                                sent_msg = planner.current_solution["data_agent"].strip()
                            sent_msg += "\n* Finally, convert the result to Pandas dataframe if it is not dataframe, and assign the prepared dataset to 'result'."
                            data_agent_full_msg = data_agent.run(message=sent_msg, message_type=msg_type, selected_tables=agent_state.get_local_tables("current_tables"))
                        st.session_state.messages[current_step.step_name]["state_history"].append(copy.deepcopy(data_agent_full_msg))
                        if 'Final Answer' in str(data_agent_full_msg):
                            conclusion = str(data_agent_full_msg).split("Final Answer:")[-1].strip()
                            st.session_state.messages[current_step.step_name]["state_result"] = {"text": conclusion, "result": data_agent.get_last_result()}
                            complete_current_step()
                        else:
                            st.rerun()

                    if current_step.step_name in [data_visualization, data_analysis, data_modeling]:
                        try:
                            dataset = data_agent.get_last_result()
                            if dataset is None:
                                st.warning("Dataset is not initialized for this step.")
                            elif not isinstance(dataset, pd.DataFrame):
                                st.warning("Current dataset is not Pandas dataframe, cannot continue.")
                                st.write(dataset)
                                complete_current_step()
                            else:
                                used_agent = task_to_agent[current_step.step_name]
                                if sent_msg == st.session_state.user_question["content"]:
                                    sent_msg = planner.current_solution.get(used_agent.agent_name, "No required.").strip()
                                    msg_type = question_intent["category"]
                                with st.spinner("Generating response"):
                                    full_msg = used_agent.run(message=sent_msg, message_type=msg_type, selected_tables=agent_state.get_local_tables("current_tables"), dataset=dataset)
                                st.session_state.messages[current_step.step_name]["state_history"].append(full_msg)
                                if "Final Answer" in str(full_msg):
                                    conclusion = str(full_msg).split("Final Answer:")[-1].strip()
                                    st.session_state.messages[current_step.step_name]["state_result"] = {"text": conclusion, "result": used_agent.get_last_result()}
                                    complete_current_step()
                                else:
                                    st.rerun()

                        except Exception:
                            print(format_exc())
                            st.warning("An error occurred. Please reload page and try again.")
                            current_step.is_done = True
        else:
            last_process_state = list(st.session_state.messages.keys())[-1]
            apology_message = not_supported_msg(f"Since {question_intent['explanation'].lower().replace('.', '')}, this question is classified as {question_intent['category']} question")
            if "Apology" not in st.session_state.messages.keys():
                st.session_state.messages["Apology"] = {"state_history": [], "state_result": "Ignore"}
            st.session_state.messages["Apology"]["state_history"].append(apology_message)
            with st.chat_message("assistant"):
                empty = st.empty()
                apology_message.show(empty)

            renew_conversation()

        # print("APP SESSION MESSAGES")
        # for msg in st.session_state.messages:
        #     print(msg.role)
        #     print('----')
        #     msg.print_messages()
        #     print('----')
