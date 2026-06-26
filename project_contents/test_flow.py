from copy import deepcopy
import pandas as pd
import streamlit
from dotenv import load_dotenv
from streamlit_tree_select import tree_select
import os
# from common.di_config import INTERNAL_PROXY

load_dotenv("env")
# os.environ["http_proxy"]=INTERNAL_PROXY
# os.environ["https_proxy"]=INTERNAL_PROXY

from common.app_utils import AppData, page_prepare
from agents.agent_utils import *
from agents.agent_flow import *
from common.simple_model_utils import lgbm_model
from common.display_utils import show_model
# SPARK #
from pyspark.sql import SparkSession

# Open AI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_parquet("./public_datasets/00_Mekong_Delta/MekongDelta_01_tinh_TKQG.parquet")
st.dataframe(df.head(10))
# df = df.merge(pd.read_parquet("./public_datasets/00_Mekong_Delta/MekongDelta_00_tinh_DSLD.parquet"), on=["năm", "tỉnh"])

agent_name = "model_agent"
message = "build model to explain GRDP of each province"
msg_type = "Diagnostic"

if agent_name not in st.session_state:
    database = AppData()
    database.load_database(db_name="00_Mekong_Delta")
    selected_tables = {
        "target": {"MekongDelta_01_tỉnh_TKQG": ["năm", "tỉnh", "TKQG_tổng sản phẩm trên địa bàn theo giá hiện hành_triệu đồng"]},
        "factor": {}
    }
    st.session_state[agent_name] = init_agent(agent_type=agent_name, database=database.current_database)

    used_agent = st.session_state[agent_name]
    if used_agent != "planner":
        result = used_agent.run(message=message, message_type=msg_type, dataset=df.copy(), selected_tables=selected_tables)
        result.show(st.empty())

# page_prepare("Talk to your data")
# USED_AGENT = ['planner', 'data_agent', 'vis_agent', 'analysis_agent']
#
# if 'data_agent' not in st.session_state:
#     database = AppData()
#     database.load_database(db_name="00_Mekong_Delta")
#     for agent in USED_AGENT:
#         st.session_state[agent] = init_agent(agent_type=agent, database=database.current_database)
#
# st.button('Run Flow', key='agree')
# if 'agree' in st.session_state and st.session_state['agree']:
#     data_agent = st.session_state.data_agent
#     vis_agent = st.session_state.vis_agent
#     analysis_agent = st.session_state.analysis_agent
#
#     selected_tables = {
#         'target': {'MekongDelta_09_chi ngân sách địa phương': ['năm', 'tỉnh', 'loại chi ngân sách địa phương_lv1', 'loại chi ngân sách địa phương_lv2', 'loại chi ngân sách địa phương_lv3', 'TKQG_chi ngân sách nhà nước trên địa bàn_triệu đồng']},
#         'factor': {}
#     }
#
#     execution_plan = dedent("""
#     1. Read data: Read data from the table "MekongDelta_09_chi ngân sách địa phương".
#     2. Filter data:
#         - Filter the 'năm' in value ranges [2018, 2022].
#     3. Grouping - Aggregation:
#         - For frequent expenditure:
#             + Filter data: filter data with 'loại chi ngân sách địa phương_lv2' as 'chi thường xuyên' and 'loại chi ngân sách địa phương_lv3' as 'total'.
#             + Grouping - Aggregation: grouping data on 'năm', calculate the sum of 'TKQG_chi ngân sách nhà nước trên địa bàn_triệu đồng'.
#         - For expenditure on investment development:
#             + Filter data: filter data with 'loại chi ngân sách địa phương_lv2' as 'chi đầu tư phát triển' and 'loại chi ngân sách địa phương_lv3' as 'total'.
#             + Grouping - Aggregation: grouping data on 'năm', calculate the sum of 'TKQG_chi ngân sách nhà nước trên địa bàn_triệu đồng'.
#     4. Merge datasets: Merge the aggregated data for frequent expenditure and expenditure on investment development to obtain overall information for the Mekong Delta region for each year.
#     5. Perform analysis: No analysis required.
#     6. Visualization: Draw a grouped bar chart to show the frequent expenditure and development expenditure of each year.
#     """)
#
#     plan_info = split_execution_plan(execution_plan=execution_plan)
#     st.write(plan_info)
#
#     tasks = plan_to_tasks(execution_plan=plan_info.get('data_engineer'))
#     data_agent_msg = data_agent.run(message=plan_info.get('data_engineer'), message_type='Descriptive', selected_tables=selected_tables)
#
#     if data_agent_msg.messages[-1].object is not None:
#         dataset = data_agent_msg.messages[-1].object
#         if plan_info.get('data_visualizer', False):
#             st.info('Data Visualizer')
#             vis_agent_msg = vis_agent.run(message=plan_info.get('data_visualizer'), message_type='Descriptive', selected_tables=selected_tables, dataset=dataset)
#         if plan_info.get('data_analyst', False):
#             st.info('Data Analysis')
#             analysis_agent_msg = analysis_agent.run(message=plan_info.get('data_analyst'), message_type='Descriptive', selected_tables=selected_tables, dataset=dataset)