from copy import deepcopy
import pandas as pd

SUGGEST_RESPONSE = {
    'ok': 'OK, move to next step.'
}

END_PROCESS_STATE = {
    'key': 'end_process',
    'value': 'Finish, and try new question.',
    'description': 'Start new conversation.'
}

# For persisting data between agents
def init_states():
    global local_tables
    local_tables = {}

    global local_data
    local_data = {}

    global local_visual
    local_visual = {}

    global local_model
    local_model = {}

# For accessing the above global vars in different files
def get_local_data(name: str = "current_data"):
    global local_data
    return local_data.get(name, None)
    return local_data[name]

def get_local_tables(name:str="current_tables"):
    global local_tables
    return local_tables.get(name, None)

def set_local_tables(name:str="current_tables", tables:dict=None):
    global local_tables
    if name not in local_tables.keys():
        local_tables[name] = deepcopy(tables)

    # assign tables to current_tables for later use
    local_tables['current_tables'] = deepcopy(tables)

def set_local_data(name:str="current_data", data:pd.DataFrame=None):
    global local_data
    if name != "" and name not in local_data.keys():
        local_data[name] = data

    # assign data to current_data for later use
    local_data['current_data'] = data

def get_local_visual(name: str = "current_viz"):
    global local_visual
    return local_visual.get(name, None)

def set_local_visual(name:str="", viz_result:dict={}):
    global local_visual
    if name not in local_visual.keys():
        local_visual[name] = deepcopy(viz_result)
    # assign current use model
    local_visual['current_viz'] = deepcopy(viz_result)

def set_local_model(name:str="", model_result:dict={}):
    global local_model
    if name not in local_model.keys():
        local_model[name] = deepcopy(model_result)
    # assign current use model
    local_model['current_result'] = deepcopy(model_result)

def get_local_model(name: str = "current_result"):
    global local_model
    return local_model.get(name, None)