from typing import List, Union
from .chat_message import VALID_AGENTS
from .planner import Planner
from .execution_agent import DataAgent, VisAgent, AnalysisAgent, ModelAgent
from common.app_utils import Toc

# THIS MODULE IS USEFUL FOR THE WORKING FLOW OF AGENTS

# Agents Common
ENGINEER_MODEL = "gpt-3.5-turbo-1106"
ANALYST_MODEL = "gpt-3.5-turbo-1106"

def init_agent(agent_type:str="data_planner", database: dict = None):
    """A function to initialize the agent in app"""
    if database is None:
        raise ValueError(f"Database must be specified for agent initialization.")

    if agent_type == "planner":
        return Planner(database=database)
    elif agent_type == "data_agent":
        return DataAgent(database=database)
    elif agent_type == "vis_agent":
        return VisAgent(database=database)
    elif agent_type == "analysis_agent":
        return AnalysisAgent(database=database)
    elif agent_type == "model_agent":
        return ModelAgent(database=database)
    else:
        raise ValueError(f"Agent type must be specified as one of {VALID_AGENTS}. Now it is {agent_type}.")

# Process common
table_selection = "Table Selection"
time_filtering = "Time Filtering"
plan_generation = "Plan Generation"
data_preparation = "Data Preparation"
data_analysis = "Data Analysis"
data_visualization = "Data Visualization"
data_modeling = "Data Modeling"

class PROCESS_STEP:
    """Each process step an agent must work with"""
    step_info = {
        # Process common
        table_selection: {
            "agent": None,
            "description": "Select relevant tables to address the question.",
            "type": "selector"
        },
        time_filtering: {
            "description": "Select expected time range and duration.",
            "type": "selector"
        },
        plan_generation: {
            "description": "Generate a suitable execution plan to solve the question.",
            "type": "selector"
        },
        data_preparation: {
            "description": "Extract and process useful data from the selected tables.",
            "type": "execution"
        },
        data_analysis: {
            "description": "Conduct suitable analysis to provide answer.",
            "type": "execution"
        },
        data_visualization: {
            "description": "Draw charts to visualize the data.",
            "type": "execution"
        },
        data_modeling: {
            "description": "Build models to explain or predict specific features.",
            "type": "execution"
        }
    }

    def __init__(self, step_name:str="", step_description:str="No description."):
        if step_name is None or step_name == "":
            raise ValueError("Parameter 'step_name' cannot be empty.")

        self.step_name = step_name
        self.step_description = self.step_info.get(step_name, dict()).get("description", step_description)
        self.step_type = self.step_info.get(step_name, dict()).get("type", None)
        self.is_done = False

        self.html_string = f"""
        <div class="process-step">
            <p>
                <b>{self.step_name}</b>
                <br>
                <span>{self.step_description}</span>
            </p>
        </div>
        """

class PROCESS:
    """A Wrapper of Process Steps"""
    def __init__(self, process_steps: List[PROCESS_STEP] = []):
        self.process_steps = process_steps
        self.current_step = None
        self.toc = Toc()

    def init_selector_steps(self, include_time_filtering:bool=True):
        if include_time_filtering:
            self.process_steps = [
                PROCESS_STEP(step_name="Table Selection"),
                PROCESS_STEP(step_name="Time Filtering"),
                PROCESS_STEP(step_name="Plan Generation"),
            ]
        else:
            self.process_steps = [
                PROCESS_STEP(step_name="Table Selection"),
                PROCESS_STEP(step_name="Plan Generation")
            ]

    def get_html(self, step_type:str="selector"):
        if len(self.process_steps) == 0:
            raise AttributeError("Currently there is no steps in the process. Cannot generate the process.")

        html_str = ""
        for step in self.process_steps:
            if step.step_type == step_type:
                html_str += f"\n{step.html_string}"

        process_html = ""
        if html_str != "":
            process_html = f"""
            <div class="process-container">
            {html_str}
            """

        return process_html

    def get_step_by_index(self, index: int = 0):
        if index >= len(self.process_steps):
            raise IndexError(
                f"Index mus smaller than the number or steps in the process, which is {len(self.process_steps)}. Currently index = {index}.")

        extracted_step = self.process_steps[index]
        return extracted_step

    def get_step_by_name(self, step_name:str):
        for step in self.process_steps:
            if step.step_name == step_name:
                return step
        return None

    def get_incomplete_steps(self):
        return [step for step in self.process_steps if not step.is_done]

    def get_next_step(self):
        '''Get the first incompleted step'''
        for step in self.process_steps:
            if not step.is_done:
                return step
        return None

    def add_step(self, step:PROCESS_STEP):
        '''Add new step to the process'''
        self.process_steps.append(step)

    def add_steps(self, steps:List[PROCESS_STEP]):
        self.process_steps = self.process_steps + steps.copy()

    def set_current_step(self, step:Union[PROCESS_STEP, None]):
        self.current_step = step

    def add_steps_from_plan(self, execution_plan:dict):
        agent_to_step_name = {
            'data_agent': data_preparation,
            'analysis_agent': data_analysis,
            'vis_agent': data_visualization,
            'model_agent': data_modeling
        }
        for agent in execution_plan.keys():
            self.add_step(PROCESS_STEP(step_name=agent_to_step_name[agent]))

    def show_process(self):
        self.toc.remove_items()
        for step in self.process_steps:
            step_name = step.step_name
            if step.is_done:
                self.toc.add_item(text=step_name, status='done')
            elif self.current_step is not None and step.step_name == self.current_step.step_name:
                self.toc.add_item(text=step_name, status='in progress')
            else:
                self.toc.add_item(text=step_name, status='pending')

        self.toc.generate(title="Conversation Steps")
