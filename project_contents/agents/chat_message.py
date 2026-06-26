import re
from typing import Union, Optional, List
from textwrap import dedent

import pandas as pd
import streamlit as st

from common.chart_utils import get_report, get_pyg_html, convert_data_to_html, REPORT_HEIGHT
from common.display_utils import show_dictionary, show_model
from code_editor import code_editor
import streamlit.components.v1 as components
from traceback import format_exc
import copy

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

    if is_concat:
        old_message += new_message

    return old_message

# 'dataframe' and 'observation' are there for convenience in mechanically inserting additional content into a message.
VALID_AI_MESSAGE_TYPES = {
    "text",
    "solution", # code or plan, which is editable & result
}
VALID_MESSAGE_ROLES = {"user", "assistant"}
VALID_AGENTS = [
    "planner", "data_agent", "vis_agent", "analysis_agent", "model_agent"
]

class Message:
    """A message. Can be text, code, or code_result and their objects. Useful for conditional rendering in Streamlit."""
    def __init__(self, msg_type: str, content: str, object=None):
        """
        :param type: type of the response, can be text / code / code result: text, dataframe, or function result.
        :param content: content of the response
        """
        if msg_type not in VALID_AI_MESSAGE_TYPES:
            raise NotImplementedError(f"Unrecognised message type: {msg_type}. It must be one of {VALID_AI_MESSAGE_TYPES}")
        self.msg_type = msg_type
        self.content = content
        self.object = object
        self.is_modified = False # is this message modified before

    def __str__(self):
        if isinstance(self.msg_type, dict):
            # result from a function, only print the keys
            return self.content.keys()
        else:
            return self.content

    def show(self, in_progress: bool = False, editable=False, language:str="python"):
        """Display this section in the current Streamlit container.

        Parameters
        ----------
        in_progress: bool, default=False
            If true, will draw a cursor after the content.
        """
        language_map = {
            "text": "newest_plan",
            "python": "newest_code"
        }
        if self.msg_type == "solution": # Editable Message
            if editable and not self.is_modified:
                code_editor(code=dedent(self.content), lang=language, theme="dracula", options={"wrap": True}, allow_reset=True, key=language_map[language])
                if language == "text":
                    st.caption(f"Press Ctrl/CMD - Enter to save plan.")
                else:
                    st.caption(f"Press Ctrl/CMD - Enter to run code (Always assign the result to variable 'result'.")
            else:
                if language != "text":
                    with st.expander(f'Processing Code', expanded=True):
                        st.markdown(dedent(f'''<pre>
                        <code>
                        {self.content}
                        </code>
                        </pre>'''), unsafe_allow_html=True)
                else:
                    with st.container(border=True):
                        st.markdown(dedent(self.content))
        elif self.msg_type == "text":
            if in_progress:
                st.markdown(self.content + "▌")
            else:
                st.markdown(self.content, unsafe_allow_html=True)
        else:
            raise NotImplementedError(f"This type of message has no implemented display method: {self.msg_type}")

        if self.object is not None:
            if isinstance(self.object, pd.DataFrame):
                try:
                    st.write(f"The data has {self.object.shape[0]} rows. Here are the first {min(self.object.shape[0], 10)} rows:")
                    st.dataframe(self.object.head(10))
                except:
                    st.write(self.object.head(10))
            elif isinstance(self.object, dict):
                if "model" in self.object:
                    show_model(self.object)
                else:
                    show_dictionary(self.object, title="Current Result")
            else:
                try: # check if object is plotly chart
                    st.plotly_chart(self.object)
                except:
                    if isinstance(self.object, str):
                        st.text(self.object)
                    else:
                        st.write(self.object)

class FullMessage:
    """Wrapper of Message, specified by role (Assistant or User) & agent"""
    def __init__(
            self,
            role:str = None,
            agent:str = None,
            messages: List[Message]=[],
            in_progress:bool=False,
    ):
        if (role is not None) and (role not in VALID_MESSAGE_ROLES):
            raise NotImplementedError(f"Unrecognised message role: {role}. It must be one of {VALID_MESSAGE_ROLES}")
        self.role = role

        if (agent is not None) and (agent not in VALID_AGENTS):
            raise NotImplementedError(f"Unrecognised agent: {agent}. It must be one of {VALID_AGENTS}")
        self.agent = agent

        self.messages = messages
        self.__current_type = "text"
        self.in_progress = in_progress

        self.__tmp_mode = False
        self.__tmp_buffer = ""

    def __str__(self):
        full_msg_string = ""
        for msg in self.messages:
            if msg.msg_type == "text":
                full_msg_string += str(msg)
            elif msg.msg_type == "solution":
                full_msg_string += f'\n```python\n{str(msg)}```'
            else:
                full_msg_string += f"\n{str(msg)}"

        return full_msg_string

    def show(self, placeholder: st.empty, start_index:int=0):
        if start_index >= len(self.messages):
            self.print_messages()
            return
            # raise IndexError(f"Message list contains only {len(self.messages)} messages, which is smaller than start_index = {start_index}.")
        with placeholder.container():
            for msg in self.messages[start_index:]:
                msg.show()

    def add_delta(self, delta: str):
        # print(delta, end='')
        """Incrementally update this message with streaming deltas"""
        if self.role == "user":
            raise AttributeError("Cannot add new deltas to a user message")

        # Potential start / end of code block
        break_tmp_mode = False
        if '`' in delta or self.__tmp_mode:
            self.__tmp_mode = True
            self.__tmp_buffer += delta
            if self.__tmp_buffer.startswith('```python\n'):
                # code block
                self.__current_type = "solution"
                break_tmp_mode = True
            elif self.__tmp_buffer.startswith('```\n'):
                # return to text block
                self.__current_type = 'text'
                break_tmp_mode = True

        if not self.__tmp_mode:
            if len(self.messages) > 0 and self.messages[-1].msg_type == self.__current_type: # append msg to the last msg
                self.messages[-1].content += delta
            else:
                # add new msg_type
                self.messages.append(
                    Message(
                        msg_type=self.__current_type,
                        content=delta,
                    )
                )

        if break_tmp_mode:
            self.__tmp_mode = False
            self.__tmp_buffer = ""

    def print_messages(self):
        for msg in self.messages:
            print("Type:", msg.msg_type)
            print("Content:", msg.content)

    def concat(self, other: 'FullMessage'):
        if other.role != self.role:
            raise ValueError("Cannot concatenate messages of different roles")

        if self.messages[-1].msg_type == 'text' and other.messages[0].msg_type == 'text':
            new_msg = normalize_message(self.messages[-1].content, other.messages[0].content, is_concat=True)
            self.messages[-1].content = new_msg
            self.messages += other.messages[1:]
        else:
            self.messages += other.messages
        # State variables
        self.__current_type = other.__current_type
        self.__tmp_buffer = other.__tmp_buffer
        self.__tmp_mode = other.__tmp_mode
        self.agent = other.agent

        # Both side effect and normal return for convenience
        return self

def display_all_msg(msg_list:List['FullMessage']):
    """Display all messages list """
    for msg in msg_list:
        with st.chat_message(msg.role):
            empty = st.empty()
            msg.show(empty)

# User messages' contents should be a single string instead.
init_msg = FullMessage(
    role='assistant',
    messages = [
        Message(
            msg_type='text',
            content="Hello there! I'm your personal AI data analyst. You can ask me anything about the datasets in the left sidebar."
        )
    ]
)

not_supported_msg = lambda reason: FullMessage(
    role='assistant',
    messages = [
        Message(
            msg_type='text',
            content=f"{reason}. I am not implemented to support this type of question. Please try another question about the data."
        )
    ]
)

unavailable_agent_msg = FullMessage(
    role='assistant',
    messages = [
        Message(
            msg_type='text',
            content=f"This agent is not available at the moment. Please try another agents."
        )
    ]
)